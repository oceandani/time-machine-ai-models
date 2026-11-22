import io
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import HTTPException
from PIL import Image

from core.image_processor import (
    MAX_IDENTITY_COPY_RETENTION_CEILING,
    MIN_CORRIDOR_RETENTION_FLOOR,
    MIN_EFFECTIVE_REWRITE_MSE,
    PERCENTAGE_SCALE,
    REPAIR_GATE_TOLERANCE,
    build_targeted_repair_tensors,
    build_unified_global_tensors,
    calculate_rewrite_diff,
    composite_locked_regions,
    compute_candidate_clarity_score,
    crop_to_original,
    evaluate_geometric_compliance,
    generate_mask_diagnostic_artifacts,
    prepare_ipadapter_reference_image,
    process_and_standardize_image,
    select_single_priority_defect_zone,
    stabilize_archival_color_grading,
)
from core.job_manager import JobManager
from core.logging import logger
from services.inference_service import (
    ENABLE_IPADAPTER,
    FIXED_DEBUG_SEED,
    STRICT_VISUAL_REFERENCE_REQUIRED,
    run_comfyui_global_rebuild,
)
from services.vlm_service import (
    DEFAULT_ESTIMATED_STORIES,
    DEFAULT_FALLBACK_HEIGHT_RATIO,
    DEFAULT_LOW_RISE_MAX_STORIES,
    HEIGHT_ENVELOPE_TOLERANCE,
    LEGACY_COORD_SCALE,
    MIN_PASSING_CLARITY_SCORE,
    PERSPECTIVE_OUTER_ROOF_Y_RATIO,
    RATIO_MAX,
    BackdropSkyPolicy,
    PerspectivePolicy,
    VerticalMassPolicy,
    plan_historical_reconstruction,
    verify_reconstruction_dual_track,
)

load_dotenv()

# ==============================================================================
# 1. 頂層常數與前置「全彩日間寫實 + 行人/載具/建築史實鎖定」硬護欄 (零魔法數字散落)
# ==============================================================================
# 階段 A 隔離測試預設單輪 (MAX_CORRECTION_PASSES=1)；階段 B / 正式多輪模式可於 .env 設為 3
DEFAULT_MAX_CORRECTION_PASSES = max(
    1, int(os.getenv("MAX_CORRECTION_PASSES", "1"))
)

MAX_GLOBAL_PROMPT_WORD_BUDGET = 74
MAX_REPAIR_PROMPT_WORD_BUDGET = 56
QA_SCORE_WEIGHT = 0.75
CLARITY_SCORE_WEIGHT = 0.25
MAX_STORY_ERROR_SENTINEL = 999

HTTP_STATUS_UNPROCESSABLE_ENTITY = 422
HTTP_STATUS_INTERNAL_ERROR = 500
HTTP_STATUS_SERVICE_UNAVAILABLE = 503

DEFAULT_OUTER_WING_MAX_HEIGHT_RATIO = round(
    RATIO_MAX - PERSPECTIVE_OUTER_ROOF_Y_RATIO, 4
)

PHOTOREALISTIC_DOCUMENTARY_LEAD = (
    "full-color Kodachrome archival documentary photograph, vibrant natural daylight colors, "
    "realistic physical masonry and asphalt textures, sharp optical lens focus"
)

ARCHIVAL_DOCUMENTARY_QUALITY_NEG = (
    "monochrome, black and white, greyscale, grayscale, desaturated, sepia, night scene, dusk, dark sky, "
    "artificial blue mask, solid color block, flat digital overlay, dark enclosed courtyard, "
    "blind wall blocking street, cartoon, anime, illustration, painting, vector art, vintage poster, "
    "video game screenshot, 3D render, CGI, cel shading, stylized graphics, airbrushed, "
    "plastic smooth texture, toy vehicles, floating trams, modern 2020s cars, Ford Crown Victoria, "
    "modern LED bus, modern clothing, hoodies, backpacks, smartphones, distorted ghost pedestrians, "
    "narrow alley, ancient Chinese village, wooden shophouse, mud wall, brown atmospheric wash, "
    "grey washout, heavy fog, haze, blurry, modern glass skyscraper, glass curtain wall, LED screen, garbled text"
)

SKY_ISOLATION_NEGATIVE_PREFIX = (
    "buildings, skyscrapers, high-rise towers, spires, rooftops, architecture, billboards, neon signs, wires, "
)


def _deduplicate_and_budget_words(
    segments: List[str], max_words: int = MAX_GLOBAL_PROMPT_WORD_BUDGET
) -> str:
    unique_phrases: List[str] = []
    seen_lower = set()
    for seg in segments:
        for phrase in seg.split(","):
            clean_p = phrase.strip()
            if clean_p and clean_p.lower() not in seen_lower:
                seen_lower.add(clean_p.lower())
                unique_phrases.append(clean_p)

    joined = ", ".join(unique_phrases)
    words = joined.split()
    if len(words) <= max_words:
        return joined
    return " ".join(words[:max_words]).rstrip(",;")


# ==============================================================================
# 2. v10.5 字典序候選圖守門員 (防輪迴黑名單 #9：樓層誤差與史實合規優先，嚴禁同分劣幣覆蓋)
# ==============================================================================
class CandidateSnapshotManager:
    def __init__(self, initial_canvas_bytes: bytes):
        self.initial_canvas_bytes: bytes = initial_canvas_bytes
        self.best_candidate_bytes: Optional[bytes] = None
        self.best_qa_passed: bool = False
        self.best_qa_score: float = -1.0
        self.best_passing_zone_count: int = -1
        self.best_story_error: int = MAX_STORY_ERROR_SENTINEL
        self.best_composite_score: float = -1.0
        self.best_clarity_score: float = 0.0
        self.best_pass_num: int = 0
        self.best_has_valid_foundation: bool = False
        self.best_zone_verification_map: Dict[str, bool] = {}
        self.pass_history: List[Dict[str, Any]] = []

    def merge_and_check_all_zones_verified(
        self,
        is_global_pass: bool,
        repaired_zones: List[str],
        current_qa_zone_map: Dict[str, bool],
    ) -> Tuple[Dict[str, bool], List[str]]:
        if is_global_pass or not self.best_zone_verification_map:
            merged_map = dict(current_qa_zone_map)
        else:
            merged_map = dict(self.best_zone_verification_map)
            for z_name, z_passed in current_qa_zone_map.items():
                if z_name in repaired_zones or not z_passed:
                    merged_map[z_name] = z_passed

        unverified_zones = [
            z_name for z_name, passed in merged_map.items() if not passed
        ]
        return merged_map, unverified_zones

    def evaluate_and_update_candidate(
        self,
        pass_num: int,
        candidate_bytes: bytes,
        qa_score: float,
        passing_zone_count: int,
        story_error: int,
        clarity_score: float,
        repaired_zone_label: str,
        qa_passed: bool,
        has_valid_foundation: bool,
        merged_zone_map: Dict[str, bool],
        phase1_metrics: Dict[str, Any],
        execution_audit: Dict[str, Any],
    ) -> bool:
        composite_score = round(
            (qa_score * QA_SCORE_WEIGHT) + (clarity_score * CLARITY_SCORE_WEIGHT),
            2,
        )
        clarity_acceptable = clarity_score >= MIN_PASSING_CLARITY_SCORE

        if self.best_candidate_bytes is None:
            is_improved = True
        elif qa_passed and clarity_acceptable and not self.best_qa_passed:
            is_improved = True
        elif self.best_qa_passed and not (qa_passed and clarity_acceptable):
            is_improved = False
        else:
            if has_valid_foundation != self.best_has_valid_foundation:
                is_improved = has_valid_foundation > self.best_has_valid_foundation
            elif qa_score != self.best_qa_score:
                is_improved = qa_score > self.best_qa_score
            elif passing_zone_count != self.best_passing_zone_count:
                is_improved = passing_zone_count > self.best_passing_zone_count
            elif story_error != self.best_story_error:
                is_improved = story_error < self.best_story_error
            else:
                is_improved = clarity_score > self.best_clarity_score

        if is_improved:
            self.best_candidate_bytes = candidate_bytes
            self.best_qa_passed = qa_passed and clarity_acceptable
            self.best_qa_score = qa_score
            self.best_passing_zone_count = passing_zone_count
            self.best_story_error = story_error
            self.best_composite_score = composite_score
            self.best_clarity_score = clarity_score
            self.best_pass_num = pass_num
            self.best_has_valid_foundation = has_valid_foundation
            self.best_zone_verification_map = dict(merged_zone_map)

        self.pass_history.append(
            {
                "pass_num": pass_num,
                "target_zone": repaired_zone_label,
                "qa_passed": qa_passed,
                "has_valid_foundation": has_valid_foundation,
                "qa_score": qa_score,
                "passing_zone_count": passing_zone_count,
                "story_error": story_error,
                "clarity_score": clarity_score,
                "composite_score": composite_score,
                "unverified_zones": [
                    k for k, v in merged_zone_map.items() if not v
                ],
                "execution_audit": execution_audit,
                "phase1_verification_metrics": phase1_metrics,
                "accepted_as_best": is_improved,
            }
        )
        return is_improved

    def get_best_candidate_canvas(self) -> bytes:
        if self.best_candidate_bytes is not None:
            return self.best_candidate_bytes
        return self.initial_canvas_bytes


# ==============================================================================
# 3. 全區結構整合與前置全彩寫實鎖定編譯器 (SemanticRouter v10.5)
# ==============================================================================
class SemanticRouter:
    @classmethod
    def extract_human_confirmation_alerts(
        cls, plan_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        alerts = []
        for c in plan_data.get("region_contracts", []):
            if isinstance(c, dict) and (
                bool(c.get("requires_human_confirmation", False))
                or bool(c.get("uncertainty_flag", False))
            ):
                alerts.append(
                    {
                        "region_id": c.get("region_id"),
                        "topological_zone": c.get("topological_zone"),
                        "change_operator": c.get("change_operator"),
                        "evidence_ref_id": c.get("evidence_ref_id"),
                        "uncertainty_flag": c.get("uncertainty_flag"),
                        "uncertainty_reason": c.get("uncertainty_reason"),
                        "confidence": c.get("confidence"),
                    }
                )
        return alerts

    @classmethod
    def compile_global_synthesis_directives(
        cls,
        plan_data: Dict[str, Any],
        year: str,
        location: str,
        remediation_directives: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        visual_arch = plan_data.get("visual_archetype", {})
        policy_matrix = plan_data.get("policy_matrix", {})
        contracts = plan_data.get("region_contracts", [])
        color_palette = plan_data.get("verified_color_palette_prompt", "")

        master_arch = visual_arch.get(
            "architectural_prompt", f"authentic {year} {location}"
        )
        perspective_policy = policy_matrix.get("perspective_policy", "")
        backdrop_policy = policy_matrix.get("backdrop_sky_policy", "")

        facade_prompts: List[str] = []
        street_prompts: List[str] = []

        for c in contracts:
            if not isinstance(c, dict):
                continue
            t_zone = c.get("topological_zone", "")
            r_pos = c.get("regional_positive_prompt", "").strip()
            if not r_pos:
                continue
            if t_zone in ("LEFT_FACADE", "CENTRAL_BACKDROP", "RIGHT_FACADE"):
                facade_prompts.append(r_pos)
            elif t_zone in ("STREET_CORRIDOR", "WATER_AND_DOCKYARD"):
                street_prompts.append(r_pos)

        pos_parts: List[str] = [
            color_palette or PHOTOREALISTIC_DOCUMENTARY_LEAD,
            f"{year} {location}",
            master_arch,
        ]
        if facade_prompts:
            pos_parts.append(facade_prompts[0])

        if perspective_policy == PerspectivePolicy.LOCK_STREET_CORRIDOR.value:
            pos_parts.append("wide open multi-lane street corridor vista")
        if street_prompts:
            pos_parts.append(street_prompts[0])

        if backdrop_policy == BackdropSkyPolicy.RELEASE_UPPER_SKY.value:
            pos_parts.append(
                "open clear daylight blue sky with natural clouds above low-rise 5-story rooftops"
            )
        if remediation_directives:
            pos_parts.append(remediation_directives[0])

        compact_pos = _deduplicate_and_budget_words(
            pos_parts, MAX_GLOBAL_PROMPT_WORD_BUDGET
        )

        return {
            "positive_prompt": compact_pos,
            "negative_prompt": ARCHIVAL_DOCUMENTARY_QUALITY_NEG,
        }

    @classmethod
    def compile_targeted_repair_directives(
        cls,
        plan_data: Dict[str, Any],
        year: str,
        location: str,
        target_zones: List[str],
        remediation_directives: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        contracts = plan_data.get("region_contracts", [])
        visual_arch = plan_data.get("visual_archetype", {})
        color_palette = plan_data.get("verified_color_palette_prompt", "")

        if target_zones == ["SKY_CANVAS"]:
            sky_pos = (
                "full-color Kodachrome archival photograph, clear unobstructed bright blue daylight sky, "
                "natural atmospheric horizon, clean photographic sky texture"
            )
            return {
                "positive_prompt": sky_pos,
                "negative_prompt": SKY_ISOLATION_NEGATIVE_PREFIX
                + ARCHIVAL_DOCUMENTARY_QUALITY_NEG,
            }

        zone_prompts: List[str] = [
            color_palette or PHOTOREALISTIC_DOCUMENTARY_LEAD,
            f"authentic {year} {location}",
        ]

        for c in contracts:
            if (
                isinstance(c, dict)
                and c.get("topological_zone") in target_zones
                and c.get("topological_zone") != "SKY_CANVAS"
            ):
                r_pos = c.get("regional_positive_prompt", "").strip()
                if r_pos:
                    zone_prompts.append(r_pos)

        if "SKY_CANVAS" in target_zones:
            zone_prompts.append("clear unobstructed blue sky in upper frame")

        if len(zone_prompts) <= 2:
            zone_prompts.append(
                visual_arch.get(
                    "architectural_prompt",
                    f"period-accurate {year} {location} streetscape",
                )
            )

        if remediation_directives:
            zone_prompts.append(remediation_directives[0])

        return {
            "positive_prompt": _deduplicate_and_budget_words(
                zone_prompts, MAX_REPAIR_PROMPT_WORD_BUDGET
            ),
            "negative_prompt": ARCHIVAL_DOCUMENTARY_QUALITY_NEG,
        }

    @classmethod
    def determine_route(cls, semantic_data: Dict[str, Any]) -> Dict[str, Any]:
        directives = cls.compile_global_synthesis_directives(
            semantic_data, year="", location=""
        )
        protected_bboxes = [
            z["bbox"]
            for z in semantic_data.get("protected_zones", [])
            if isinstance(z, dict) and "bbox" in z
        ]
        return {
            "tier": 2,
            "target_bboxes": [
                [0, 0, LEGACY_COORD_SCALE, LEGACY_COORD_SCALE]
            ],
            "protected_bboxes": protected_bboxes,
            "strategy_name": semantic_data.get(
                "primary_archetype", "V10_5_TWO_STAGE_ISOLATED_GRAPH"
            ),
            "prompts": {
                "pos": directives["positive_prompt"],
                "neg": directives["negative_prompt"],
            },
            "geography": semantic_data.get("geography_profile", {}),
        }


# ==============================================================================
# 4. v10.5 兩階段隔離測試與閉環調度器 (AgentService)
# ==============================================================================
class AgentService:
    MAX_CORRECTION_PASSES = DEFAULT_MAX_CORRECTION_PASSES

    MAX_PROTECTED_LEAK_MSE = 3.5
    MIN_STRUCTURE_REWRITE_STD = 12.0
    MIN_SKY_REWRITE_STD = 4.0

    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager

    def _pre_generation_mask_and_geometry_gate(
        self,
        pass_num: int,
        mode_label: str,
        base_bytes: bytes,
        mask_bytes: bytes,
        ctrl_bytes: bytes,
        rewrite_ratio: float,
        allowed_max_repair_ratio: float,
        mask_stats: Dict[str, Any],
    ) -> None:
        with (
            Image.open(io.BytesIO(base_bytes)) as b_img,
            Image.open(io.BytesIO(mask_bytes)) as m_img,
            Image.open(io.BytesIO(ctrl_bytes)) as c_img,
        ):
            if b_img.size != m_img.size or b_img.size != c_img.size:
                raise ValueError(
                    f"[P2 前置檢查失敗] 張量尺寸不對齊: Base={b_img.size}, Mask={m_img.size}, Ctrl={c_img.size}"
                )

        if not bool(mask_stats.get("dimensions_aligned", False)):
            raise ValueError(
                f"[P2 遮罩審計失敗] 遮罩與原圖尺寸不一致: {mask_stats}"
            )

        if not bool(mask_stats.get("is_strict_binary", False)):
            raise ValueError(
                f"[防輪迴黑名單 #1 攔截] Pass {pass_num} 送入 ComfyUI 的遮罩含灰階洩漏 "
                f"({mask_stats.get('grey_1_to_254_leak_pct')}%)，拒絕執行！"
            )

        if pass_num > 1 and mode_label == "TRUE_TARGETED_REPAIR":
            if rewrite_ratio > (allowed_max_repair_ratio + REPAIR_GATE_TOLERANCE):
                raise RuntimeError(
                    f"[P1/P2 局部修補硬性斷言失敗] Pass {pass_num} 有效修補面積 ({rewrite_ratio * PERCENTAGE_SCALE:.1f}%) "
                    f"超出允許上限 ({allowed_max_repair_ratio * PERCENTAGE_SCALE:.1f}%)！"
                )

    async def run_time_machine_pipeline(
        self, job_id: str, year: str, location: str, image_bytes: bytes
    ) -> bytes:
        stage_label = (
            "階段 B (IPAdapter 歷史視覺對照模式)"
            if ENABLE_IPADAPTER
            else "階段 A (隔離測試：顯式停用 IPAdapter)"
        )
        logger.info(
            f"[{job_id}] 🚀 啟動時光機 v10.5 {stage_label} | "
            f"Passes={self.MAX_CORRECTION_PASSES} | FixedSeed={FIXED_DEBUG_SEED} | "
            f"StrictVisualRef={STRICT_VISUAL_REFERENCE_REQUIRED} | 目標: {location}, {year}"
        )

        # Stage 0: 無黑邊影像標準化 (cv2.BORDER_REPLICATE)
        original_standardized_bytes, metadata = process_and_standardize_image(
            image_bytes
        )
        self.job_manager.save_image(
            job_id, 0, "processed_input.png", original_standardized_bytes
        )

        snapshot_manager = CandidateSnapshotManager(original_standardized_bytes)

        # Stage 1: 歷史證據檢索與跨年代場景變化圖規劃 (Pass 0)
        logger.info(
            f"[{job_id}] 🧠 [Stage 1] 啟動歷史證據檢索與跨年代場景變化圖規劃..."
        )
        plan_eval = await plan_historical_reconstruction(
            original_standardized_bytes, year, location
        )
        raw_ref_image_path = plan_eval.pop("reference_image_path", None)
        reference_image_bytes = plan_eval.pop("reference_image_bytes", None)
        reference_is_monochrome = bool(
            plan_eval.pop("reference_is_monochrome", False)
        )
        color_reference_bytes = plan_eval.pop("color_reference_bytes", None)
        input_is_daylight = bool(plan_eval.pop("input_is_daylight", True))
        self.job_manager.save_metadata(job_id, 0, "qa_result.json", plan_eval)

        # 僅於 ENABLE_IPADAPTER=True (階段 B) 時才準備送入 Node 16 之參考圖，階段 A 嚴格為 None
        ipadapter_ref_path: Optional[str] = None
        if ENABLE_IPADAPTER:
            ipadapter_ref_path = raw_ref_image_path
            prepared_ipa_ref_bytes = prepare_ipadapter_reference_image(
                reference_image_bytes=reference_image_bytes,
                reference_is_monochrome=reference_is_monochrome,
                color_reference_bytes=color_reference_bytes,
                input_is_daylight=input_is_daylight,
            )
            if prepared_ipa_ref_bytes is not None:
                ipadapter_ref_path = self.job_manager.save_image(
                    job_id,
                    0,
                    "ipadapter_historical_ref.png",
                    prepared_ipa_ref_bytes,
                )

        plan_data = plan_eval.get("raw_semantic_data", {})
        plan_id = plan_data.get("plan_id", "PLAN_FALLBACK")
        evidence_ref_id = plan_data.get("evidence_ref_id", "EVID_UNKNOWN")
        allowed_max_stories = int(
            plan_data.get("max_historical_stories", DEFAULT_LOW_RISE_MAX_STORIES)
        )
        policy_matrix = plan_data.get("policy_matrix", {})
        geo_profile = plan_data.get("geography_profile", {})
        outer_wing_max_height = float(
            geo_profile.get(
                "outer_wing_max_height_ratio",
                DEFAULT_OUTER_WING_MAX_HEIGHT_RATIO,
            )
        )
        enforce_low_rise_truncation = (
            policy_matrix.get("backdrop_sky_policy")
            == BackdropSkyPolicy.RELEASE_UPPER_SKY.value
            or policy_matrix.get("vertical_mass_policy")
            == VerticalMassPolicy.TRUNCATE_TO_LOW_RISE.value
        )

        initial_error_level = plan_eval.get("error_level", "CONTENT_REJECT")

        if initial_error_level == "SYSTEM_FATAL":
            raise HTTPException(
                status_code=HTTP_STATUS_INTERNAL_ERROR,
                detail="系統視覺規劃大腦配置錯誤",
            )

        if initial_error_level == "SYSTEM_WARNING_SKIP" or not plan_data:
            raise HTTPException(
                status_code=HTTP_STATUS_SERVICE_UNAVAILABLE,
                detail="Stage 1 歷史規劃服務暫時無法回應，請稍後重試。",
            )

        if plan_eval.get("pass", False) and initial_error_level == "NONE":
            logger.info(
                f"[{job_id}] ✅ Stage 1 判定輸入影像已符合 {year} {location} 史實特徵，直接輸出。"
            )
            return crop_to_original(original_standardized_bytes, metadata)

        human_alerts = SemanticRouter.extract_human_confirmation_alerts(plan_data)
        if human_alerts:
            self.job_manager.save_metadata(
                job_id,
                0,
                "human_confirmation_alerts.json",
                {"alerts": human_alerts},
            )

        remediation_directives: List[str] = []
        failed_region_ids: List[str] = []
        failed_topological_zones: List[str] = []
        verified_passing_zones: List[str] = []
        already_repaired_zones: List[str] = []
        primary_defect_zone: str = "NONE"
        global_retry_count: int = 0
        last_stories_exceeded: bool = False

        final_qa_passed: bool = False
        last_qa_eval: Dict[str, Any] = {}
        last_execution_audit: Dict[str, Any] = {}
        last_mask_stats: Dict[str, Any] = {}

        # Stage 2, 3 & 4:
        for pass_num in range(1, self.MAX_CORRECTION_PASSES + 1):
            if (
                pass_num == 1
                or snapshot_manager.best_candidate_bytes is None
                or not snapshot_manager.best_has_valid_foundation
            ):
                mode_label = "ANCHORED_GLOBAL_CANDIDATE"
                is_global_pass = True
                adaptive_feedback = {
                    "retry_index": global_retry_count,
                    "stories_exceeded": last_stories_exceeded,
                    "verified_passing_zones": verified_passing_zones,
                }
                directives = SemanticRouter.compile_global_synthesis_directives(
                    plan_data=plan_data,
                    year=year,
                    location=location,
                    remediation_directives=remediation_directives,
                )
                tensors = build_unified_global_tensors(
                    image_bytes=original_standardized_bytes,
                    plan_data=plan_data,
                    adaptive_feedback=adaptive_feedback,
                    best_candidate_bytes=snapshot_manager.best_candidate_bytes,
                )
                base_for_composite = (
                    snapshot_manager.best_candidate_bytes
                    if (
                        global_retry_count > 0
                        and snapshot_manager.best_candidate_bytes is not None
                    )
                    else original_standardized_bytes
                )
                global_retry_count += 1
            else:
                mode_label = "TRUE_TARGETED_REPAIR"
                is_global_pass = False
                candidate_base = snapshot_manager.get_best_candidate_canvas()
                selected_zone = select_single_priority_defect_zone(
                    primary_defect_zone=primary_defect_zone,
                    failed_topological_zones=failed_topological_zones,
                    already_repaired_zones=already_repaired_zones,
                )
                tensors = build_targeted_repair_tensors(
                    current_canvas_bytes=candidate_base,
                    original_bytes=original_standardized_bytes,
                    plan_data=plan_data,
                    failed_region_ids=failed_region_ids,
                    failed_topological_zones=failed_topological_zones,
                    target_single_zone=selected_zone,
                )
                repaired_now = tensors["repaired_topological_zones"]
                for rz in repaired_now:
                    if rz not in already_repaired_zones:
                        already_repaired_zones.append(rz)
                directives = SemanticRouter.compile_targeted_repair_directives(
                    plan_data=plan_data,
                    year=year,
                    location=location,
                    target_zones=repaired_now,
                    remediation_directives=remediation_directives,
                )
                base_for_composite = candidate_base

            prepared_base_bytes = tensors["prepared_base_canvas_bytes"]
            inpaint_mask_bytes = tensors["inpaint_noise_mask_bytes"]
            ctrl_map_bytes = tensors["structural_control_map_bytes"]
            lock_mask_bytes = tensors["composite_protection_mask_bytes"]
            effective_denoise = tensors["recommended_denoise"]
            effective_ctrl_strength = tensors["recommended_controlnet_strength"]
            effective_ctrl_end_pct = tensors["recommended_controlnet_end_percent"]
            effective_ipa_weight = tensors["recommended_ipadapter_weight"]
            effective_ipa_end_at = tensors["recommended_ipadapter_end_at"]
            rewrite_ratio = tensors["rewrite_coverage_ratio"]
            protected_ratio = tensors["protected_coverage_ratio"]
            allowed_max_repair_ratio = tensors["allowed_max_repair_ratio"]
            repaired_zones = tensors["repaired_topological_zones"]
            target_zone_label = ",".join(repaired_zones)

            # 階段 A 專用：生成並存出可視化遮罩疊加圖與純二值／重疊審計報表
            mask_overlay_bytes, mask_stats = generate_mask_diagnostic_artifacts(
                base_canvas_bytes=prepared_base_bytes,
                inpaint_noise_mask_bytes=inpaint_mask_bytes,
                composite_protection_mask_bytes=lock_mask_bytes,
                plan_data=plan_data,
            )
            last_mask_stats = mask_stats
            self.job_manager.save_image(
                job_id, pass_num, "debug_mask_overlay.png", mask_overlay_bytes
            )
            self.job_manager.save_metadata(
                job_id, pass_num, "debug_mask_stats.json", mask_stats
            )
            logger.info(
                f"🔍 [Pass {pass_num} 遮罩可視化與二值審計] 尺寸對齊={mask_stats.get('dimensions_aligned')} | "
                f"純二值={mask_stats.get('is_strict_binary')} | "
                f"白(重繪255)={mask_stats.get('white_rewrite_255_pct')}% | "
                f"黑(鎖定0)={mask_stats.get('black_lock_0_pct')}% | "
                f"灰階洩漏={mask_stats.get('grey_1_to_254_leak_pct')}% | "
                f"分區重疊={mask_stats.get('polygon_overlap_pct')}%"
            )

            self._pre_generation_mask_and_geometry_gate(
                pass_num=pass_num,
                mode_label=mode_label,
                base_bytes=prepared_base_bytes,
                mask_bytes=inpaint_mask_bytes,
                ctrl_bytes=ctrl_map_bytes,
                rewrite_ratio=rewrite_ratio,
                allowed_max_repair_ratio=allowed_max_repair_ratio,
                mask_stats=mask_stats,
            )

            base_path = self.job_manager.save_image(
                job_id, pass_num, "base_input.png", prepared_base_bytes
            )
            inpaint_mask_path = self.job_manager.save_image(
                job_id, pass_num, "inpaint_noise_mask.png", inpaint_mask_bytes
            )
            ctrl_map_path = self.job_manager.save_image(
                job_id, pass_num, "structural_control_map.png", ctrl_map_bytes
            )
            self.job_manager.save_image(
                job_id, pass_num, "composite_lock_mask.png", lock_mask_bytes
            )

            logger.info(
                f"[{job_id}] 🎨 [Stage 3 {mode_label}] Pass {pass_num}/{self.MAX_CORRECTION_PASSES} | "
                f"TargetZones: {repaired_zones} | 白(重繪): {rewrite_ratio * PERCENTAGE_SCALE:.1f}% | "
                f"黑(鎖定): {protected_ratio * PERCENTAGE_SCALE:.1f}% | Denoise: {effective_denoise:.2f} | "
                f"CtrlStrength: {effective_ctrl_strength:.2f} (EndPct: {effective_ctrl_end_pct:.2f}) | "
                f"IPAdapterEnabled={ENABLE_IPADAPTER}"
            )

            comfy_result = run_comfyui_global_rebuild(
                year=year,
                location=location,
                base_image_path=base_path,
                mask_image_path=inpaint_mask_path,
                denoise=effective_denoise,
                controlnet_strength=effective_ctrl_strength,
                pos_prompt=directives["positive_prompt"],
                neg_prompt=directives["negative_prompt"],
                controlnet_image_path=ctrl_map_path,
                controlnet_end_percent=effective_ctrl_end_pct,
                reference_image_path=ipadapter_ref_path,
                ipadapter_weight=effective_ipa_weight,
                ipadapter_end_at=effective_ipa_end_at,
            )

            raw_gen_bytes = comfy_result["image_bytes"]
            seed_used = int(comfy_result["seed_used"])
            historical_reference_applied = bool(
                comfy_result["historical_reference_applied"]
            )
            ipadapter_status = str(comfy_result["ipadapter_status"])
            execution_audit = {
                "seed_used": seed_used,
                "prompt_id": comfy_result["prompt_id"],
                "ipadapter_requested": comfy_result["ipadapter_requested"],
                "historical_reference_applied": historical_reference_applied,
                "ipadapter_status": ipadapter_status,
                "executed_nodes": comfy_result["executed_nodes"],
                "missing_models": comfy_result["missing_models"],
                "reference_image_uploaded": comfy_result[
                    "reference_image_uploaded"
                ],
            }
            last_execution_audit = execution_audit

            # 執行自動色階去霧、日間明度守恆與懷舊彩色菲林色譜對齊
            color_stabilized_gen_bytes = stabilize_archival_color_grading(
                target_image_bytes=raw_gen_bytes,
                color_reference_bytes=color_reference_bytes,
                original_image_bytes=original_standardized_bytes,
                input_is_daylight=input_is_daylight,
            )

            _, raw_vae_drift_mse, _ = calculate_rewrite_diff(
                base_for_composite,
                color_stabilized_gen_bytes,
                inpaint_mask_bytes,
            )

            composited_bytes = composite_locked_regions(
                original_bytes=original_standardized_bytes,
                generated_bytes=color_stabilized_gen_bytes,
                composite_protection_mask_bytes=lock_mask_bytes,
                base_canvas_bytes=base_for_composite,
                inpaint_noise_mask_bytes=inpaint_mask_bytes,
            )

            # 雙基準物理驗證：重繪幅度對照現代原圖，鎖定區零洩漏對照當輪鎖定基底
            rewrite_mse, _, rewrite_std = calculate_rewrite_diff(
                original_standardized_bytes, composited_bytes, inpaint_mask_bytes
            )
            _, post_protected_mse, _ = calculate_rewrite_diff(
                base_for_composite, composited_bytes, inpaint_mask_bytes
            )

            protected_mse_str = (
                "N/A (全圖四算子重構，0% 鎖定區)"
                if post_protected_mse < 0
                else (
                    f"{post_protected_mse:.2f} "
                    f"(合成前VAE漂移={raw_vae_drift_mse:.2f}, 黑鎖定佔比={protected_ratio * PERCENTAGE_SCALE:.1f}%)"
                )
            )
            logger.info(
                f"📊 [Pass {pass_num} Micro-Gate] Seed={seed_used} | "
                f"相對於原圖之核心重繪 MSE: {rewrite_mse:.2f} (最低門檻={MIN_EFFECTIVE_REWRITE_MSE}) | "
                f"鎖定保護區 MSE: {protected_mse_str} | "
                f"核心重繪 StdDev: {rewrite_std:.2f}"
            )

            min_required_std = (
                self.MIN_SKY_REWRITE_STD
                if repaired_zones == ["SKY_CANVAS"]
                else self.MIN_STRUCTURE_REWRITE_STD
            )

            if (
                post_protected_mse >= 0
                and post_protected_mse > self.MAX_PROTECTED_LEAK_MSE
            ) or rewrite_std < min_required_std:
                self.job_manager.save_image(
                    job_id,
                    pass_num,
                    "synthesized_output_microgate_rejected.png",
                    composited_bytes,
                )
                logger.error(
                    f"🚨 [Pass {pass_num} Micro-Gate 攔截] 保護區洩漏或重繪標準差過低，已另存法證圖並捨棄本輪輸出。"
                )
                continue

            self.job_manager.save_image(
                job_id, pass_num, "synthesized_output.png", composited_bytes
            )

            logger.info(
                f"[{job_id}] ⚖️ [Stage 4] 啟動三圖對照盲測取證 + 逐分區反原圖複製審計 (Pass {pass_num})..."
            )

            cv_compliance = evaluate_geometric_compliance(
                generated_bytes=composited_bytes,
                plan_data=plan_data,
                original_bytes=original_standardized_bytes,
            )
            clarity_score = compute_candidate_clarity_score(composited_bytes)
            logger.info(
                f"📐 [Stage 4 CV 物理量測] 天際線邊緣合規: {cv_compliance['cv_sky_compliant']} "
                f"(頂部邊緣密度={cv_compliance['upper_edge_density']}) | "
                f"逐區重繪幅度: {cv_compliance.get('zone_rewrite_diffs', {})} | "
                f"未重繪殘留分區: {cv_compliance.get('under_restyled_zones', [])} | "
                f"走廊結構保留度: {cv_compliance['corridor_structure_retention']:.2f} | "
                f"非天空區清晰度: {clarity_score:.1f}/100"
            )

            qa_eval = await verify_reconstruction_dual_track(
                original_image_bytes=original_standardized_bytes,
                generated_image_bytes=composited_bytes,
                year=year,
                location=location,
                plan_data=plan_data,
                cv_compliance=cv_compliance,
                clarity_score=clarity_score,
                reference_image_bytes=reference_image_bytes,
                historical_reference_applied=historical_reference_applied,
            )
            qa_eval["execution_audit"] = execution_audit
            qa_eval["mask_audit_stats"] = mask_stats

            current_zone_map = qa_eval.get("zone_verification_map", {})
            merged_zone_map, unverified_zones = (
                snapshot_manager.merge_and_check_all_zones_verified(
                    is_global_pass=is_global_pass,
                    repaired_zones=repaired_zones,
                    current_qa_zone_map=current_zone_map,
                )
            )

            if unverified_zones and bool(qa_eval.get("pass", False)):
                logger.warning(
                    f"🛡️ [分區合格矩陣攔截] 局部修補後仍有未達標分區 {unverified_zones}，拒絕放行全圖！"
                )
                qa_eval["pass"] = False
                qa_eval["error_level"] = "CONTENT_REJECT"
                qa_eval["failed_topological_zones"] = unverified_zones
                qa_eval["primary_defect_zone"] = unverified_zones[0]

            pass_verdict = (
                bool(qa_eval.get("pass", False))
                and (qa_eval.get("error_level") == "NONE")
                and (len(unverified_zones) == 0)
            )
            qa_score = float(qa_eval.get("qa_compliance_score", 0.0))
            passing_zone_count = sum(
                1 for passed in merged_zone_map.values() if passed
            )
            has_valid_foundation = bool(
                qa_eval.get("has_valid_global_foundation", False)
            )
            estimated_stories = int(
                qa_eval.get(
                    "estimated_max_building_stories", DEFAULT_ESTIMATED_STORIES
                )
            )
            measured_height = float(
                qa_eval.get(
                    "measured_building_height_ratio",
                    DEFAULT_FALLBACK_HEIGHT_RATIO,
                )
            )
            story_error = (
                max(0, estimated_stories - allowed_max_stories)
                if enforce_low_rise_truncation
                else 0
            )
            last_stories_exceeded = enforce_low_rise_truncation and (
                estimated_stories > allowed_max_stories
                or measured_height
                > (outer_wing_max_height + HEIGHT_ENVELOPE_TOLERANCE)
            )

            verified_passing_zones = [
                z_k for z_k, z_ok in merged_zone_map.items() if z_ok
            ]

            dual_qa_detail = qa_eval.get("raw_semantic_data", {}).get(
                "dual_track_qa", {}
            )
            s_id_audit = dual_qa_detail.get("scene_identity_audit", {})
            t1_audit = dual_qa_detail.get("track1_audit", {})
            t2_audit = dual_qa_detail.get("track2_audit", {})

            under_restyled = cv_compliance.get("under_restyled_zones", [])
            genuinely_rewritten = (
                rewrite_mse >= MIN_EFFECTIVE_REWRITE_MSE
                and len(under_restyled) == 0
            )

            v1_passed = bool(
                t1_audit.get("perspective_and_corridor_aligned", False)
            ) and bool(s_id_audit.get("road_scale_and_width_preserved", False))

            v2_passed = genuinely_rewritten and (
                (
                    bool(cv_compliance.get("cv_sky_compliant", False))
                    and not last_stories_exceeded
                )
                if enforce_low_rise_truncation
                else bool(t1_audit.get("envelope_and_sky_compliant", False))
            )

            if post_protected_mse >= 0.0:
                v3_passed = (
                    post_protected_mse <= self.MAX_PROTECTED_LEAK_MSE
                    and genuinely_rewritten
                )
            else:
                v3_passed = genuinely_rewritten

            v4_score = float(cv_compliance["corridor_structure_retention"])
            v4_passed = (
                MIN_CORRIDOR_RETENTION_FLOOR
                <= v4_score
                <= MAX_IDENTITY_COPY_RETENTION_CEILING
                and genuinely_rewritten
            )

            v5_visual_entities_ok = (
                bool(t2_audit.get("architectural_typology_verified", False))
                and bool(t2_audit.get("signage_and_vehicles_valid", False))
                and bool(qa_eval.get("vehicles_realistic", False))
                and bool(qa_eval.get("pedestrians_authentic", False))
            )
            # 嚴格原則：若未在生成器實際執行 IPAdapter 注入，絕不把 V5 歷史參考視覺驗證標為 True
            v5_passed = v5_visual_entities_ok and (
                historical_reference_applied
                if reference_image_bytes is not None
                else True
            )

            phase1_metrics = {
                "v1_landmarks_anchored": v1_passed,
                "v2_operators_effective": v2_passed,
                "v3_targeted_zone_control": v3_passed,
                "v4_corridor_structure_retention": v4_score,
                "v4_corridor_retained_not_copied": v4_passed,
                "v5_visual_entities_verified": v5_visual_entities_ok,
                "v5_historical_reference_applied": historical_reference_applied,
                "v5_evidence_and_entities_verified": v5_passed,
                "v5_evidence_linked_id": evidence_ref_id,
            }
            logger.info(
                f"📋 [Phase 1 五大真實結果指標] V1走廊地標配準={v1_passed} | "
                f"V2四算子真實重繪={v2_passed} (MSE={rewrite_mse:.1f}) | "
                f"V3分區受控={v3_passed} (黑鎖定={protected_ratio * PERCENTAGE_SCALE:.1f}%, 保護MSE={post_protected_mse:.2f}) | "
                f"V4結構保留且非照抄={v4_passed} ({v4_score:.2f}, 區間={MIN_CORRIDOR_RETENTION_FLOOR}~{MAX_IDENTITY_COPY_RETENTION_CEILING}) | "
                f"V5史實視覺驗證={v5_passed} (RefApplied={historical_reference_applied}, {ipadapter_status})"
            )

            is_best = snapshot_manager.evaluate_and_update_candidate(
                pass_num=pass_num,
                candidate_bytes=composited_bytes,
                qa_score=qa_score,
                passing_zone_count=passing_zone_count,
                story_error=story_error,
                clarity_score=clarity_score,
                repaired_zone_label=target_zone_label,
                qa_passed=pass_verdict,
                has_valid_foundation=has_valid_foundation,
                merged_zone_map=merged_zone_map,
                phase1_metrics=phase1_metrics,
                execution_audit=execution_audit,
            )
            logger.info(
                f"🏆 [字典序防退化守門員] Pass {pass_num} QA得分={qa_score:.2f} "
                f"(合格分區={passing_zone_count}/5, 樓層誤差={story_error}, 清晰度={clarity_score:.1f}) -> "
                f"{'✅ 更新為最佳候選圖' if is_best else f'⏪ 拒絕退化，保留 Pass {snapshot_manager.best_pass_num} (QA={snapshot_manager.best_qa_score:.2f}, 誤差={snapshot_manager.best_story_error})'}"
            )

            self.job_manager.save_metadata(
                job_id, pass_num, "qa_result.json", qa_eval
            )
            last_qa_eval = qa_eval

            if pass_verdict:
                final_qa_passed = True
                logger.info(
                    f"[{job_id}] ✅ [Stage 4 三圖對照取證完全通過] Phase 1 五大驗證指標與建築/車輛/行人史實寫實度均合規！(Plan ID: {plan_id})"
                )
                break

            if pass_num < self.MAX_CORRECTION_PASSES:
                failed_topological_zones = list(
                    dict.fromkeys(
                        unverified_zones
                        + qa_eval.get("failed_topological_zones", [])
                    )
                )
                primary_defect_zone = select_single_priority_defect_zone(
                    primary_defect_zone=qa_eval.get(
                        "primary_defect_zone", "NONE"
                    ),
                    failed_topological_zones=failed_topological_zones,
                    already_repaired_zones=already_repaired_zones,
                )
                failed_region_ids = qa_eval.get("failed_region_ids", [])
                remediation_directives = list(
                    dual_qa_detail.get("remediation_directives", [])
                )

        # ======================================================================
        # P0 嚴格終局狀態管理
        # ======================================================================
        best_canvas_bytes = snapshot_manager.get_best_candidate_canvas()
        final_cropped_bytes = crop_to_original(best_canvas_bytes, metadata)

        if not final_qa_passed:
            rejected_path = self.job_manager.save_image(
                job_id,
                self.MAX_CORRECTION_PASSES,
                "rejected_best_candidate.png",
                final_cropped_bytes,
            )
            rejection_report = {
                "status": "QA_REJECTED_NEEDS_HUMAN_REVIEW",
                "job_id": job_id,
                "plan_id": plan_id,
                "evidence_ref_id": evidence_ref_id,
                "best_candidate_pass": snapshot_manager.best_pass_num,
                "best_qa_score": snapshot_manager.best_qa_score,
                "best_story_error": snapshot_manager.best_story_error,
                "best_composite_score": snapshot_manager.best_composite_score,
                "rejected_candidate_saved_path": rejected_path,
                "historical_reference_applied": last_execution_audit.get(
                    "historical_reference_applied", False
                ),
                "ipadapter_status": last_execution_audit.get(
                    "ipadapter_status", "UNKNOWN"
                ),
                "seed_used": last_execution_audit.get(
                    "seed_used", FIXED_DEBUG_SEED
                ),
                "mask_audit_stats": last_mask_stats,
                "is_photorealistic": last_qa_eval.get(
                    "is_photorealistic", False
                ),
                "vehicles_realistic": last_qa_eval.get(
                    "vehicles_realistic", False
                ),
                "pedestrians_authentic": last_qa_eval.get(
                    "pedestrians_authentic", False
                ),
                "pass_scene_identity": last_qa_eval.get(
                    "pass_scene_identity", False
                ),
                "zone_verification_map": snapshot_manager.best_zone_verification_map,
                "failed_topological_zones": last_qa_eval.get(
                    "failed_topological_zones", []
                ),
                "qa_failure_reason": last_qa_eval.get(
                    "reason", "未能通過三圖對照視覺取證與史實照片驗收"
                ),
                "pass_history": snapshot_manager.pass_history,
            }
            self.job_manager.save_metadata(
                job_id,
                self.MAX_CORRECTION_PASSES,
                "final_rejection_report.json",
                rejection_report,
            )
            logger.error(
                f"[{job_id}] ❌ [P0 嚴格狀態攔截] 經歷 {self.MAX_CORRECTION_PASSES} 輪後 QA 仍判定 REJECT！"
                f"已儲存最佳候選圖至 {rejected_path}，拒絕回傳 200 OK，改拋出 {HTTP_STATUS_UNPROCESSABLE_ENTITY} 驗收失敗報告。"
            )
            raise HTTPException(
                status_code=HTTP_STATUS_UNPROCESSABLE_ENTITY,
                detail=rejection_report,
            )

        logger.info(
            f"[{job_id}] 🏁 [200 OK 驗收合格] 回傳 Pass {snapshot_manager.best_pass_num} 通過驗收之最終歷史重構影像。"
        )
        return final_cropped_bytes