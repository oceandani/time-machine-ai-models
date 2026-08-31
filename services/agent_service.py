import io
from typing import List, Dict, Any

from PIL import Image
from fastapi import HTTPException

from core.logging import logger
from core.image_processor import process_and_standardize_image, create_mask_from_bboxes, crop_to_original
from core.job_manager import JobManager
from services.inference_service import run_comfyui_inference, run_comfyui_inpaint, run_comfyui_global_rebuild
from services.vlm_service import evaluate_image

class SemanticRouter:
    TIER_2_ACTIONS = [
        "remove_building_or_mass",
        "rebuild_landform_or_fill",
        "rebuild_large_building_group"
    ]
    
    TIER_2_MIN_CONFIDENCE = 0.85
    TIER_2_MIN_AREA_PERCENT = 0.25

    @staticmethod
    def calculate_bbox_area_percent(bbox: List[int]) -> float:
        y_min, x_min, y_max, x_max = bbox
        area = (y_max - y_min) * (x_max - x_min)
        total_area = 1000 * 1000
        return area / total_area

    @classmethod
    def determine_route(cls, semantic_data: Dict[str, Any]) -> Dict[str, Any]:
        regions = semantic_data.get("regions", [])
        tier2_triggered = False
        tier2_bboxes = []
        tier1_bboxes = []

        logger.info("🧭 [Router] 啟動語義路由分析...")

        for region in regions:
            action = region.get("action")
            conf = region.get("confidence", 0.0)
            bbox = region.get("bbox")

            if action == "preserve":
                continue

            if action in cls.TIER_2_ACTIONS:
                area_pct = cls.calculate_bbox_area_percent(bbox)
                logger.info(f"📐 [Router] 偵測到高危動作 '{action}' | 信心: {conf} | 面積佔比: {area_pct:.1%}")
                
                if conf >= cls.TIER_2_MIN_CONFIDENCE and area_pct >= cls.TIER_2_MIN_AREA_PERCENT:
                    logger.warning("⚠️ [Router] 突破雙重門檻！鎖定為 Level 2 全域重構。")
                    tier2_triggered = True
                    tier2_bboxes.append(bbox)
                else:
                    logger.info("🛡️ [Router] 門檻不足 (降級防禦生效)。降級為 Level 1 局部處理。")
                    tier1_bboxes.append(bbox)
            else:
                tier1_bboxes.append(bbox)

        if tier2_triggered:
            return {"tier": 2, "target_bboxes": tier2_bboxes, "strategy_name": "GLOBAL_REBUILD_LEVEL_2"}
        elif tier1_bboxes:
            return {"tier": 1, "target_bboxes": tier1_bboxes, "strategy_name": "LOCAL_INPAINT_LEVEL_1"}
        else:
            return {"tier": 0, "target_bboxes": [], "strategy_name": "PRESERVE_ALL"}

class AgentService:
    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager

    async def run_time_machine_pipeline(self, job_id: str, year: str, location: str, image_bytes: bytes) -> bytes:
        logger.info(f"[{job_id}] 🚀 開始執行時光機 Agentic Pipeline...")
        
        processed_bytes, metadata = process_and_standardize_image(image_bytes)
        current_image_path = self.job_manager.save_image(job_id, 0, "processed_input.jpg", processed_bytes)
        
        logger.info(f"[{job_id}] 🎨 執行初始生圖 (Pass 0)...")
        current_image_bytes = run_comfyui_inference(year, location, current_image_path)
        
        MAX_CORRECTION_PASSES = 3
        
        for pass_num in range(MAX_CORRECTION_PASSES):
            logger.info(f"[{job_id}] 🧠 進入 VQA 評估階段 (Pass {pass_num})...")
            eval_result = await evaluate_image(current_image_bytes, year, location)
            
            self.job_manager.save_metadata(job_id, pass_num, "qa_result.json", eval_result)
            error_level = eval_result.get("error_level", "SYSTEM_WARNING_SKIP")
            
            if error_level == "NONE":
                logger.info(f"[{job_id}] ✅ 內容評估 [通過]: 圖片符合歷史事實！")
                break
                
            elif error_level == "SYSTEM_WARNING_SKIP":
                logger.warning(f"[{job_id}] ⚠️ VLM 暫時不可用，觸發優雅降級，直接回傳相片。")
                break
                
            elif error_level == "SYSTEM_FATAL":
                logger.error(f"[{job_id}] 🚨 處理發生嚴重錯誤: API 權限異常")
                raise HTTPException(status_code=500, detail="系統視覺大腦配置錯誤")
                
            elif error_level == "CONTENT_REJECT":
                logger.warning(f"[{job_id}] ❌ 內容評估 [失敗]: {eval_result.get('reason', '無提供原因')}")
                
                if pass_num == MAX_CORRECTION_PASSES - 1:
                    logger.info(f"[{job_id}] ⚠️ 達到最大修正次數，強制回傳最終相片。")
                    break
                    
                logger.info(f"[{job_id}] 準備執行修正 Pass {pass_num+1}...")
                
                raw_data = eval_result.get("raw_semantic_data")
                if raw_data:
                    routing_decision = SemanticRouter.determine_route(raw_data)
                else:
                    routing_decision = {"tier": 1, "target_bboxes": eval_result.get("bboxes", []), "strategy_name": "LEGACY_FALLBACK_LEVEL_1"}
                
                tier = routing_decision["tier"]
                target_bboxes = routing_decision["target_bboxes"]
                
                if tier == 0:
                    logger.info(f"[{job_id}] ✅ VLM 判定全部保留，提早結束修正迴圈。")
                    break

                temp_base_path = self.job_manager.save_image(job_id, pass_num + 1, f"temp_base_pass_{pass_num}.jpg", current_image_bytes)

                if tier == 2:
                    current_denoise = min(0.85 + (pass_num * 0.05), 0.95)
                    logger.info(f"[{job_id}] 🌍 [Tier 2] 啟動全域重構 | Denoise: {current_denoise:.2f}")
                    current_image_bytes = run_comfyui_global_rebuild(year, location, temp_base_path, current_denoise)
                else:
                    if not target_bboxes:
                        target_bboxes = [[400, 400, 600, 600]]
                        
                    with Image.open(io.BytesIO(current_image_bytes)) as img:
                        img_width, img_height = img.size
                        
                    mask_bytes = create_mask_from_bboxes(img_width, img_height, target_bboxes)
                    mask_path = self.job_manager.save_image(job_id, pass_num + 1, "error_mask.png", mask_bytes)
                    
                    current_denoise = min(0.75 + (pass_num * 0.05), 0.85)
                    logger.info(f"[{job_id}] 🖌️ [Tier 1] 啟動局部修補 | Denoise: {current_denoise:.2f}")
                    current_image_bytes = run_comfyui_inpaint(year, location, temp_base_path, mask_path, current_denoise)
        
        logger.info(f"[{job_id}] 成功裁除黑邊並還原圖片尺寸: {metadata['original_size']}")
        final_bytes = crop_to_original(current_image_bytes, metadata)
        
        logger.info(f"[{job_id}] 📤 任務完成，回傳最終圖片")
        return final_bytes