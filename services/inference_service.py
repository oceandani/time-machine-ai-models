import json
import os
import random
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

import requests
from dotenv import load_dotenv
from fastapi import HTTPException
from websocket import WebSocketTimeoutException, create_connection

from core.logging import logger

load_dotenv()

# ==============================================================================
# 1. 頂層連線、兩階段隔離控制與凍結參數常數 (與 global_rebuild_workflow_api.json 100% 同步)
# ==============================================================================
COMFYUI_HTTP_URL = os.getenv("COMFYUI_HTTP_URL", "http://127.0.0.1:8188")
COMFYUI_WS_URL = os.getenv("COMFYUI_WS_URL", "ws://127.0.0.1:8188/ws")

# 兩階段實驗控制開關 (可由 .env 覆寫，階段 A 預設停用 IPAdapter 並固定 Seed=42)
ENABLE_IPADAPTER = os.getenv("ENABLE_IPADAPTER", "false").lower() == "true"
STRICT_VISUAL_REFERENCE_REQUIRED = (
    os.getenv("STRICT_VISUAL_REFERENCE_REQUIRED", "true").lower() == "true"
)
FIXED_DEBUG_SEED = int(os.getenv("FIXED_DEBUG_SEED", "42"))

WS_CONNECT_TIMEOUT_SEC = 0.5
WS_DRAIN_TIMEOUT_SEC = 0.05
HTTP_POLL_TIMEOUT_SEC = 1.0
OBJECT_INFO_TIMEOUT_SEC = 2.0
WORKFLOW_EXECUTION_TIMEOUT_SEC = 600
MAX_RANDOM_SEED = 1000000000

HTTP_STATUS_OK = 200
HTTP_STATUS_INTERNAL_ERROR = 500
HTTP_STATUS_SERVICE_UNAVAILABLE = 503

# 階段 A 刻意凍結之 ControlNet / M-LSD 基準常數 (與 v10.4 Pass 1 及 JSON 100% 一致)
STANDARD_CONTROLNET_RESOLUTION = 768
DEFAULT_CONTROLNET_STRENGTH = 0.46
DEFAULT_CONTROLNET_END_PERCENT = 0.54
DEFAULT_DENOISE = 0.94

DEFAULT_IPADAPTER_WEIGHT = 0.58
DEFAULT_IPADAPTER_END_AT = 0.82
PREFERRED_IPADAPTER_PRESETS = (
    "PLUS (high strength)",
    "STANDARD (medium strength)",
    "VIT-G (medium strength)",
)
# 首選 "style transfer"，防止歷史參考圖不同拍攝視角扭曲用戶原圖 3D 街道透視
PREFERRED_IPADAPTER_WEIGHT_TYPES = (
    "style transfer",
    "linear",
    "style and composition",
    "standard",
)

# IPAdapter Plus 官方 SD1.5 × CLIP Vision 嚴格配對規則 (絕不隨意混用不相容 CLIP Vision)
CLIP_VISION_INSTALL_DIR = "ComfyUI/models/clip_vision/"
IPADAPTER_PRESET_CLIPVISION_RULES: Dict[str, Dict[str, Any]] = {
    "PLUS (high strength)": {
        "official_filename": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        "required_arch_tokens": ("vit-h", "vit_h"),
        "forbidden_arch_tokens": ("bigg", "big-g", "vit-g"),
    },
    "STANDARD (medium strength)": {
        "official_filename": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        "required_arch_tokens": ("vit-h", "vit_h"),
        "forbidden_arch_tokens": ("bigg", "big-g", "vit-g"),
    },
    "VIT-G (medium strength)": {
        "official_filename": "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors",
        "required_arch_tokens": ("bigg", "big-g", "vit-g"),
        "forbidden_arch_tokens": ("vit-h", "vit_h"),
    },
}

CHECKPOINT_NODE_ID = "4"
KSAMPLER_NODE_ID = "12"
IPADAPTER_REF_IMAGE_NODE_ID = "16"
IPADAPTER_LOADER_NODE_ID = "17"
IPADAPTER_APPLY_NODE_ID = "18"
REQUIRED_IPADAPTER_EXEC_NODE_IDS = (
    IPADAPTER_REF_IMAGE_NODE_ID,
    IPADAPTER_LOADER_NODE_ID,
    IPADAPTER_APPLY_NODE_ID,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "configs")

_CACHED_COMFYUI_OBJECT_INFO: Optional[Dict[str, Any]] = None


class ComfyUIExecutionResult(TypedDict):
    image_bytes: bytes
    seed_used: int
    prompt_id: str
    ipadapter_requested: bool
    historical_reference_applied: bool
    ipadapter_status: str
    executed_nodes: List[str]
    missing_models: List[str]
    reference_image_uploaded: Optional[str]


def _resolve_generation_seed(seed_override: Optional[int] = None) -> int:
    """若指定 seed_override 或 FIXED_DEBUG_SEED > 0，則使用固定 Seed 供 A/B 控制變因對照。"""
    if seed_override is not None and seed_override > 0:
        return int(seed_override)
    if FIXED_DEBUG_SEED > 0:
        return FIXED_DEBUG_SEED
    return random.randint(1, MAX_RANDOM_SEED)


def _get_comfyui_object_info(force_refresh: bool = False) -> Dict[str, Any]:
    """查詢本地 ComfyUI 節點與已安裝模型清單，用於 IPAdapter × CLIP Vision 嚴格配對預檢。"""
    global _CACHED_COMFYUI_OBJECT_INFO
    if _CACHED_COMFYUI_OBJECT_INFO is not None and not force_refresh:
        return _CACHED_COMFYUI_OBJECT_INFO
    try:
        resp = requests.get(
            f"{COMFYUI_HTTP_URL}/object_info", timeout=OBJECT_INFO_TIMEOUT_SEC
        )
        if resp.status_code == HTTP_STATUS_OK:
            data = resp.json()
            if isinstance(data, dict):
                _CACHED_COMFYUI_OBJECT_INFO = data
                return data
    except requests.exceptions.RequestException as e:
        logger.warning(f"⚠️ 無法查詢 ComfyUI /object_info: {e}")
    return {}


def _extract_combo_options(
    obj_info: Dict[str, Any], node_name: str, input_key: str
) -> List[str]:
    node_meta = obj_info.get(node_name, {})
    required_inputs = node_meta.get("input", {}).get("required", {})
    raw_spec = required_inputs.get(input_key)
    if isinstance(raw_spec, list) and len(raw_spec) > 0 and isinstance(raw_spec[0], list):
        return [str(item) for item in raw_spec[0]]
    return []


def _validate_clip_vision_for_preset(
    obj_info: Dict[str, Any], preset_name: str
) -> Tuple[bool, List[str], List[str]]:
    """
    嚴格核對本地 ComfyUI/models/clip_vision/ 是否安裝與目標 IPAdapter preset 相容之官方 CLIP Vision 模型。
    回傳: (是否相容就緒, 缺少之官方模型清單, 本地已檢測到之 CLIP Vision 檔案清單)
    """
    installed_clips = _extract_combo_options(
        obj_info, "CLIPVisionLoader", "clip_name"
    )
    rule = IPADAPTER_PRESET_CLIPVISION_RULES.get(preset_name)
    if not rule:
        if installed_clips:
            return True, [], installed_clips
        return (
            False,
            [
                f"任何相容於 preset '{preset_name}' 之 CLIP Vision 模型 (請放置於 {CLIP_VISION_INSTALL_DIR})"
            ],
            installed_clips,
        )

    official_name = str(rule["official_filename"])
    req_tokens = tuple(rule["required_arch_tokens"])
    forbid_tokens = tuple(rule["forbidden_arch_tokens"])

    for clip_file in installed_clips:
        clip_lower = os.path.basename(clip_file).lower()
        if clip_lower == official_name.lower():
            return True, [], installed_clips
        has_req = any(tok in clip_lower for tok in req_tokens)
        has_forbid = any(tok in clip_lower for tok in forbid_tokens)
        if has_req and not has_forbid:
            return True, [], installed_clips

    return (
        False,
        [f"{CLIP_VISION_INSTALL_DIR}{official_name} (配對 preset: {preset_name})"],
        installed_clips,
    )


def upload_image_to_comfyui(image_path: str) -> str:
    upload_url = f"{COMFYUI_HTTP_URL}/upload/image"
    try:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f)}
            data = {"overwrite": "true"}
            response = requests.post(upload_url, files=files, data=data)
            response.raise_for_status()
            return response.json()["name"]
    except Exception as e:
        logger.error(f"上傳圖片至 ComfyUI 失敗: {e}")
        raise HTTPException(
            status_code=HTTP_STATUS_INTERNAL_ERROR,
            detail=f"上傳圖片失敗: {e}",
        )


def download_image(filename: str, subfolder: str, image_type: str) -> bytes:
    view_url = (
        f"{COMFYUI_HTTP_URL}/view?filename={filename}"
        f"&subfolder={subfolder}&type={image_type}"
    )
    img_response = requests.get(view_url)
    img_response.raise_for_status()
    return img_response.content


def inject_prompts(prompt_data: dict, year: str, location: str) -> None:
    for node_id, node_info in prompt_data.items():
        if node_info.get("class_type") == "CLIPTextEncode":
            original_text = node_info["inputs"].get("text", "")
            if "[YEAR]" in original_text or "[LOCATION]" in original_text:
                new_text = original_text.replace("[YEAR]", year).replace(
                    "[LOCATION]", location
                )
                node_info["inputs"]["text"] = new_text
                logger.info(
                    f"📝 成功將 Node {node_id} 的 Prompt 佔位符替換為: {year}, {location}"
                )


def inject_dynamic_prompts(
    prompt_data: dict, positive_prompt: str, negative_prompt: str
) -> None:
    for node_id, node_info in prompt_data.items():
        if node_info.get("class_type") == "CLIPTextEncode":
            title = node_info.get("_meta", {}).get("title", "")
            if "Positive" in title:
                node_info["inputs"]["text"] = positive_prompt
                logger.info(f"📝 [Node {node_id}] 注入動態正向 Prompt")
            elif "Negative" in title:
                node_info["inputs"]["text"] = negative_prompt
                logger.info(f"📝 [Node {node_id}] 注入動態負向 Prompt")


def get_save_image_node_id(prompt_data: dict) -> str:
    for node_id, node_info in prompt_data.items():
        if node_info.get("class_type") == "SaveImage":
            return node_id
    raise ValueError("❌ 找不到 SaveImage 節點，請檢查 JSON 檔案！")


def _bypass_ipadapter_in_payload(prompt_data: dict) -> None:
    """移除 Node 16/17/18，並將 KSampler (Node 12) 的 model 輸入直接連回 Node 4 Checkpoint。"""
    for _, node_info in prompt_data.items():
        if (
            isinstance(node_info, dict)
            and node_info.get("class_type") == "KSampler"
        ):
            inputs = node_info.get("inputs")
            if isinstance(inputs, dict):
                inputs["model"] = [CHECKPOINT_NODE_ID, 0]

    for rm_id in REQUIRED_IPADAPTER_EXEC_NODE_IDS:
        prompt_data.pop(rm_id, None)


def _configure_ipadapter_conditioning(
    prompt_data: dict,
    enable_ipadapter: bool,
    strict_required: bool,
    ref_img_comfy_name: Optional[str],
    ipadapter_weight: float,
    ipadapter_end_at: float,
) -> Tuple[bool, str, List[str]]:
    """
    動態配置並預檢 ComfyUI IPAdapter 視覺條件鏈 (Nodes 16, 17, 18)。
    回傳: (payload是否已掛載IPAdapter, 狀態碼字串, 缺少模型清單)
    """
    if not enable_ipadapter:
        _bypass_ipadapter_in_payload(prompt_data)
        return False, "DISABLED_BY_ISOLATION_MODE", []

    if not ref_img_comfy_name:
        _bypass_ipadapter_in_payload(prompt_data)
        return False, "NO_HISTORICAL_REFERENCE_IMAGE", []

    obj_info = _get_comfyui_object_info(force_refresh=True)
    missing_models: List[str] = []

    if not obj_info or (
        "IPAdapterUnifiedLoader" not in obj_info
        or "IPAdapterAdvanced" not in obj_info
    ):
        missing_models.append(
            "ComfyUI_IPAdapter_plus 自訂節點 (IPAdapterUnifiedLoader / IPAdapterAdvanced)"
        )
        if strict_required:
            raise HTTPException(
                status_code=HTTP_STATUS_SERVICE_UNAVAILABLE,
                detail={
                    "error": "IPADAPTER_NODE_MISSING",
                    "historical_reference_applied": False,
                    "missing_models": missing_models,
                    "message": "嚴格模式已攔截：本地 ComfyUI 缺少 IPAdapter 節點，拒絕靜默降級。",
                },
            )
        _bypass_ipadapter_in_payload(prompt_data)
        return False, "BYPASSED_MISSING_IPADAPTER_NODE", missing_models

    # 1. 對齊 IPAdapterUnifiedLoader preset 枚舉
    valid_presets = _extract_combo_options(
        obj_info, "IPAdapterUnifiedLoader", "preset"
    )
    chosen_preset = PREFERRED_IPADAPTER_PRESETS[0]
    if (
        IPADAPTER_LOADER_NODE_ID in prompt_data
        and isinstance(prompt_data[IPADAPTER_LOADER_NODE_ID].get("inputs"), dict)
    ):
        cur_preset = prompt_data[IPADAPTER_LOADER_NODE_ID]["inputs"].get(
            "preset", chosen_preset
        )
        if valid_presets and cur_preset not in valid_presets:
            chosen_preset = next(
                (p for p in PREFERRED_IPADAPTER_PRESETS if p in valid_presets),
                valid_presets[0],
            )
        else:
            chosen_preset = cur_preset
        prompt_data[IPADAPTER_LOADER_NODE_ID]["inputs"]["preset"] = chosen_preset

    # 2. 嚴格預檢 CLIP Vision 模型與 preset 的官方型號配對 (防輪迴黑名單 #6)
    clip_ok, clip_missing, installed_clips = _validate_clip_vision_for_preset(
        obj_info, chosen_preset
    )
    if not clip_ok:
        missing_models.extend(clip_missing)
        logger.error(
            f"🚨 [IPAdapter × CLIP Vision 預檢失敗] preset='{chosen_preset}' 缺少配對模型: {clip_missing} | "
            f"本地已安裝 CLIP Vision: {installed_clips or '空目錄'}"
        )
        if strict_required:
            raise HTTPException(
                status_code=HTTP_STATUS_SERVICE_UNAVAILABLE,
                detail={
                    "error": "CLIP_VISION_MODEL_NOT_FOUND",
                    "historical_reference_applied": False,
                    "required_preset": chosen_preset,
                    "missing_models": missing_models,
                    "installed_clip_vision_models": installed_clips,
                    "message": (
                        f"嚴格模式已中止任務 (STRICT_VISUAL_REFERENCE_REQUIRED=True)："
                        f"IPAdapter preset '{chosen_preset}' 找不到配對的官方 CLIP Vision 模型，"
                        f"請下載並放入 {CLIP_VISION_INSTALL_DIR} 後再啟用 IPAdapter。"
                    ),
                },
            )
        _bypass_ipadapter_in_payload(prompt_data)
        return False, "BYPASSED_MISSING_CLIPVISION_MODEL", missing_models

    # 3. 對齊 IPAdapterAdvanced weight_type ("style transfer" 優先)
    valid_w_types = _extract_combo_options(
        obj_info, "IPAdapterAdvanced", "weight_type"
    )
    if (
        valid_w_types
        and IPADAPTER_APPLY_NODE_ID in prompt_data
        and isinstance(prompt_data[IPADAPTER_APPLY_NODE_ID].get("inputs"), dict)
    ):
        chosen_wt = next(
            (
                wt
                for wt in PREFERRED_IPADAPTER_WEIGHT_TYPES
                if wt in valid_w_types
            ),
            valid_w_types[0],
        )
        prompt_data[IPADAPTER_APPLY_NODE_ID]["inputs"]["weight_type"] = chosen_wt

    # 4. 注入 Node 16 參考圖與 Node 18 權重
    ref_injected = False
    ipa_injected = False
    for node_id, node_info in prompt_data.items():
        if not isinstance(node_info, dict):
            continue
        c_type = node_info.get("class_type")
        title = node_info.get("_meta", {}).get("title", "")
        inputs = node_info.get("inputs")
        if not isinstance(inputs, dict):
            continue

        if c_type == "LoadImage" and "Historical Reference Image" in title:
            inputs["image"] = ref_img_comfy_name
            ref_injected = True
        elif c_type in ("IPAdapterAdvanced", "IPAdapter"):
            inputs["weight"] = float(ipadapter_weight)
            if "end_at" in inputs:
                inputs["end_at"] = float(ipadapter_end_at)
            ipa_injected = True
            logger.info(
                f"🔒 [覆寫合約] 動態定位 IPAdapter (Node {node_id}) -> "
                f"Preset={chosen_preset}, HistoricalRef={ref_img_comfy_name}, "
                f"Weight={ipadapter_weight:.2f}, WeightType={inputs.get('weight_type')}, EndAt={ipadapter_end_at:.2f}"
            )

    if not (ref_injected and ipa_injected):
        missing_models.append("JSON 缺少 Node 16 (Historical Reference Image) 或 Node 18 (IPAdapterAdvanced)")
        if strict_required:
            raise HTTPException(
                status_code=HTTP_STATUS_SERVICE_UNAVAILABLE,
                detail={
                    "error": "IPADAPTER_WORKFLOW_NODES_INVALID",
                    "historical_reference_applied": False,
                    "missing_models": missing_models,
                },
            )
        _bypass_ipadapter_in_payload(prompt_data)
        return False, "BYPASSED_INVALID_WORKFLOW_NODES", missing_models

    return True, "CONFIGURED_PENDING_EXECUTION_PROOF", []


def run_comfyui_inference(
    year: str, location: str, input_image_path: str
) -> bytes:
    client_id = str(uuid.uuid4())
    logger.info(f"[{client_id}] 1. 正在上傳圖片至 ComfyUI...")
    comfy_image_name = upload_image_to_comfyui(input_image_path)

    json_path = os.path.join(CONFIG_DIR, "pass0_workflow_api.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ 找不到工作流檔案: {json_path}")
        raise HTTPException(
            status_code=HTTP_STATUS_INTERNAL_ERROR,
            detail="系統缺少 pass0_workflow_api.json",
        )

    target_node_id = get_save_image_node_id(prompt_data)
    inject_prompts(prompt_data, year, location)

    image_injected = False
    for _, node_info in prompt_data.items():
        if node_info.get("class_type") == "LoadImage":
            node_info["inputs"]["image"] = comfy_image_name
            image_injected = True
        elif node_info.get("class_type") == "KSampler":
            node_info["inputs"]["seed"] = _resolve_generation_seed()

    if not image_injected:
        logger.warning("⚠️ 找不到 LoadImage 節點，無法注入原圖！")

    img_bytes, _, _, _ = _execute_comfyui_workflow(
        client_id, prompt_data, target_node_id=target_node_id
    )
    return img_bytes


def run_comfyui_inpaint(
    year: str,
    location: str,
    base_image_path: str,
    mask_image_path: str,
    denoise: float,
    pos_prompt: str,
    neg_prompt: str,
) -> bytes:
    client_id = str(uuid.uuid4())
    logger.info(f"[{client_id}] 🎨 開始執行 Tier 1 局部 Inpaint 修正...")

    base_img_name = upload_image_to_comfyui(base_image_path)
    mask_img_name = upload_image_to_comfyui(mask_image_path)
    json_path = os.path.join(CONFIG_DIR, "inpaint_workflow_api.json")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ 找不到工作流檔案: {json_path}")
        raise HTTPException(
            status_code=HTTP_STATUS_INTERNAL_ERROR,
            detail="系統缺少 inpaint_workflow_api.json",
        )

    target_node_id = get_save_image_node_id(prompt_data)
    inject_dynamic_prompts(prompt_data, pos_prompt, neg_prompt)

    base_injected, mask_injected = False, False
    for _, node_info in prompt_data.items():
        if node_info.get("class_type") == "LoadImage":
            title = node_info.get("_meta", {}).get("title", "")
            if "Base" in title:
                node_info["inputs"]["image"] = base_img_name
                base_injected = True
            elif "Mask" in title:
                node_info["inputs"]["image"] = mask_img_name
                mask_injected = True
        elif node_info.get("class_type") == "KSampler":
            node_info["inputs"]["seed"] = _resolve_generation_seed()
            node_info["inputs"]["denoise"] = float(denoise)

    if not base_injected or not mask_injected:
        raise RuntimeError(
            "Tier 1 JSON 缺少標題為 'Base Image' 或 'Mask Image' 的節點！"
        )

    img_bytes, _, _, _ = _execute_comfyui_workflow(
        client_id, prompt_data, target_node_id=target_node_id
    )
    return img_bytes


def run_comfyui_global_rebuild(
    year: str,
    location: str,
    base_image_path: str,
    mask_image_path: str,
    denoise: float = DEFAULT_DENOISE,
    controlnet_strength: float = DEFAULT_CONTROLNET_STRENGTH,
    pos_prompt: str = "",
    neg_prompt: str = "",
    controlnet_image_path: Optional[str] = None,
    controlnet_end_percent: float = DEFAULT_CONTROLNET_END_PERCENT,
    reference_image_path: Optional[str] = None,
    ipadapter_weight: float = DEFAULT_IPADAPTER_WEIGHT,
    ipadapter_end_at: float = DEFAULT_IPADAPTER_END_AT,
    enable_ipadapter_override: Optional[bool] = None,
    strict_visual_ref_override: Optional[bool] = None,
    seed_override: Optional[int] = None,
) -> ComfyUIExecutionResult:
    """
    v10.5 兩階段控制變因 ComfyUI 執行器：
    - 強制核驗 Node 10 為 VAEEncodeForInpaint (防輪迴黑名單 #1：徹底隔離 9 通道 c_concat 背景原圖)。
    - 階段 A (ENABLE_IPADAPTER=False)：顯式停用 IPAdapter 並使用固定 Seed 驗證純遮罩重繪管線。
    - 階段 B (ENABLE_IPADAPTER=True + STRICT_VISUAL_REFERENCE_REQUIRED=True)：
      預檢 CLIP Vision 配對；載入或執行失敗立即中止報錯 (HTTP 503)，絕不靜默降級。
    - historical_reference_applied 僅在 ComfyUI 確認 Node 16/17/18 全數成功執行後才設為 True。
    """
    client_id = str(uuid.uuid4())
    effective_enable_ipa = (
        ENABLE_IPADAPTER
        if enable_ipadapter_override is None
        else bool(enable_ipadapter_override)
    )
    effective_strict_ref = (
        STRICT_VISUAL_REFERENCE_REQUIRED
        if strict_visual_ref_override is None
        else bool(strict_visual_ref_override)
    )
    seed_to_use = _resolve_generation_seed(seed_override)

    mode_desc = (
        "階段 B (IPAdapter 歷史視覺注入 - 嚴格核驗)"
        if effective_enable_ipa
        else "階段 A (隔離測試 - 顯式停用 IPAdapter)"
    )
    logger.info(
        f"[{client_id}] 🌍 開始執行 v10.5 {mode_desc} | "
        f"VAEEncodeForInpaint + ControlNet + 純二值遮罩 | Seed={seed_to_use}"
    )

    base_img_name = upload_image_to_comfyui(base_image_path)
    mask_img_name = upload_image_to_comfyui(mask_image_path)
    effective_ctrl_path = (
        controlnet_image_path if controlnet_image_path else base_image_path
    )
    ctrl_img_name = upload_image_to_comfyui(effective_ctrl_path)

    ref_img_name: Optional[str] = None
    if (
        effective_enable_ipa
        and reference_image_path
        and os.path.isfile(reference_image_path)
    ):
        ref_img_name = upload_image_to_comfyui(reference_image_path)

    json_path = os.path.join(CONFIG_DIR, "global_rebuild_workflow_api.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=HTTP_STATUS_INTERNAL_ERROR,
            detail="系統缺少 global_rebuild_workflow_api.json",
        )

    if not isinstance(prompt_data, dict):
        raise ValueError("ComfyUI Payload 缺乏有效的字典結構")

    target_node_id = get_save_image_node_id(prompt_data)
    inject_dynamic_prompts(prompt_data, pos_prompt, neg_prompt)

    base_injected, mask_injected, ctrl_img_injected = False, False, False
    inpaint_encode_verified = False
    controlnet_injected, ksampler_injected = False, False

    for node_id, node_info in prompt_data.items():
        if not isinstance(node_info, dict):
            continue

        class_type = node_info.get("class_type")
        title = node_info.get("_meta", {}).get("title", "")
        inputs = node_info.get("inputs")

        # 防輪迴黑名單 #1 強制攔截：9 通道 dreamshaper_8Inpainting 嚴禁使用標準 VAEEncode / SetLatentNoiseMask
        if class_type in ("VAEEncode", "SetLatentNoiseMask"):
            raise RuntimeError(
                f"🚨 [防輪迴黑名單 #1 攔截] 偵測到 Node {node_id} ({class_type})！"
                "9 通道 dreamshaper_8Inpainting 模型必須使用 VAEEncodeForInpaint (Node 10) 隔離背景原圖，"
                "否則會觸發原圖 95% 複製 (Identity Copy)！"
            )

        if class_type == "LoadImage" and isinstance(inputs, dict):
            if "Base Image" in title:
                inputs["image"] = base_img_name
                base_injected = True
            elif "Mask Image" in title:
                inputs["image"] = mask_img_name
                mask_injected = True
            elif "ControlNet Image" in title:
                inputs["image"] = ctrl_img_name
                ctrl_img_injected = True

        elif class_type == "VAEEncodeForInpaint" and isinstance(inputs, dict):
            inpaint_encode_verified = True

        elif class_type == "M-LSDPreprocessor" and isinstance(inputs, dict):
            inputs["resolution"] = STANDARD_CONTROLNET_RESOLUTION

        elif class_type == "KSampler" and isinstance(inputs, dict):
            inputs["seed"] = seed_to_use
            inputs["denoise"] = float(denoise)
            ksampler_injected = True
            logger.info(
                f"🔒 [覆寫合約] 動態定位 KSampler (Node {node_id}) -> "
                f"Seed = {seed_to_use}, Denoise = {denoise:.2f}"
            )

        elif class_type in (
            "ControlNetApply",
            "ControlNetApplyAdvanced",
        ) and isinstance(inputs, dict):
            inputs["strength"] = float(controlnet_strength)
            if "end_percent" in inputs:
                inputs["end_percent"] = float(controlnet_end_percent)
            controlnet_injected = True
            logger.info(
                f"🔒 [覆寫合約] 動態定位 ControlNet (Node {node_id}) -> "
                f"Strength = {controlnet_strength:.2f}, EndPct = {controlnet_end_percent:.2f}"
            )

    if not all(
        [
            base_injected,
            mask_injected,
            ctrl_img_injected,
            inpaint_encode_verified,
            controlnet_injected,
            ksampler_injected,
        ]
    ):
        missing = []
        if not base_injected:
            missing.append("Base Image (Node 1)")
        if not mask_injected:
            missing.append("Mask Image (Node 2)")
        if not ctrl_img_injected:
            missing.append("ControlNet Image (Node 15)")
        if not inpaint_encode_verified:
            missing.append("VAEEncodeForInpaint (Node 10)")
        if not controlnet_injected:
            missing.append("ControlNet (Node 9)")
        if not ksampler_injected:
            missing.append("KSampler (Node 12)")
        logger.error(
            f"🚨 [API 注入硬性斷言失敗] 以下節點未成功注入或驗證: {', '.join(missing)}"
        )
        raise RuntimeError(
            f"global_rebuild_workflow_api.json 缺少或無效節點: {', '.join(missing)}"
        )

    ipadapter_configured, ipa_status, missing_models = (
        _configure_ipadapter_conditioning(
            prompt_data=prompt_data,
            enable_ipadapter=effective_enable_ipa,
            strict_required=effective_strict_ref,
            ref_img_comfy_name=ref_img_name,
            ipadapter_weight=ipadapter_weight,
            ipadapter_end_at=ipadapter_end_at,
        )
    )

    ksampler_model_link = (
        prompt_data.get(KSAMPLER_NODE_ID, {}).get("inputs", {}).get("model", [])
    )
    logger.info(
        f"🖼️ [ComfyUI 生成器視覺輸入審計] "
        f"Node 1 (Base)={base_img_name} | "
        f"Node 2 (BinaryMask)={mask_img_name} | "
        f"Node 15 (Control)={ctrl_img_name} | "
        f"Node 16 (IPAdapter HistRef)={ref_img_name if ipadapter_configured else f'未啟用 ({ipa_status})'} | "
        f"KSampler.model={ksampler_model_link}"
    )

    try:
        image_bytes, executed_nodes, exec_status_str, prompt_id = (
            _execute_comfyui_workflow(
                client_id, prompt_data, target_node_id=target_node_id
            )
        )
    except Exception as exec_err:
        if ipadapter_configured and effective_strict_ref:
            logger.error(
                f"🚨 [嚴格模式中止] IPAdapter 執行階段失敗 ({exec_err})，拒絕靜默降級！"
            )
            raise HTTPException(
                status_code=HTTP_STATUS_SERVICE_UNAVAILABLE,
                detail={
                    "error": "IPADAPTER_EXECUTION_FAILED",
                    "historical_reference_applied": False,
                    "message": str(exec_err),
                },
            )
        raise exec_err

    # 四重實證判定 historical_reference_applied (絕不單憑路徑或權重設定就標為 True)
    executed_set = set(executed_nodes)
    all_ipa_nodes_executed = all(
        nid in executed_set for nid in REQUIRED_IPADAPTER_EXEC_NODE_IDS
    )
    ksampler_uses_ipa = (
        isinstance(ksampler_model_link, list)
        and len(ksampler_model_link) >= 1
        and str(ksampler_model_link[0]) == IPADAPTER_APPLY_NODE_ID
    )
    historical_reference_applied = bool(
        effective_enable_ipa
        and ipadapter_configured
        and ref_img_name is not None
        and ksampler_uses_ipa
        and exec_status_str == "success"
        and all_ipa_nodes_executed
    )

    if ipadapter_configured:
        if historical_reference_applied:
            ipa_status = "APPLIED_VERIFIED"
        else:
            ipa_status = "UNVERIFIED_EXECUTION_NODES"
            logger.warning(
                f"⚠️ [IPAdapter 執行實證未通過] executed_nodes={sorted(executed_set)} 缺少必要節點 {REQUIRED_IPADAPTER_EXEC_NODE_IDS}，"
                "強制維持 historical_reference_applied=False！"
            )

    logger.info(
        f"✅ [ComfyUI 執行實證] PromptID={prompt_id} | Seed={seed_to_use} | "
        f"historical_reference_applied={historical_reference_applied} ({ipa_status}) | "
        f"ExecutedNodes={sorted(executed_set)}"
    )

    return {
        "image_bytes": image_bytes,
        "seed_used": seed_to_use,
        "prompt_id": prompt_id,
        "ipadapter_requested": effective_enable_ipa,
        "historical_reference_applied": historical_reference_applied,
        "ipadapter_status": ipa_status,
        "executed_nodes": sorted(executed_set),
        "missing_models": missing_models,
        "reference_image_uploaded": ref_img_name if ipadapter_configured else None,
    }


def _drain_websocket_node_events(
    ws: Any,
    prompt_id: str,
    executed_nodes: Set[str],
    target_node_id: Optional[str],
    drain_timeout: float,
) -> Optional[Dict[str, Any]]:
    """讀取 WebSocket 佇列中所有待處理事件，精準追蹤每個已執行／命中快取之節點 ID。"""
    found_image_info: Optional[Dict[str, Any]] = None
    prev_timeout = ws.gettimeout()
    ws.settimeout(drain_timeout)
    try:
        while True:
            out = ws.recv()
            if not isinstance(out, str):
                continue
            msg = json.loads(out)
            msg_type = msg.get("type")
            data = msg.get("data", {})
            if not isinstance(data, dict):
                continue

            if msg_type == "execution_error" and data.get("prompt_id") == prompt_id:
                raise RuntimeError(
                    f"ComfyUI 執行階段錯誤 (Node {data.get('node_id')} - {data.get('node_type')}): "
                    f"{data.get('exception_message')}"
                )
            if msg_type == "execution_cached" and data.get("prompt_id") == prompt_id:
                for cached_nid in data.get("nodes", []):
                    executed_nodes.add(str(cached_nid))
            elif msg_type == "executing" and data.get("prompt_id") == prompt_id:
                exec_nid = data.get("node")
                if exec_nid is not None:
                    executed_nodes.add(str(exec_nid))
            elif msg_type == "executed" and data.get("prompt_id") == prompt_id:
                exec_nid = data.get("node")
                if exec_nid is not None:
                    executed_nodes.add(str(exec_nid))
                if target_node_id and str(exec_nid) != str(target_node_id):
                    continue
                images = data.get("output", {}).get("images", [])
                if len(images) > 0:
                    found_image_info = images[0]
    except (WebSocketTimeoutException, TimeoutError, OSError):
        pass
    finally:
        ws.settimeout(prev_timeout)
    return found_image_info


def _execute_comfyui_workflow(
    client_id: str, prompt_data: dict, target_node_id: Optional[str] = None
) -> Tuple[bytes, List[str], str, str]:
    ws = None
    try:
        ws = create_connection(f"{COMFYUI_WS_URL}?clientId={client_id}")
        ws.settimeout(WS_CONNECT_TIMEOUT_SEC)
    except Exception as e:
        logger.error(f"無法連接 WebSocket: {e}")
        raise HTTPException(
            status_code=HTTP_STATUS_INTERNAL_ERROR,
            detail=f"無法連接 WebSocket: {e}",
        )

    try:
        payload = {"prompt": prompt_data, "client_id": client_id}
        response = requests.post(f"{COMFYUI_HTTP_URL}/prompt", json=payload)
        if response.status_code != HTTP_STATUS_OK:
            raise RuntimeError(
                f"ComfyUI /prompt 拒絕請求 ({response.status_code}): {response.text}"
            )
        prompt_id = str(response.json().get("prompt_id", ""))
        logger.info(f"[{client_id}] 任務已提交 (ID: {prompt_id})，開始監控...")

        final_image_info: Optional[Dict[str, Any]] = None
        executed_nodes: Set[str] = set()
        exec_status_str = "unknown"
        start_time = time.time()

        while time.time() - start_time < WORKFLOW_EXECUTION_TIMEOUT_SEC:
            ws_img = _drain_websocket_node_events(
                ws=ws,
                prompt_id=prompt_id,
                executed_nodes=executed_nodes,
                target_node_id=target_node_id,
                drain_timeout=WS_CONNECT_TIMEOUT_SEC,
            )
            if ws_img is not None:
                final_image_info = ws_img

            try:
                hist_resp = requests.get(
                    f"{COMFYUI_HTTP_URL}/history/{prompt_id}",
                    timeout=HTTP_POLL_TIMEOUT_SEC,
                )
                if hist_resp.status_code == HTTP_STATUS_OK:
                    hist = hist_resp.json()
                    if prompt_id in hist:
                        status_info = hist[prompt_id].get("status", {})
                        exec_status_str = str(
                            status_info.get("status_str", "success")
                        )
                        for msg_entry in status_info.get("messages", []):
                            if (
                                isinstance(msg_entry, list)
                                and len(msg_entry) >= 2
                                and msg_entry[0] == "execution_cached"
                                and isinstance(msg_entry[1], dict)
                            ):
                                for cnid in msg_entry[1].get("nodes", []):
                                    executed_nodes.add(str(cnid))

                        if exec_status_str == "error":
                            raise RuntimeError(
                                f"ComfyUI 節點執行錯誤: {status_info.get('messages')}"
                            )

                        outputs = hist[prompt_id].get("outputs", {})
                        for out_nid in outputs.keys():
                            executed_nodes.add(str(out_nid))

                        if target_node_id and target_node_id in outputs:
                            images = outputs[target_node_id].get("images", [])
                            if len(images) > 0:
                                final_image_info = images[0]
                        elif not target_node_id:
                            for _, node_output in outputs.items():
                                images = node_output.get("images", [])
                                if len(images) > 0:
                                    final_image_info = images[0]
                                    break

                        if final_image_info is not None:
                            _drain_websocket_node_events(
                                ws=ws,
                                prompt_id=prompt_id,
                                executed_nodes=executed_nodes,
                                target_node_id=target_node_id,
                                drain_timeout=WS_DRAIN_TIMEOUT_SEC,
                            )
                            break
            except requests.exceptions.RequestException as poll_err:
                logger.debug(f"輪詢 ComfyUI history 暫時異常: {poll_err}")

        if not final_image_info:
            raise HTTPException(
                status_code=HTTP_STATUS_INTERNAL_ERROR,
                detail="生圖超時或 API 未回傳有效影像",
            )

        if exec_status_str == "unknown":
            exec_status_str = "success"

        logger.info(f"[{client_id}] 成功獲取圖片: {final_image_info['filename']}")
        img_bytes = download_image(final_image_info["filename"], "", "output")
        return img_bytes, sorted(executed_nodes), exec_status_str, prompt_id

    finally:
        if ws:
            try:
                ws.close()
            except Exception as close_err:
                logger.warning(f"關閉 WebSocket 時發生輕微異常: {close_err}")