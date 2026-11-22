import io
import math
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import cv2
import numpy as np
from PIL import Image, ImageOps

from core.logging import logger
from services.vlm_service import (
    DEFAULT_HISTORICAL_CEILING_RATIO,
    DEFAULT_IPADAPTER_END_AT,
    DEFAULT_IPADAPTER_WEIGHT,
    DEFAULT_VANISHING_POINT_X,
    DEFAULT_VANISHING_POINT_Y,
    PERSPECTIVE_INNER_CORRIDOR_X_LEFT,
    PERSPECTIVE_INNER_CORRIDOR_X_RIGHT,
)

# ==============================================================================
# 1. 頂層物理與幾何常數定義 (零魔法數字散落，與 JSON / inference_service / vlm_service 100% 同步)
# ==============================================================================
STANDARD_MAX_IMAGE_SIZE = 1024
LATENT_GRID_ALIGNMENT = 64
MAX_PIXEL_VAL = 255.0
PERCENTAGE_SCALE = 100.0
BINARY_MASK_FULL_REWRITE = 255
BINARY_MASK_ZERO_LOCK = 0
CORE_MASK_BINARY_THRESHOLD = 128
FEATHER_TAIL_THRESHOLD = 16

NORMALIZED_COORD_SCALE = 1000.0
MIN_POLYGON_VERTICES = 3
MIN_CEILING_MARGIN_PX = 20
TOP_BORDER_LOCK_TOLERANCE_PX = 4
MAX_DYNAMIC_CEILING_Y_RATIO = 0.64

TIER1_DILATE_KERNEL = 5
TIER1_BLUR_KERNEL = 11
TIER2_DILATE_KERNEL = 3
MASK_DILATE_ITERATIONS = 1

TARGETED_REPAIR_DILATE_KERNEL = 7
COMPOSITE_FEATHER_KERNEL = 13

MAX_JOINT_REPAIR_AREA_RATIO = 0.65
BASE_SINGLE_REPAIR_MAX_RATIO = 0.48
REPAIR_ENVELOPE_MARGIN = 0.02
REPAIR_GATE_TOLERANCE = 0.04
MIN_CLAMP_DIM_PX = 10
FALLBACK_CORRIDOR_HALF_W_RATIO = 0.14
FALLBACK_CENTRAL_PAD_X_RATIO = 0.18
FALLBACK_ROAD_TOP_OFFSET_RATIO = 0.08

# 宏觀走廊與微觀實體 (車輛/行人/LED字體) 分離濾波常數 (黑名單 #7：保留 2~4px 電車路軌)
ERASE_VERTICAL_SMOOTH_KERNEL = 99
ERASE_TRANSITION_FEATHER_KERNEL = 35
MESO_VERTICAL_DAMPING_KERNEL_W = 21
MESO_VERTICAL_DAMPING_KERNEL_H = 3
FACADE_DAMPING_BOTTOM_Y_RATIO = 0.72

MICRO_ENTITY_MEDIAN_BLUR_K = 5
CANYON_BILLBOARD_MEDIAN_BLUR_K = 7

DEFAULT_NEUTRAL_PAD_RGB = (210, 215, 220)

# KEEP_EXISTING_SKY (時代廣場 / 曼克頓中央公園) 真實純天空像素分割常數
TRUE_SKY_BRIGHTNESS_MIN = 170
TRUE_SKY_EDGE_DILATE_KERNEL = 9
TRUE_SKY_EDGE_DILATE_ITERATIONS = 2

# Reinhard Kodachrome 懷舊彩色菲林色譜轉移、日間明度守恆與 Auto-Levels 去霧常數
DEFAULT_KODACHROME_LAB_MEAN = (136.0, 131.5, 138.5)
DEFAULT_KODACHROME_LAB_STD = (46.0, 11.5, 16.5)
MIN_ACCEPTABLE_HSV_SATURATION = 22.0
LAB_STD_SAFE_FLOOR = 5.0
LAB_TARGET_STD_SAFE_FLOOR = 3.0
LAB_NORM_DIVISOR_FLOOR = 6.0
COLOR_TRANSFER_STRENGTH_NORMAL = 0.22
COLOR_TRANSFER_STRENGTH_MONOCHROME_RECOVERY = 0.80
DAYLIGHT_LUMA_RETENTION_FLOOR = 0.92
DAYLIGHT_MIN_TARGET_LUMA = 105.0
DAYLIGHT_LUMA_BLEND_WEIGHT = 0.75

DEHAZE_BLACK_PERCENTILE = 2.0
DEHAZE_WHITE_PERCENTILE = 98.5
DEHAZE_ELEVATED_BLACK_THRESHOLD = 18.0
DEHAZE_MIN_DYNAMIC_SPAN = 40.0
DEHAZE_STRENGTH_ALPHA = 0.60

BILATERAL_FILTER_DIAMETER = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# v10.5 階段 A 隔離測試常數：
# 1. VAEEncodeForInpaint 防 0.5 死灰基準 Denoise (>= 0.93，與 JSON 0.94 同步)
# 2. 刻意凍結 ControlNet 強度與區間 (0.46 / 0.54)，確保不與 VAEEncodeForInpaint 混雜多變因
GLOBAL_DENOISE_DEMOLISH = 0.95
GLOBAL_DENOISE_TRUNCATE = 0.94
GLOBAL_DENOISE_CANYON_RESTYLE = 0.94
TARGETED_REPAIR_DENOISE = 0.93

CTRL_STRENGTH_DEMOLISH = 0.40
CTRL_END_PCT_DEMOLISH = 0.48
CTRL_STRENGTH_TRUNCATE = 0.46
CTRL_END_PCT_TRUNCATE = 0.54
CTRL_STRENGTH_CANYON = 0.46
CTRL_END_PCT_CANYON = 0.54
CTRL_STRENGTH_TARGETED_REPAIR = 0.44
CTRL_END_PCT_TARGETED_REPAIR = 0.50

# 跨輪次自適應動態反饋步長與安全上下限
ADAPTIVE_CTRL_STRENGTH_STEP = 0.04
ADAPTIVE_CTRL_END_PCT_STEP = 0.04
ADAPTIVE_CEILING_EXPAND_RATIO = 0.05
ADAPTIVE_DENOISE_BOOST_STEP = 0.01
ADAPTIVE_IPADAPTER_WEIGHT_BOOST = 0.05
MIN_ADAPTIVE_CTRL_STRENGTH = 0.36
MIN_ADAPTIVE_CTRL_END_PCT = 0.42
MAX_ADAPTIVE_DENOISE = 0.96
MAX_ADAPTIVE_IPADAPTER_WEIGHT = 0.72

# 非天空區清晰度評分與 Stage 4 CV 客觀物理量測常數
CLARITY_EVAL_Y_MIN_RATIO = 0.50
CONTRAST_CHECK_Y_MIN_RATIO = 0.50
CLARITY_MAX_HALF_SCORE = 50.0
CLARITY_REF_CONTRAST_STD = 48.0
CLARITY_REF_LAPLACIAN_VAR = 280.0

CANNY_LOW_THRESHOLD = 60
CANNY_HIGH_THRESHOLD = 160
UPPER_SKY_EDGE_DENSITY_MAX = 0.080
UPPER_SKY_CHECK_HEIGHT_FACTOR = 0.65
MIN_ACCEPTABLE_CONTRAST_STD = 24.0

# 防止原圖未重繪 (Anti-Identity-Copy) 物理門檻常數
MIN_EFFECTIVE_REWRITE_MSE = 22.0
MIN_ACTIVE_ZONE_REWRITE_DIFF = 15.0
MIN_CORRIDOR_RETENTION_FLOOR = 0.32
MAX_IDENTITY_COPY_RETENTION_CEILING = 0.86
DEFAULT_FALLBACK_CORRIDOR_RETENTION = 0.60

CORRIDOR_RETENTION_Y_MIN_RATIO = 0.52
CORRIDOR_BLUR_KERNEL = 15
SOBEL_KERNEL_SIZE = 3
CORRIDOR_LUMA_WEIGHT = 0.50
CORRIDOR_GRADIENT_WEIGHT = 0.50
CORR_MATRIX_DIM = 2

MSE_PIXEL_TOLERANCE = 3
MSE_LEAK_RATIO_TOLERANCE = 0.001

# 階段 A 遮罩可視化診斷圖疊加常數 (紅=255重繪區, 綠=0鎖定保護區, 黃=分區多邊形框)
OVERLAY_BASE_ALPHA = 0.68
OVERLAY_MASK_ALPHA = 0.32
OVERLAY_REWRITE_BGR = (40, 40, 235)
OVERLAY_LOCK_BGR = (40, 215, 60)
OVERLAY_POLYGON_BGR = (0, 235, 255)
OVERLAY_LABEL_BG_BGR = (20, 20, 20)
OVERLAY_LABEL_TEXT_BGR = (255, 255, 255)
OVERLAY_FONT_SCALE = 0.52
OVERLAY_LINE_THICKNESS = 2
OVERLAY_TEXT_THICKNESS = 1
OVERLAY_LABEL_PAD_PX = 4
OVERLAY_TEXT_MARGIN_PX = 16

DEFECT_ZONE_PRIORITY_ORDER = (
    "SKY_CANVAS",
    "CENTRAL_BACKDROP",
    "LEFT_FACADE",
    "RIGHT_FACADE",
    "STREET_CORRIDOR",
    "WATER_AND_DOCKYARD",
)


class ImageMetadata(TypedDict):
    original_size: Tuple[int, int]
    scale_ratio: float
    pre_pad_size: Tuple[int, int]
    target_size: Tuple[int, int]
    padding: Dict[str, int]


class DecoupledLayerTensors(TypedDict):
    prepared_base_canvas_bytes: bytes
    inpaint_noise_mask_bytes: bytes
    structural_control_map_bytes: bytes
    composite_protection_mask_bytes: bytes
    recommended_denoise: float
    recommended_controlnet_strength: float
    recommended_controlnet_end_percent: float
    recommended_ipadapter_weight: float
    recommended_ipadapter_end_at: float
    rewrite_coverage_ratio: float
    protected_coverage_ratio: float
    allowed_max_repair_ratio: float
    active_region_ids: List[str]
    repaired_topological_zones: List[str]


# ==============================================================================
# 2. 無黑邊影像標準化 (BORDER_REPLICATE) 與統一座標轉換工具
# ==============================================================================
def process_and_standardize_image(
    image_bytes: bytes, max_size: int = STANDARD_MAX_IMAGE_SIZE
) -> Tuple[bytes, Dict]:
    """
    Stage 0 無黑邊標準化 (防輪迴黑名單 #4)：
    使用 cv2.BORDER_REPLICATE 複製邊緣像素對齊 64 倍數網格，
    徹底消除純黑邊框在畫面頂部引發的 M-LSD 粗黑橫線偽影。
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            background = Image.new("RGB", image.size, DEFAULT_NEUTRAL_PAD_RGB)
            background.paste(
                image, mask=image.split()[3] if image.mode == "RGBA" else None
            )
            image = background
        else:
            image = image.convert("RGB")

        original_width, original_height = image.size
        max_dim = max(original_width, original_height)
        scale_ratio = max_size / max_dim if max_dim > max_size else 1.0

        if scale_ratio < 1.0:
            new_w = int(original_width * scale_ratio)
            new_h = int(original_height * scale_ratio)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        current_w, current_h = image.size
        target_w = (
            math.ceil(current_w / LATENT_GRID_ALIGNMENT) * LATENT_GRID_ALIGNMENT
        )
        target_h = (
            math.ceil(current_h / LATENT_GRID_ALIGNMENT) * LATENT_GRID_ALIGNMENT
        )

        pad_w, pad_h = target_w - current_w, target_h - current_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top

        rgb_arr = np.array(image, dtype=np.uint8)
        if pad_w > 0 or pad_h > 0:
            rgb_arr = cv2.copyMakeBorder(
                rgb_arr,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REPLICATE,
            )

        padded_image = Image.fromarray(rgb_arr)
        output = io.BytesIO()
        padded_image.save(output, format="PNG")
        metadata: ImageMetadata = {
            "original_size": (original_width, original_height),
            "scale_ratio": scale_ratio,
            "pre_pad_size": (current_w, current_h),
            "target_size": (target_w, target_h),
            "padding": {
                "left": pad_left,
                "top": pad_top,
                "right": pad_right,
                "bottom": pad_bottom,
            },
        }
        return output.getvalue(), metadata
    except Exception as e:
        logger.error(f"圖片預處理失敗: {str(e)}")
        raise RuntimeError(f"圖片預處理失敗: {str(e)}")


def _ensure_odd_kernel(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def _binarize_mask_strict(mask: np.ndarray) -> np.ndarray:
    """強制輸出嚴格純二值遮罩 (僅含 0 與 255，杜絕 1~254 灰階洩漏，防輪迴黑名單 #1)。"""
    return np.where(
        mask >= CORE_MASK_BINARY_THRESHOLD,
        BINARY_MASK_FULL_REWRITE,
        BINARY_MASK_ZERO_LOCK,
    ).astype(np.uint8)


def _unnormalize_bbox(
    bbox: List[int], width: int, height: int
) -> Tuple[int, int, int, int]:
    x1 = int(np.clip(bbox[0] * width / NORMALIZED_COORD_SCALE, 0, width - 1))
    y1 = int(np.clip(bbox[1] * height / NORMALIZED_COORD_SCALE, 0, height - 1))
    x2 = int(np.clip(bbox[2] * width / NORMALIZED_COORD_SCALE, 0, width))
    y2 = int(np.clip(bbox[3] * height / NORMALIZED_COORD_SCALE, 0, height))
    return y1, x1, max(y1 + 1, y2), max(x1 + 1, x2)


def _contract_to_pixel_polygon(
    contract: Dict[str, Any], width: int, height: int
) -> np.ndarray:
    pts: List[List[int]] = []
    xy_ratios = contract.get("polygon_xy_ratios", [])
    if isinstance(xy_ratios, list) and len(xy_ratios) >= MIN_POLYGON_VERTICES:
        for pt in xy_ratios:
            if isinstance(pt, list) and len(pt) == 2:
                px = int(np.clip(float(pt[0]) * width, 0, width - 1))
                py = int(np.clip(float(pt[1]) * height, 0, height - 1))
                pts.append([px, py])

    if len(pts) < MIN_POLYGON_VERTICES:
        x1 = int(
            np.clip(float(contract.get("x_min_ratio", 0.0)) * width, 0, width - 1)
        )
        y1 = int(
            np.clip(
                float(contract.get("y_min_ratio", 0.0)) * height, 0, height - 1
            )
        )
        x2 = int(
            np.clip(
                float(contract.get("x_max_ratio", 1.0)) * width, x1 + 1, width
            )
        )
        y2 = int(
            np.clip(
                float(contract.get("y_max_ratio", 1.0)) * height, y1 + 1, height
            )
        )
        pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))


def _clamp_zone_mask_preserving_canvas_borders(
    mask: np.ndarray,
    zone_name: str,
    max_area_ratio: float,
) -> np.ndarray:
    height, width = mask.shape[:2]
    total_pixels = float(height * width)
    current_ratio = float(np.count_nonzero(mask)) / total_pixels
    if current_ratio <= max_area_ratio or current_ratio <= 0.0:
        return mask

    scale_factor = max_area_ratio / current_ratio
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return mask

    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    x_min, x_max = int(np.min(xs)), int(np.max(xs))

    clamped = np.zeros_like(mask)
    if zone_name == "SKY_CANVAS":
        new_y_max = max(1, int(y_min + (y_max - y_min) * scale_factor))
        clamped[0:new_y_max, :] = mask[0:new_y_max, :]
    elif zone_name == "STREET_CORRIDOR":
        new_y_min = min(height - 1, int(y_max - (y_max - y_min) * scale_factor))
        clamped[new_y_min:height, :] = mask[new_y_min:height, :]
    elif zone_name == "LEFT_FACADE":
        new_x_max = max(1, int(x_min + (x_max - x_min) * scale_factor))
        clamped[:, 0:new_x_max] = mask[:, 0:new_x_max]
    elif zone_name == "RIGHT_FACADE":
        new_x_min = min(width - 1, int(x_max - (x_max - x_min) * scale_factor))
        clamped[:, new_x_min:width] = mask[:, new_x_min:width]
    else:
        cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
        half_h = max(
            MIN_CLAMP_DIM_PX,
            int(((y_max - y_min) * math.sqrt(scale_factor)) / 2),
        )
        half_w = max(
            MIN_CLAMP_DIM_PX,
            int(((x_max - x_min) * math.sqrt(scale_factor)) / 2),
        )
        ny1, ny2 = max(0, cy - half_h), min(height, cy + half_h)
        nx1, nx2 = max(0, cx - half_w), min(width, cx + half_w)
        clamped[ny1:ny2, nx1:nx2] = mask[ny1:ny2, nx1:nx2]

    return clamped


# ==============================================================================
# 3. 懷舊彩色菲林色譜轉移、日間明度守恆、自動去霧算子與 IPAdapter 參考圖預處理
# ==============================================================================
def _restore_crisp_daylight_contrast(bgr_img: np.ndarray) -> np.ndarray:
    """確定性自動色階去霧算子 (防輪迴黑名單 #1 配套)：消除 VAEEncodeForInpaint 可能殘留之輕微灰霧感。"""
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]
    p_low = float(np.percentile(l_chan, DEHAZE_BLACK_PERCENTILE))
    p_high = float(np.percentile(l_chan, DEHAZE_WHITE_PERCENTILE))

    if (
        p_low > DEHAZE_ELEVATED_BLACK_THRESHOLD
        and (p_high - p_low) > DEHAZE_MIN_DYNAMIC_SPAN
    ):
        stretched_l = (l_chan - p_low) * (MAX_PIXEL_VAL / (p_high - p_low))
        stretched_l = np.clip(stretched_l, 0.0, MAX_PIXEL_VAL)
        lab[:, :, 0] = (
            DEHAZE_STRENGTH_ALPHA * stretched_l
            + (1.0 - DEHAZE_STRENGTH_ALPHA) * l_chan
        )
        lab = np.clip(lab, 0.0, MAX_PIXEL_VAL).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr_img


def stabilize_archival_color_grading(
    target_image_bytes: bytes,
    color_reference_bytes: Optional[bytes] = None,
    original_image_bytes: Optional[bytes] = None,
    input_is_daylight: bool = True,
) -> bytes:
    """
    結構、光照與色彩三解耦之懷舊菲林色譜錨定與去霧器：
    1. 執行確定性 Auto-Levels 去霧算子。
    2. 提取歷史彩色參考圖之 A/B 色度通道分佈，防止黑白退化。
    3. 強制執行「原圖日間明度守恆」，防止黃昏參考圖將日間原圖拉暗為夜晚。
    """
    target_bgr = cv2.imdecode(
        np.frombuffer(target_image_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    if target_bgr is None:
        return target_image_bytes

    target_bgr = _restore_crisp_daylight_contrast(target_bgr)

    hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))

    ref_mean = np.array(DEFAULT_KODACHROME_LAB_MEAN, dtype=np.float32)
    ref_std = np.array(DEFAULT_KODACHROME_LAB_STD, dtype=np.float32)

    if color_reference_bytes is not None:
        ref_bgr = cv2.imdecode(
            np.frombuffer(color_reference_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        if ref_bgr is not None:
            ref_hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)
            if float(np.mean(ref_hsv[:, :, 1])) >= MIN_ACCEPTABLE_HSV_SATURATION:
                ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(
                    np.float32
                )
                ref_mean = np.mean(ref_lab, axis=(0, 1))
                ref_std = np.maximum(
                    np.std(ref_lab, axis=(0, 1)), LAB_STD_SAFE_FLOOR
                )

    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    t_mean = np.mean(target_lab, axis=(0, 1))
    t_std = np.maximum(
        np.std(target_lab, axis=(0, 1)), LAB_TARGET_STD_SAFE_FLOOR
    )

    is_monochrome_degraded = mean_sat < MIN_ACCEPTABLE_HSV_SATURATION
    blend_alpha = (
        COLOR_TRANSFER_STRENGTH_MONOCHROME_RECOVERY
        if is_monochrome_degraded
        else COLOR_TRANSFER_STRENGTH_NORMAL
    )

    transferred_lab = target_lab.copy()
    for ch in (1, 2):
        norm_ch = (target_lab[:, :, ch] - t_mean[ch]) * (
            ref_std[ch] / max(t_std[ch], LAB_NORM_DIVISOR_FLOOR)
        ) + ref_mean[ch]
        transferred_lab[:, :, ch] = (
            blend_alpha * norm_ch + (1.0 - blend_alpha) * target_lab[:, :, ch]
        )

    if input_is_daylight:
        desired_l_floor = DAYLIGHT_MIN_TARGET_LUMA
        if original_image_bytes is not None:
            orig_bgr = cv2.imdecode(
                np.frombuffer(original_image_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if orig_bgr is not None:
                orig_lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2LAB).astype(
                    np.float32
                )
                orig_l_mean = float(np.mean(orig_lab[:, :, 0]))
                desired_l_floor = max(
                    DAYLIGHT_MIN_TARGET_LUMA,
                    orig_l_mean * DAYLIGHT_LUMA_RETENTION_FLOOR,
                )

        if t_mean[0] < desired_l_floor:
            l_boost = desired_l_floor - t_mean[0]
            boosted_l = target_lab[:, :, 0] + (
                l_boost * DAYLIGHT_LUMA_BLEND_WEIGHT
            )
            transferred_lab[:, :, 0] = np.clip(boosted_l, 0.0, MAX_PIXEL_VAL)
            logger.info(
                f"☀️ [日間明度守恆] 偵測到候選圖明度偏暗 (L={t_mean[0]:.1f} 少於 {desired_l_floor:.1f})，已補償還原日間光照。"
            )

    transferred_lab = np.clip(transferred_lab, 0.0, MAX_PIXEL_VAL).astype(
        np.uint8
    )
    out_bgr = cv2.cvtColor(transferred_lab, cv2.COLOR_LAB2BGR)
    if is_monochrome_degraded:
        logger.info(
            f"🎨 [懷舊彩色菲林色譜修復] 偵測到低飽和度影像 (Sat={mean_sat:.1f})，已透過 LAB 色譜轉移注入懷舊色彩。"
        )
    _, out_buf = cv2.imencode(".png", out_bgr)
    return out_buf.tobytes()


def prepare_ipadapter_reference_image(
    reference_image_bytes: Optional[bytes],
    reference_is_monochrome: bool,
    color_reference_bytes: Optional[bytes],
    input_is_daylight: bool = True,
) -> Optional[bytes]:
    """
    為 ComfyUI IPAdapter (Node 16) 準備全彩視覺參考圖 (防輪迴黑名單 #5)：
    若檢索到的歷史參考照為黑白照片，先注入懷舊彩色色譜並確保日間明度，防止染黑輸出圖。
    """
    if reference_image_bytes is None:
        return None
    if reference_is_monochrome or input_is_daylight:
        return stabilize_archival_color_grading(
            target_image_bytes=reference_image_bytes,
            color_reference_bytes=color_reference_bytes,
            original_image_bytes=None,
            input_is_daylight=input_is_daylight,
        )
    return reference_image_bytes


def _extract_true_sky_lock_mask(
    orig_bgr: np.ndarray,
    candidate_sky_poly_mask: np.ndarray,
) -> np.ndarray:
    """
    專用於 KEEP_EXISTING_SKY (時代廣場 / 曼克頓中央公園)：
    僅鎖定 candidate_sky_poly_mask 內部「高明度且無建築/招牌邊緣」的真實純天空像素。
    """
    gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    dilate_k = np.ones(
        (TRUE_SKY_EDGE_DILATE_KERNEL, TRUE_SKY_EDGE_DILATE_KERNEL), np.uint8
    )
    thick_edges = cv2.dilate(
        edges, dilate_k, iterations=TRUE_SKY_EDGE_DILATE_ITERATIONS
    )

    bright_sky = (gray >= TRUE_SKY_BRIGHTNESS_MIN).astype(np.uint8) * int(
        MAX_PIXEL_VAL
    )
    smooth_sky = cv2.bitwise_and(bright_sky, cv2.bitwise_not(thick_edges))
    return cv2.bitwise_and(smooth_sky, candidate_sky_poly_mask)


# ==============================================================================
# 4. 全地貌「宏觀走廊骨架 vs 微觀車輛/行人輪廓分離」結構控制圖調制器
# ==============================================================================
def _synthesize_frequency_modulated_control_map(
    orig_bgr: np.ndarray,
    plan_data: Dict[str, Any],
    composite_lock_mask: np.ndarray,
    ceiling_expand_px: int = 0,
) -> Tuple[np.ndarray, int, int, int]:
    """
    v10.5 全地貌結構控制圖調制器 (遵守黑名單 #3 & #7)：
    1. DEMOLISH 與 GEOMORPHIC_CHANGE 區域：100% 抹除現代高樓或填海建築結構線，且嚴格鎖死 y=0.0 頂邊不動！
    2. STREET_CORRIDOR 區域：使用輕量 5x5 中值濾波與雙邊濾波，平滑現代車頭燈與行人衣著細碎紋理，
       同時保留地面 2~4px 寬的電車路軌與馬路邊界收斂主線。
    3. LEFT/RIGHT/CENTRAL_FACADE 區域：
       - TRUNCATE_TO_LOW_RISE：抑制現代高樓密集垂直鋁窗與竹棚線，保留水平騎樓層線。
       - KEEP_CANYON_MASS：保留摩天大廈主體外框與天際線，平滑內部 LED 文字細格。
    """
    height, width = orig_bgr.shape[:2]
    policy_matrix = plan_data.get("policy_matrix", {})
    geo_profile = plan_data.get("geography_profile", {})
    contracts = plan_data.get("region_contracts", [])
    vertical_policy = policy_matrix.get(
        "vertical_mass_policy", "TRUNCATE_TO_LOW_RISE"
    )

    max_arch_ratio = float(
        geo_profile.get(
            "max_architecture_height_ratio", DEFAULT_HISTORICAL_CEILING_RATIO
        )
    )
    base_ceiling_y = int(
        np.clip(
            height * (1.0 - max_arch_ratio),
            MIN_CEILING_MARGIN_PX,
            height - MIN_CEILING_MARGIN_PX,
        )
    )
    max_expand_y = int(height * MAX_DYNAMIC_CEILING_Y_RATIO)
    ceiling_y = int(
        np.clip(
            base_ceiling_y + ceiling_expand_px,
            MIN_CEILING_MARGIN_PX,
            max_expand_y,
        )
    )

    vp_x_ratio = float(
        geo_profile.get("vanishing_point_x_ratio", DEFAULT_VANISHING_POINT_X)
    )
    vp_y_ratio = float(
        geo_profile.get("vanishing_point_y_ratio", DEFAULT_VANISHING_POINT_Y)
    )
    vp_x = int(np.clip(vp_x_ratio * width, 0, width - 1))
    vp_y = int(np.clip(vp_y_ratio * height, 0, height - 1))

    preserved_structure_bgr = cv2.bilateralFilter(
        orig_bgr,
        BILATERAL_FILTER_DIAMETER,
        BILATERAL_SIGMA_COLOR,
        BILATERAL_SIGMA_SPACE,
    )

    if vertical_policy == "TRUNCATE_TO_LOW_RISE":
        facade_damp_bottom_y = max(
            ceiling_y + MIN_CEILING_MARGIN_PX,
            int(height * FACADE_DAMPING_BOTTOM_Y_RATIO),
        )
        kw = _ensure_odd_kernel(MESO_VERTICAL_DAMPING_KERNEL_W)
        kh = _ensure_odd_kernel(MESO_VERTICAL_DAMPING_KERNEL_H)
        damped_meso = cv2.GaussianBlur(preserved_structure_bgr, (kw, kh), 0)
        preserved_structure_bgr[0:facade_damp_bottom_y, :] = damped_meso[
            0:facade_damp_bottom_y, :
        ]
    else:
        canyon_med_k = _ensure_odd_kernel(CANYON_BILLBOARD_MEDIAN_BLUR_K)
        preserved_structure_bgr = cv2.medianBlur(
            preserved_structure_bgr, canyon_med_k
        )

    street_zone_mask = np.zeros((height, width), dtype=np.uint8)
    erase_line_mask = np.zeros((height, width), dtype=np.uint8)

    for contract in contracts:
        if not isinstance(contract, dict):
            continue
        change_op = contract.get("change_operator", "")
        topo_zone = contract.get("topological_zone", "")
        poly = _contract_to_pixel_polygon(contract, width, height)

        if change_op in ("DEMOLISH", "GEOMORPHIC_CHANGE"):
            if ceiling_expand_px > 0 and topo_zone == "SKY_CANVAS":
                # 防輪迴黑名單 #3：僅下移非頂邊之包絡線下緣頂點，永遠鎖死 y=0.0 頂邊！
                lower_pts = poly[:, 0, 1] > TOP_BORDER_LOCK_TOLERANCE_PX
                poly[lower_pts, 0, 1] = np.clip(
                    poly[lower_pts, 0, 1] + ceiling_expand_px, 0, max_expand_y
                )
            cv2.fillPoly(erase_line_mask, [poly], int(MAX_PIXEL_VAL))
        elif topo_zone == "STREET_CORRIDOR":
            cv2.fillPoly(street_zone_mask, [poly], int(MAX_PIXEL_VAL))

    if np.any(street_zone_mask > 0):
        med_k = _ensure_odd_kernel(MICRO_ENTITY_MEDIAN_BLUR_K)
        macro_street = cv2.medianBlur(preserved_structure_bgr, med_k)
        preserved_structure_bgr[street_zone_mask > 0] = macro_street[
            street_zone_mask > 0
        ]

    row_mean_bgr = np.mean(orig_bgr, axis=1, keepdims=True).astype(np.uint8)
    horizontally_uniform_bgr = np.repeat(row_mean_bgr, width, axis=1)
    v_blur_k = _ensure_odd_kernel(ERASE_VERTICAL_SMOOTH_KERNEL)
    erased_geometry_bgr = cv2.GaussianBlur(
        horizontally_uniform_bgr, (1, v_blur_k), 0
    )

    erase_line_mask = cv2.bitwise_and(
        erase_line_mask, cv2.bitwise_not(composite_lock_mask)
    )

    trans_k = _ensure_odd_kernel(ERASE_TRANSITION_FEATHER_KERNEL)
    soft_erase_alpha = (
        cv2.GaussianBlur(erase_line_mask, (trans_k, trans_k), 0).astype(
            np.float32
        )
        / MAX_PIXEL_VAL
    )[..., np.newaxis]

    structural_map = (
        soft_erase_alpha * erased_geometry_bgr.astype(np.float32)
        + (1.0 - soft_erase_alpha) * preserved_structure_bgr.astype(np.float32)
    ).astype(np.uint8)

    return structural_map, vp_x, vp_y, ceiling_y


# ==============================================================================
# 5. Stage 2 核心引擎 A：純淨基底 + 嚴格純二值遮罩 (0/255) 張量生成
# ==============================================================================
def build_unified_global_tensors(
    image_bytes: bytes,
    plan_data: Dict[str, Any],
    adaptive_feedback: Optional[Dict[str, Any]] = None,
    best_candidate_bytes: Optional[bytes] = None,
) -> DecoupledLayerTensors:
    """
    v10.5 單一真實純二值遮罩建構器 (遵守黑名單 #1, #2, #8)：
    1. 徹底拔除畫布預塗色塊與隱藏鎖定遮罩：Pass 1 直接送入純淨標準化原圖給 Node 10 VAEEncodeForInpaint。
    2. composite_lock_mask 嚴格只記錄 change_operator == 'KEEP' 的跨時代不變區 (僅此區從現代原圖貼回)。
    3. 重試輪次若存在已合格之非 KEEP 分區，僅從 best_candidate_bytes 繼承保護，絕不從現代原圖貼回！
    4. inpaint_noise_mask 嚴格純二值化 (僅含 0 與 255，灰階洩漏 = 0.0%)。
    """
    orig_bgr = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    if orig_bgr is None:
        raise ValueError("無法解碼輸入影像以建立全域錨定張量")

    height, width = orig_bgr.shape[:2]
    policy_matrix = plan_data.get("policy_matrix", {})
    contracts = plan_data.get("region_contracts", [])

    vertical_policy = policy_matrix.get(
        "vertical_mass_policy", "TRUNCATE_TO_LOW_RISE"
    )
    backdrop_policy = policy_matrix.get(
        "backdrop_sky_policy", "RELEASE_UPPER_SKY"
    )

    retry_idx = 0
    stories_exceeded = False
    verified_passing_zones: List[str] = []
    if isinstance(adaptive_feedback, dict):
        retry_idx = int(adaptive_feedback.get("retry_index", 0))
        stories_exceeded = bool(
            adaptive_feedback.get("stories_exceeded", False)
        )
        verified_passing_zones = list(
            adaptive_feedback.get("verified_passing_zones", [])
        )

    candidate_bgr: Optional[np.ndarray] = None
    if best_candidate_bytes is not None:
        decoded_cand = cv2.imdecode(
            np.frombuffer(best_candidate_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        if decoded_cand is not None and decoded_cand.shape == orig_bgr.shape:
            candidate_bgr = decoded_cand

    composite_lock_mask = np.zeros((height, width), dtype=np.uint8)
    candidate_verified_lock_mask = np.zeros((height, width), dtype=np.uint8)
    active_region_ids: List[str] = []
    active_zones: List[str] = []

    for contract in contracts:
        if not isinstance(contract, dict):
            continue
        change_op = contract.get("change_operator", "")
        action = contract.get("action_type", "")
        topo_zone = contract.get("topological_zone", "")
        gen_inputs = contract.get("generator_inputs", {})
        post_rule = gen_inputs.get("post_composite_rule", "")
        mask_mode = gen_inputs.get(
            "inpaint_mask_mode", contract.get("inpaint_mask_mode", "")
        )

        poly = _contract_to_pixel_polygon(contract, width, height)
        poly_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly], int(MAX_PIXEL_VAL))

        is_true_keep_zone = (
            change_op == "KEEP"
            or action == "DETERMINISTIC_LOCK"
            or post_rule == "PIXEL_EXACT_COPY_BACK"
            or mask_mode == "ZERO_LOCK"
        )

        if is_true_keep_zone:
            if (
                topo_zone == "SKY_CANVAS"
                and backdrop_policy == "KEEP_EXISTING_SKY"
            ):
                true_sky = _extract_true_sky_lock_mask(orig_bgr, poly_mask)
                composite_lock_mask = cv2.bitwise_or(
                    composite_lock_mask, true_sky
                )
            else:
                composite_lock_mask = cv2.bitwise_or(
                    composite_lock_mask, poly_mask
                )
        elif (
            retry_idx > 0
            and candidate_bgr is not None
            and topo_zone in verified_passing_zones
        ):
            candidate_verified_lock_mask = cv2.bitwise_or(
                candidate_verified_lock_mask, poly_mask
            )
        else:
            active_region_ids.append(contract.get("region_id", "region"))
            if topo_zone and topo_zone not in active_zones:
                active_zones.append(topo_zone)

    ceiling_expand_px = (
        int(height * ADAPTIVE_CEILING_EXPAND_RATIO * retry_idx)
        if stories_exceeded
        else 0
    )

    # 純淨基底畫布 (黑名單 #2：絕不預塗人工藍色或水體色塊)
    prepared_base_bgr = orig_bgr.copy()
    if candidate_bgr is not None and np.any(candidate_verified_lock_mask > 0):
        prepared_base_bgr[candidate_verified_lock_mask > 0] = candidate_bgr[
            candidate_verified_lock_mask > 0
        ]

    total_zero_noise_mask = cv2.bitwise_or(
        composite_lock_mask, candidate_verified_lock_mask
    )

    structural_map, _, _, ceiling_y = _synthesize_frequency_modulated_control_map(
        orig_bgr,
        plan_data,
        composite_lock_mask,
        ceiling_expand_px=ceiling_expand_px,
    )

    # 嚴格純二值遮罩 (0=鎖定, 255=重繪；零 1~254 灰階洩漏)
    if np.any(total_zero_noise_mask > 0):
        dilate_k = np.ones((TIER2_DILATE_KERNEL, TIER2_DILATE_KERNEL), np.uint8)
        expanded_lock = cv2.dilate(
            total_zero_noise_mask, dilate_k, iterations=MASK_DILATE_ITERATIONS
        )
        inpaint_mask = _binarize_mask_strict(cv2.bitwise_not(expanded_lock))
    else:
        inpaint_mask = np.full(
            (height, width), BINARY_MASK_FULL_REWRITE, dtype=np.uint8
        )

    if vertical_policy == "DEMOLISH_TO_WATER_LEVEL":
        base_denoise = GLOBAL_DENOISE_DEMOLISH
        base_strength = CTRL_STRENGTH_DEMOLISH
        base_end_pct = CTRL_END_PCT_DEMOLISH
    elif vertical_policy == "KEEP_CANYON_MASS":
        base_denoise = GLOBAL_DENOISE_CANYON_RESTYLE
        base_strength = CTRL_STRENGTH_CANYON
        base_end_pct = CTRL_END_PCT_CANYON
    else:
        base_denoise = GLOBAL_DENOISE_TRUNCATE
        base_strength = CTRL_STRENGTH_TRUNCATE
        base_end_pct = CTRL_END_PCT_TRUNCATE

    base_ipa_weight = float(
        plan_data.get("ipadapter_weight", DEFAULT_IPADAPTER_WEIGHT)
    )
    base_ipa_end_at = float(
        plan_data.get("ipadapter_end_at", DEFAULT_IPADAPTER_END_AT)
    )

    rec_denoise = min(
        MAX_ADAPTIVE_DENOISE,
        base_denoise + (retry_idx * ADAPTIVE_DENOISE_BOOST_STEP),
    )
    rec_strength = max(
        MIN_ADAPTIVE_CTRL_STRENGTH,
        base_strength - (retry_idx * ADAPTIVE_CTRL_STRENGTH_STEP),
    )
    rec_end_pct = max(
        MIN_ADAPTIVE_CTRL_END_PCT,
        base_end_pct - (retry_idx * ADAPTIVE_CTRL_END_PCT_STEP),
    )
    rec_ipa_weight = min(
        MAX_ADAPTIVE_IPADAPTER_WEIGHT,
        base_ipa_weight + (retry_idx * ADAPTIVE_IPADAPTER_WEIGHT_BOOST),
    )

    total_px = float(height * width)
    protected_ratio = float(
        np.sum(inpaint_mask == BINARY_MASK_ZERO_LOCK)
    ) / total_px
    rewrite_ratio = float(
        np.sum(inpaint_mask == BINARY_MASK_FULL_REWRITE)
    ) / total_px

    logger.info(
        f"🌐 [Stage 2 純二值遮罩與宏觀控制張量] 白(重繪255): {rewrite_ratio * PERCENTAGE_SCALE:.1f}% | "
        f"黑(鎖定0): {protected_ratio * PERCENTAGE_SCALE:.1f}% | 透視天際包絡 Y={ceiling_y}px | "
        f"Denoise={rec_denoise:.2f} | CtrlStrength={rec_strength:.2f} (EndPct={rec_end_pct:.2f}) | "
        f"IPAdapterWeight={rec_ipa_weight:.2f}"
    )

    _, base_buf = cv2.imencode(".png", prepared_base_bgr)
    _, inpaint_buf = cv2.imencode(".png", inpaint_mask)
    _, ctrl_buf = cv2.imencode(".png", structural_map)
    _, lock_buf = cv2.imencode(".png", composite_lock_mask)

    return {
        "prepared_base_canvas_bytes": base_buf.tobytes(),
        "inpaint_noise_mask_bytes": inpaint_buf.tobytes(),
        "structural_control_map_bytes": ctrl_buf.tobytes(),
        "composite_protection_mask_bytes": lock_buf.tobytes(),
        "recommended_denoise": round(rec_denoise, 2),
        "recommended_controlnet_strength": round(rec_strength, 2),
        "recommended_controlnet_end_percent": round(rec_end_pct, 2),
        "recommended_ipadapter_weight": round(rec_ipa_weight, 2),
        "recommended_ipadapter_end_at": round(base_ipa_end_at, 2),
        "rewrite_coverage_ratio": round(rewrite_ratio, 4),
        "protected_coverage_ratio": round(protected_ratio, 4),
        "allowed_max_repair_ratio": 1.0,
        "active_region_ids": active_region_ids or ["ALL_ACTIVE_ZONES"],
        "repaired_topological_zones": active_zones,
    }


# ==============================================================================
# 6. Stage 2 核心引擎 B：P1 多缺陷聯合局部修補張量生成 (嚴格純二值 0/255)
# ==============================================================================
def select_single_priority_defect_zone(
    primary_defect_zone: str,
    failed_topological_zones: List[str],
    already_repaired_zones: List[str],
) -> str:
    candidates: List[str] = []
    if (
        primary_defect_zone
        and primary_defect_zone != "NONE"
        and primary_defect_zone in DEFECT_ZONE_PRIORITY_ORDER
    ):
        candidates.append(primary_defect_zone)

    for z in DEFECT_ZONE_PRIORITY_ORDER:
        if z in failed_topological_zones and z not in candidates:
            candidates.append(z)

    for c in candidates:
        if c not in already_repaired_zones:
            return c

    return candidates[0] if candidates else "STREET_CORRIDOR"


def build_targeted_repair_tensors(
    current_canvas_bytes: bytes,
    original_bytes: bytes,
    plan_data: Dict[str, Any],
    failed_region_ids: List[str],
    failed_topological_zones: List[str],
    target_single_zone: Optional[str] = None,
) -> DecoupledLayerTensors:
    canvas_bgr = cv2.imdecode(
        np.frombuffer(current_canvas_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    orig_bgr = cv2.imdecode(
        np.frombuffer(original_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    if canvas_bgr is None or orig_bgr is None:
        raise ValueError("無法解碼影像以建立局部精準修補張量")

    height, width = canvas_bgr.shape[:2]
    total_pixels = float(height * width)
    contracts = plan_data.get("region_contracts", [])
    policy_matrix = plan_data.get("policy_matrix", {})
    backdrop_policy = policy_matrix.get("backdrop_sky_policy", "")

    composite_lock_mask = np.zeros((height, width), dtype=np.uint8)
    for contract in contracts:
        if (
            contract.get("change_operator") == "KEEP"
            or contract.get("action_type") == "DETERMINISTIC_LOCK"
        ):
            poly = _contract_to_pixel_polygon(contract, width, height)
            poly_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [poly], int(MAX_PIXEL_VAL))
            if (
                contract.get("topological_zone") == "SKY_CANVAS"
                and backdrop_policy == "KEEP_EXISTING_SKY"
            ):
                true_sky = _extract_true_sky_lock_mask(orig_bgr, poly_mask)
                composite_lock_mask = cv2.bitwise_or(
                    composite_lock_mask, true_sky
                )
            else:
                composite_lock_mask = cv2.bitwise_or(
                    composite_lock_mask, poly_mask
                )

    structural_map, vp_x, vp_y, ceiling_y = (
        _synthesize_frequency_modulated_control_map(
            orig_bgr, plan_data, composite_lock_mask
        )
    )

    ordered_targets: List[str] = []
    if target_single_zone and target_single_zone != "NONE":
        ordered_targets.append(target_single_zone)
    for fz in failed_topological_zones:
        if fz and fz != "NONE" and fz not in ordered_targets:
            ordered_targets.append(fz)
    if not ordered_targets:
        ordered_targets = ["STREET_CORRIDOR"]

    repair_mask = np.zeros((height, width), dtype=np.uint8)
    matched_region_ids: List[str] = []
    included_zones: List[str] = []

    for z_name in ordered_targets:
        zone_mask = np.zeros((height, width), dtype=np.uint8)
        zone_ids: List[str] = []
        for contract in contracts:
            if (
                not isinstance(contract, dict)
                or contract.get("change_operator") == "KEEP"
            ):
                continue
            c_id = contract.get("region_id", "")
            c_zone = contract.get("topological_zone", "")
            if c_zone == z_name or (
                failed_region_ids and c_id in failed_region_ids
            ):
                poly = _contract_to_pixel_polygon(contract, width, height)
                cv2.fillPoly(zone_mask, [poly], int(MAX_PIXEL_VAL))
                zone_ids.append(c_id or z_name)

        if not np.any(zone_mask > 0):
            road_top_y = max(
                ceiling_y + MIN_CLAMP_DIM_PX,
                int(vp_y + height * FALLBACK_ROAD_TOP_OFFSET_RATIO),
            )
            corridor_half_w = max(
                MIN_CEILING_MARGIN_PX,
                int(width * FALLBACK_CORRIDOR_HALF_W_RATIO),
            )
            if z_name == "SKY_CANVAS":
                pts = [[0, 0], [width, 0], [width, ceiling_y], [0, ceiling_y]]
            elif z_name == "STREET_CORRIDOR":
                pts = [
                    [max(0, vp_x - corridor_half_w), road_top_y],
                    [min(width, vp_x + corridor_half_w), road_top_y],
                    [width, height],
                    [0, height],
                ]
            elif z_name == "LEFT_FACADE":
                pts = [
                    [0, 0],
                    [max(1, vp_x - corridor_half_w // 2), ceiling_y],
                    [max(1, vp_x - corridor_half_w // 2), road_top_y],
                    [0, height],
                ]
            elif z_name == "RIGHT_FACADE":
                pts = [
                    [min(width - 1, vp_x + corridor_half_w // 2), ceiling_y],
                    [width, 0],
                    [width, height],
                    [min(width - 1, vp_x + corridor_half_w // 2), road_top_y],
                ]
            else:
                pad_x = max(
                    MIN_CEILING_MARGIN_PX,
                    int(width * FALLBACK_CENTRAL_PAD_X_RATIO),
                )
                pts = [
                    [max(0, vp_x - pad_x), ceiling_y],
                    [min(width, vp_x + pad_x), ceiling_y],
                    [min(width, vp_x + pad_x), road_top_y],
                    [max(0, vp_x - pad_x), road_top_y],
                ]
            fallback_poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(zone_mask, [fallback_poly], int(MAX_PIXEL_VAL))
            zone_ids.append(z_name)

        zone_mask = _clamp_zone_mask_preserving_canvas_borders(
            zone_mask, z_name, BASE_SINGLE_REPAIR_MAX_RATIO
        )
        candidate_union = cv2.bitwise_or(repair_mask, zone_mask)
        union_ratio = float(np.count_nonzero(candidate_union)) / total_pixels

        if (
            not included_zones
            or union_ratio
            <= MAX_JOINT_REPAIR_AREA_RATIO - REPAIR_ENVELOPE_MARGIN
        ):
            repair_mask = candidate_union
            included_zones.append(z_name)
            matched_region_ids.extend(zone_ids)

    dilate_k = np.ones(
        (TARGETED_REPAIR_DILATE_KERNEL, TARGETED_REPAIR_DILATE_KERNEL), np.uint8
    )
    repair_mask = cv2.dilate(
        repair_mask, dilate_k, iterations=MASK_DILATE_ITERATIONS
    )

    lock_dilate_k = np.ones((TIER2_DILATE_KERNEL, TIER2_DILATE_KERNEL), np.uint8)
    expanded_lock = cv2.dilate(
        composite_lock_mask, lock_dilate_k, iterations=MASK_DILATE_ITERATIONS
    )
    repair_mask = cv2.bitwise_and(repair_mask, cv2.bitwise_not(expanded_lock))
    binary_repair_mask = _binarize_mask_strict(repair_mask)

    rewrite_ratio = float(
        np.sum(binary_repair_mask == BINARY_MASK_FULL_REWRITE)
    ) / total_pixels
    protected_ratio = 1.0 - rewrite_ratio

    base_ipa_weight = float(
        plan_data.get("ipadapter_weight", DEFAULT_IPADAPTER_WEIGHT)
    )
    base_ipa_end_at = float(
        plan_data.get("ipadapter_end_at", DEFAULT_IPADAPTER_END_AT)
    )

    logger.info(
        f"🎯 [P1 聯合局部精準修補張量] 修補分區: {included_zones} (IDs={matched_region_ids}) | "
        f"白(重繪255): {rewrite_ratio * PERCENTAGE_SCALE:.1f}% (<={MAX_JOINT_REPAIR_AREA_RATIO * PERCENTAGE_SCALE:.0f}% 上限合規) | "
        f"黑(鎖定0): {protected_ratio * PERCENTAGE_SCALE:.1f}%"
    )

    _, base_buf = cv2.imencode(".png", canvas_bgr)
    _, inpaint_buf = cv2.imencode(".png", binary_repair_mask)
    _, ctrl_buf = cv2.imencode(".png", structural_map)
    _, lock_buf = cv2.imencode(".png", composite_lock_mask)

    return {
        "prepared_base_canvas_bytes": base_buf.tobytes(),
        "inpaint_noise_mask_bytes": inpaint_buf.tobytes(),
        "structural_control_map_bytes": ctrl_buf.tobytes(),
        "composite_protection_mask_bytes": lock_buf.tobytes(),
        "recommended_denoise": TARGETED_REPAIR_DENOISE,
        "recommended_controlnet_strength": CTRL_STRENGTH_TARGETED_REPAIR,
        "recommended_controlnet_end_percent": CTRL_END_PCT_TARGETED_REPAIR,
        "recommended_ipadapter_weight": round(base_ipa_weight, 2),
        "recommended_ipadapter_end_at": round(base_ipa_end_at, 2),
        "rewrite_coverage_ratio": round(rewrite_ratio, 4),
        "protected_coverage_ratio": round(protected_ratio, 4),
        "allowed_max_repair_ratio": MAX_JOINT_REPAIR_AREA_RATIO,
        "active_region_ids": matched_region_ids,
        "repaired_topological_zones": included_zones,
    }


def build_layer_tensors(
    image_bytes: bytes,
    plan_data: Dict[str, Any],
    target_layer_index: Optional[int] = None,
) -> DecoupledLayerTensors:
    return build_unified_global_tensors(image_bytes, plan_data)


# ==============================================================================
# 7. 階段 A 專用：遮罩可視化存圖與純二值／無重疊審計工具
# ==============================================================================
def generate_mask_diagnostic_artifacts(
    base_canvas_bytes: bytes,
    inpaint_noise_mask_bytes: bytes,
    composite_protection_mask_bytes: bytes,
    plan_data: Dict[str, Any],
) -> Tuple[bytes, Dict[str, Any]]:
    """
    產出階段 A 隔離測試專用之可視化遮罩疊加圖 (debug_mask_overlay.png) 與量化審計報表 (debug_mask_stats.json)：
    1. 核對 base / inpaint_mask / lock_mask 三張圖之像素尺寸是否 100% 一致。
    2. 核對 inpaint_mask 黑白方向與純二值純淨度 (white_255_pct / black_0_pct / grey_1_to_254_pct)。
    3. 核對各分區多邊形之間是否存在重疊像素 (overlap_pixels) 及每區實際落入重繪/鎖定遮罩之佔比。
    4. 疊加半透明紅色 (重繪=255)、半透明綠色 (鎖定=0) 與黃色分區多邊形框線供肉眼核查。
    """
    base_bgr = cv2.imdecode(
        np.frombuffer(base_canvas_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    inpaint_mask = cv2.imdecode(
        np.frombuffer(inpaint_noise_mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE
    )
    lock_mask = cv2.imdecode(
        np.frombuffer(composite_protection_mask_bytes, np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )

    if base_bgr is None or inpaint_mask is None or lock_mask is None:
        return base_canvas_bytes, {
            "error": "無法解碼遮罩診斷影像",
            "dimensions_aligned": False,
        }

    height, width = base_bgr.shape[:2]
    total_px = float(height * width)
    dimensions_aligned = (
        inpaint_mask.shape == (height, width)
        and lock_mask.shape == (height, width)
    )

    white_px = int(np.sum(inpaint_mask == BINARY_MASK_FULL_REWRITE))
    black_px = int(np.sum(inpaint_mask == BINARY_MASK_ZERO_LOCK))
    grey_px = int(
        np.sum(
            (inpaint_mask > BINARY_MASK_ZERO_LOCK)
            & (inpaint_mask < BINARY_MASK_FULL_REWRITE)
        )
    )
    keep_lock_px = int(np.sum(lock_mask >= CORE_MASK_BINARY_THRESHOLD))

    white_pct = round((white_px / total_px) * PERCENTAGE_SCALE, 2)
    black_pct = round((black_px / total_px) * PERCENTAGE_SCALE, 2)
    grey_pct = round((grey_px / total_px) * PERCENTAGE_SCALE, 4)
    keep_lock_pct = round((keep_lock_px / total_px) * PERCENTAGE_SCALE, 2)

    # 計算分區覆蓋度與重疊像素審計
    contracts = plan_data.get("region_contracts", [])
    coverage_counter = np.zeros((height, width), dtype=np.int32)
    per_zone_stats: Dict[str, Dict[str, Any]] = {}

    overlay_tint = np.zeros_like(base_bgr)
    if dimensions_aligned:
        overlay_tint[inpaint_mask == BINARY_MASK_FULL_REWRITE] = (
            OVERLAY_REWRITE_BGR
        )
        overlay_tint[inpaint_mask == BINARY_MASK_ZERO_LOCK] = OVERLAY_LOCK_BGR
    overlay_bgr = cv2.addWeighted(
        base_bgr, OVERLAY_BASE_ALPHA, overlay_tint, OVERLAY_MASK_ALPHA, 0.0
    )

    for contract in contracts:
        if not isinstance(contract, dict):
            continue
        z_name = str(contract.get("topological_zone", "UNKNOWN"))
        change_op = str(contract.get("change_operator", "UNKNOWN"))
        poly = _contract_to_pixel_polygon(contract, width, height)

        z_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(z_mask, [poly], int(MAX_PIXEL_VAL))
        z_area_px = int(np.count_nonzero(z_mask))
        coverage_counter[z_mask > 0] += 1

        if z_area_px > 0 and dimensions_aligned:
            z_rewrite_px = int(
                np.sum(
                    (z_mask > 0) & (inpaint_mask == BINARY_MASK_FULL_REWRITE)
                )
            )
            z_lock_px = int(
                np.sum((z_mask > 0) & (inpaint_mask == BINARY_MASK_ZERO_LOCK))
            )
            per_zone_stats[z_name] = {
                "change_operator": change_op,
                "zone_area_pct": round(
                    (z_area_px / total_px) * PERCENTAGE_SCALE, 2
                ),
                "in_rewrite_mask_255_pct": round(
                    (z_rewrite_px / float(z_area_px)) * PERCENTAGE_SCALE, 2
                ),
                "in_lock_mask_0_pct": round(
                    (z_lock_px / float(z_area_px)) * PERCENTAGE_SCALE, 2
                ),
            }

        cv2.polylines(
            overlay_bgr,
            [poly],
            isClosed=True,
            color=OVERLAY_POLYGON_BGR,
            thickness=OVERLAY_LINE_THICKNESS,
        )
        xs = poly[:, 0, 0]
        ys = poly[:, 0, 1]
        cx = int(
            np.clip(
                int(np.mean(xs)) - OVERLAY_TEXT_MARGIN_PX * 3,
                OVERLAY_TEXT_MARGIN_PX,
                max(OVERLAY_TEXT_MARGIN_PX, width - 220),
            )
        )
        cy = int(
            np.clip(
                int(np.mean(ys)),
                OVERLAY_TEXT_MARGIN_PX * 2,
                height - OVERLAY_TEXT_MARGIN_PX,
            )
        )
        label_text = f"{z_name} [{change_op}]"
        (tw, th), _ = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            OVERLAY_FONT_SCALE,
            OVERLAY_TEXT_THICKNESS,
        )
        cv2.rectangle(
            overlay_bgr,
            (cx - OVERLAY_LABEL_PAD_PX, cy - th - OVERLAY_LABEL_PAD_PX),
            (cx + tw + OVERLAY_LABEL_PAD_PX, cy + OVERLAY_LABEL_PAD_PX),
            OVERLAY_LABEL_BG_BGR,
            -1,
        )
        cv2.putText(
            overlay_bgr,
            label_text,
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            OVERLAY_FONT_SCALE,
            OVERLAY_LABEL_TEXT_BGR,
            OVERLAY_TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    overlap_px = int(np.sum(coverage_counter > 1))
    uncovered_px = int(np.sum(coverage_counter == 0))

    stats_report: Dict[str, Any] = {
        "dimensions_aligned": dimensions_aligned,
        "canvas_size_wh": [width, height],
        "inpaint_mask_size_wh": [
            int(inpaint_mask.shape[1]),
            int(inpaint_mask.shape[0]),
        ],
        "mask_polarity_convention": "255(WHITE/RED)=REWRITE, 0(BLACK/GREEN)=LOCK_PROTECTED",
        "is_strict_binary": grey_px == 0,
        "white_rewrite_255_pct": white_pct,
        "black_lock_0_pct": black_pct,
        "grey_1_to_254_leak_pct": grey_pct,
        "keep_original_lock_pct": keep_lock_pct,
        "polygon_overlap_pct": round(
            (overlap_px / total_px) * PERCENTAGE_SCALE, 4
        ),
        "polygon_uncovered_pct": round(
            (uncovered_px / total_px) * PERCENTAGE_SCALE, 4
        ),
        "per_zone_mask_coverage": per_zone_stats,
    }

    _, overlay_buf = cv2.imencode(".png", overlay_bgr)
    return overlay_buf.tobytes(), stats_report


# ==============================================================================
# 8. 確定性雙階段像素合成、非天空清晰度評分與逐分區反原圖複製物理量測
# ==============================================================================
def composite_locked_regions(
    original_bytes: bytes,
    generated_bytes: bytes,
    composite_protection_mask_bytes: bytes,
    base_canvas_bytes: Optional[bytes] = None,
    inpaint_noise_mask_bytes: Optional[bytes] = None,
) -> bytes:
    orig_bgr = cv2.imdecode(
        np.frombuffer(original_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    gen_bgr = cv2.imdecode(
        np.frombuffer(generated_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    lock_mask = cv2.imdecode(
        np.frombuffer(composite_protection_mask_bytes, np.uint8),
        cv2.IMREAD_GRAYSCALE,
    )

    if orig_bgr is None or gen_bgr is None or lock_mask is None:
        return generated_bytes
    if orig_bgr.shape != gen_bgr.shape or orig_bgr.shape[:2] != lock_mask.shape:
        return generated_bytes

    working_bgr = gen_bgr.astype(np.float32)

    if base_canvas_bytes is not None and inpaint_noise_mask_bytes is not None:
        base_bgr = cv2.imdecode(
            np.frombuffer(base_canvas_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        inpaint_mask = cv2.imdecode(
            np.frombuffer(inpaint_noise_mask_bytes, np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        if (
            base_bgr is not None
            and inpaint_mask is not None
            and base_bgr.shape == gen_bgr.shape
            and inpaint_mask.shape == gen_bgr.shape[:2]
        ):
            # 在最終像素合成階段做窄邊羽化，使鎖定區與重繪區自然過渡，同時維持送入 ComfyUI 的遮罩純二值
            if np.any(inpaint_mask == BINARY_MASK_ZERO_LOCK):
                seam_k = _ensure_odd_kernel(COMPOSITE_FEATHER_KERNEL)
                soft_inpaint = cv2.GaussianBlur(
                    inpaint_mask, (seam_k, seam_k), 0
                )
                erode_k = np.ones(
                    (TIER2_DILATE_KERNEL, TIER2_DILATE_KERNEL), np.uint8
                )
                core_zero = cv2.erode(
                    (inpaint_mask == BINARY_MASK_ZERO_LOCK).astype(np.uint8)
                    * int(MAX_PIXEL_VAL),
                    erode_k,
                    iterations=MASK_DILATE_ITERATIONS,
                )
                soft_inpaint[core_zero == int(MAX_PIXEL_VAL)] = (
                    BINARY_MASK_ZERO_LOCK
                )
                inpaint_alpha = (
                    soft_inpaint.astype(np.float32) / MAX_PIXEL_VAL
                )[..., np.newaxis]
            else:
                inpaint_alpha = (
                    inpaint_mask.astype(np.float32) / MAX_PIXEL_VAL
                )[..., np.newaxis]

            working_bgr = (
                inpaint_alpha * working_bgr
                + (1.0 - inpaint_alpha) * base_bgr.astype(np.float32)
            )

    if np.any(lock_mask > 0):
        feather_k = _ensure_odd_kernel(COMPOSITE_FEATHER_KERNEL)
        soft_lock = cv2.GaussianBlur(lock_mask, (feather_k, feather_k), 0)
        erode_k = np.ones((TIER2_DILATE_KERNEL, TIER2_DILATE_KERNEL), np.uint8)
        core_lock = cv2.erode(
            lock_mask, erode_k, iterations=MASK_DILATE_ITERATIONS
        )
        soft_lock[core_lock == int(MAX_PIXEL_VAL)] = int(MAX_PIXEL_VAL)

        lock_alpha = (soft_lock.astype(np.float32) / MAX_PIXEL_VAL)[
            ..., np.newaxis
        ]
        working_bgr = (
            lock_alpha * orig_bgr.astype(np.float32)
            + (1.0 - lock_alpha) * working_bgr
        )

    composited = np.clip(working_bgr, 0, int(MAX_PIXEL_VAL)).astype(np.uint8)
    _, out_buf = cv2.imencode(".png", composited)
    return out_buf.tobytes()


def compute_candidate_clarity_score(image_bytes: bytes) -> float:
    """防輪迴黑名單 #9：僅對下半部建築與街道區域 (Y >= CLARITY_EVAL_Y_MIN_RATIO) 計算清晰度，不懲罰開闊藍天。"""
    bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return 0.0
    height = bgr.shape[0]
    eval_y_start = int(height * CLARITY_EVAL_Y_MIN_RATIO)
    roi_bgr = bgr[eval_y_start:height, :]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    contrast_std = float(np.std(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    contrast_score = min(
        CLARITY_MAX_HALF_SCORE,
        (contrast_std / CLARITY_REF_CONTRAST_STD) * CLARITY_MAX_HALF_SCORE,
    )
    sharpness_score = min(
        CLARITY_MAX_HALF_SCORE,
        (laplacian_var / CLARITY_REF_LAPLACIAN_VAR) * CLARITY_MAX_HALF_SCORE,
    )
    return round(contrast_score + sharpness_score, 2)


def evaluate_geometric_compliance(
    generated_bytes: bytes,
    plan_data: Dict[str, Any],
    original_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    v10.5 Stage 4 客觀物理量測與「非天空分區反原圖複製審計」：
    1. 檢查 RELEASE_UPPER_SKY 模式下頂部天際區是否殘留高樓邊緣 (cv_sky_compliant)。
    2. 逐個檢查所有非 KEEP 且非 SKY_CANVAS 之實體分區相對於現代原圖的真實像素差異。
    """
    gen_bgr = cv2.imdecode(
        np.frombuffer(generated_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    if gen_bgr is None:
        return {
            "cv_overall_compliant": True,
            "cv_sky_compliant": True,
            "haze_washout_detected": False,
            "unrewritten_central_void_detected": False,
            "under_restyled_zones": [],
            "zone_rewrite_diffs": {},
            "upper_edge_density": 0.0,
            "corridor_structure_retention": DEFAULT_FALLBACK_CORRIDOR_RETENTION,
            "failed_topological_zones": [],
        }

    height, width = gen_bgr.shape[:2]
    policy_matrix = plan_data.get("policy_matrix", {})
    geo_profile = plan_data.get("geography_profile", {})
    contracts = plan_data.get("region_contracts", [])
    backdrop_policy = policy_matrix.get("backdrop_sky_policy", "")

    max_arch_ratio = float(
        geo_profile.get(
            "max_architecture_height_ratio", DEFAULT_HISTORICAL_CEILING_RATIO
        )
    )
    contrast_y_start = int(height * CONTRAST_CHECK_Y_MIN_RATIO)
    ceiling_y = int(
        np.clip(
            height * (1.0 - max_arch_ratio) * UPPER_SKY_CHECK_HEIGHT_FACTOR,
            MIN_CEILING_MARGIN_PX,
            contrast_y_start,
        )
    )

    gray = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2GRAY)
    upper_edges = cv2.Canny(
        gray[0:ceiling_y, :], CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD
    )
    overall_edge_density = float(np.count_nonzero(upper_edges)) / float(
        max(1, upper_edges.size)
    )

    failed_zones: List[str] = []
    under_restyled_zones: List[str] = []
    zone_rewrite_diffs: Dict[str, float] = {}

    cv_sky_compliant = True
    if backdrop_policy == "RELEASE_UPPER_SKY":
        c_x1 = int(width * PERSPECTIVE_INNER_CORRIDOR_X_LEFT)
        c_x2 = int(width * PERSPECTIVE_INNER_CORRIDOR_X_RIGHT)
        center_sky_edges = upper_edges[:, c_x1:c_x2]
        center_density = float(np.count_nonzero(center_sky_edges)) / float(
            max(1, center_sky_edges.size)
        )
        if center_density > UPPER_SKY_EDGE_DENSITY_MAX:
            cv_sky_compliant = False
            if "SKY_CANVAS" not in failed_zones:
                failed_zones.append("SKY_CANVAS")

    lower_half_std = float(np.std(gray[contrast_y_start:height, :]))
    haze_washout_detected = lower_half_std < MIN_ACCEPTABLE_CONTRAST_STD
    if haze_washout_detected and "STREET_CORRIDOR" not in failed_zones:
        failed_zones.append("STREET_CORRIDOR")

    unrewritten_central_void_detected = False
    corridor_structure_retention = DEFAULT_FALLBACK_CORRIDOR_RETENTION

    if original_bytes is not None:
        orig_bgr = cv2.imdecode(
            np.frombuffer(original_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        if orig_bgr is not None and orig_bgr.shape == gen_bgr.shape:
            full_diff = cv2.absdiff(orig_bgr, gen_bgr)

            for contract in contracts:
                if not isinstance(contract, dict):
                    continue
                if contract.get("change_operator") == "KEEP":
                    continue
                z_name = contract.get("topological_zone", "")
                if not z_name:
                    continue
                poly = _contract_to_pixel_polygon(contract, width, height)
                z_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(z_mask, [poly], int(MAX_PIXEL_VAL))
                if np.any(z_mask > 0):
                    z_diff_val = round(float(np.mean(full_diff[z_mask > 0])), 2)
                    zone_rewrite_diffs[z_name] = z_diff_val
                    if (
                        z_name != "SKY_CANVAS"
                        and z_diff_val < MIN_ACTIVE_ZONE_REWRITE_DIFF
                    ):
                        unrewritten_central_void_detected = True
                        if z_name not in under_restyled_zones:
                            under_restyled_zones.append(z_name)
                        if z_name not in failed_zones:
                            failed_zones.append(z_name)

            corr_y_start = int(height * CORRIDOR_RETENTION_Y_MIN_RATIO)
            orig_gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
            blur_k = _ensure_odd_kernel(CORRIDOR_BLUR_KERNEL)
            lower_orig = cv2.GaussianBlur(
                orig_gray[corr_y_start:height, :], (blur_k, blur_k), 0
            )
            lower_gen = cv2.GaussianBlur(
                gray[corr_y_start:height, :], (blur_k, blur_k), 0
            )

            luma_corr = 0.0
            corr_matrix = np.corrcoef(
                lower_orig.flatten(), lower_gen.flatten()
            )
            if corr_matrix.shape == (
                CORR_MATRIX_DIM,
                CORR_MATRIX_DIM,
            ) and not np.isnan(corr_matrix[0, 1]):
                luma_corr = max(0.0, float(corr_matrix[0, 1]))

            orig_gx = cv2.Sobel(
                lower_orig, cv2.CV_32F, 1, 0, ksize=SOBEL_KERNEL_SIZE
            )
            orig_gy = cv2.Sobel(
                lower_orig, cv2.CV_32F, 0, 1, ksize=SOBEL_KERNEL_SIZE
            )
            gen_gx = cv2.Sobel(
                lower_gen, cv2.CV_32F, 1, 0, ksize=SOBEL_KERNEL_SIZE
            )
            gen_gy = cv2.Sobel(
                lower_gen, cv2.CV_32F, 0, 1, ksize=SOBEL_KERNEL_SIZE
            )

            orig_mag = np.hypot(orig_gx, orig_gy).flatten()
            gen_mag = np.hypot(gen_gx, gen_gy).flatten()
            denom = float(np.linalg.norm(orig_mag) * np.linalg.norm(gen_mag))
            grad_sim = (
                float(np.dot(orig_mag, gen_mag)) / denom if denom > 0.0 else 0.0
            )

            corridor_structure_retention = round(
                (CORRIDOR_LUMA_WEIGHT * luma_corr)
                + (CORRIDOR_GRADIENT_WEIGHT * grad_sim),
                4,
            )

    cv_overall_compliant = (
        cv_sky_compliant
        and not haze_washout_detected
        and not unrewritten_central_void_detected
    )

    return {
        "cv_overall_compliant": cv_overall_compliant,
        "cv_sky_compliant": cv_sky_compliant,
        "haze_washout_detected": haze_washout_detected,
        "unrewritten_central_void_detected": unrewritten_central_void_detected,
        "under_restyled_zones": under_restyled_zones,
        "zone_rewrite_diffs": zone_rewrite_diffs,
        "upper_edge_density": round(overall_edge_density, 4),
        "corridor_structure_retention": corridor_structure_retention,
        "failed_topological_zones": failed_zones,
    }


# ==============================================================================
# 9. 向下相容輔助函數與真實物理差異計算
# ==============================================================================
def create_tier1_mask(
    img_width: int,
    img_height: int,
    target_bboxes: List[List[int]],
    protected_bboxes: Optional[List[List[int]]] = None,
) -> bytes:
    if protected_bboxes is None:
        protected_bboxes = []
    target_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    protected_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for bbox in target_bboxes:
        y1, x1, y2, x2 = _unnormalize_bbox(bbox, img_width, img_height)
        cv2.rectangle(target_mask, (x1, y1), (x2, y2), int(MAX_PIXEL_VAL), -1)
    for bbox in protected_bboxes:
        y1, x1, y2, x2 = _unnormalize_bbox(bbox, img_width, img_height)
        cv2.rectangle(
            protected_mask, (x1, y1), (x2, y2), int(MAX_PIXEL_VAL), -1
        )
    safe_binary_mask = cv2.bitwise_and(
        target_mask, cv2.bitwise_not(protected_mask)
    )
    dilate_kernel = np.ones((TIER1_DILATE_KERNEL, TIER1_DILATE_KERNEL), np.uint8)
    dilated_mask = cv2.dilate(
        safe_binary_mask, dilate_kernel, iterations=MASK_DILATE_ITERATIONS
    )
    blurred_mask = cv2.GaussianBlur(
        dilated_mask, (TIER1_BLUR_KERNEL, TIER1_BLUR_KERNEL), 0
    )
    final_mask = cv2.bitwise_and(blurred_mask, cv2.bitwise_not(protected_mask))
    _, buffer = cv2.imencode(".png", final_mask)
    return buffer.tobytes()


def prepare_tier2_assets(
    image_bytes: bytes, protected_bboxes: List[List[int]]
) -> Tuple[bytes, bytes]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width = original_img_bgr.shape[:2]

    temp_total_mask = np.zeros((height, width), dtype=np.uint8)
    for bbox in protected_bboxes:
        y1, x1, y2, x2 = _unnormalize_bbox(bbox, width, height)
        cv2.rectangle(
            temp_total_mask, (x1, y1), (x2, y2), int(MAX_PIXEL_VAL), -1
        )

    dilate_kernel = np.ones((TIER2_DILATE_KERNEL, TIER2_DILATE_KERNEL), np.uint8)
    dilated_total = cv2.dilate(
        temp_total_mask, dilate_kernel, iterations=MASK_DILATE_ITERATIONS
    )
    master_rewrite_mask = _binarize_mask_strict(cv2.bitwise_not(dilated_total))

    _, mask_buf = cv2.imencode(".png", master_rewrite_mask)
    return image_bytes, mask_buf.tobytes()


def crop_to_original(image_bytes: bytes, metadata: Dict) -> bytes:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        pad = metadata["padding"]
        target_w, target_h = metadata["target_size"]
        if image.size != (target_w, target_h):
            return image_bytes
        crop_box = (
            pad["left"],
            pad["top"],
            target_w - pad["right"],
            target_h - pad["bottom"],
        )
        cropped_image = image.crop(crop_box)
        orig_w, orig_h = metadata["original_size"]
        if cropped_image.size != (orig_w, orig_h):
            cropped_image = cropped_image.resize(
                (orig_w, orig_h), Image.Resampling.LANCZOS
            )
        output = io.BytesIO()
        cropped_image.save(output, format="PNG")
        return output.getvalue()
    except Exception as e:
        logger.error(f"還原裁切失敗: {str(e)}")
        return image_bytes


def calculate_rewrite_diff(
    base_bytes: bytes, gen_bytes: bytes, mask_bytes: bytes
) -> Tuple[float, float, float]:
    try:
        base_img = cv2.imdecode(
            np.frombuffer(base_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        gen_img = cv2.imdecode(
            np.frombuffer(gen_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        mask_img = cv2.imdecode(
            np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE
        )

        if base_img is None or gen_img is None or mask_img is None:
            return 0.0, 0.0, 0.0
        if (
            base_img.shape != gen_img.shape
            or base_img.shape[:2] != mask_img.shape
        ):
            return 0.0, 0.0, 0.0

        diff = cv2.absdiff(base_img, gen_img)
        rewrite_pixels = mask_img >= CORE_MASK_BINARY_THRESHOLD
        protected_pixels = mask_img < FEATHER_TAIL_THRESHOLD

        rewrite_mse = (
            float(np.mean(diff[rewrite_pixels])) if np.any(rewrite_pixels) else 0.0
        )

        if np.any(protected_pixels):
            protected_diff = diff[protected_pixels]
            significant_diff = protected_diff[protected_diff > MSE_PIXEL_TOLERANCE]
            leakage_ratio = len(significant_diff) / len(protected_diff)
            protected_mse = (
                float(np.mean(protected_diff))
                if leakage_ratio >= MSE_LEAK_RATIO_TOLERANCE
                else 0.0
            )
        else:
            protected_mse = -1.0

        rewrite_std = (
            float(np.std(gen_img[rewrite_pixels]))
            if np.any(rewrite_pixels)
            else 0.0
        )
        return rewrite_mse, protected_mse, rewrite_std
    except Exception as e:
        logger.error(f"計算三維度物理指標時發生錯誤: {e}")
        return 0.0, 0.0, 0.0