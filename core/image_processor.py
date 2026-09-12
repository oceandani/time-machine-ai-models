import io
import math
import cv2
import numpy as np
from typing import Tuple, Dict, TypedDict, List
from PIL import Image, ImageOps
from core.logging import logger

# --- 接口契約 (Pipeline Contract) ---
class ImageMetadata(TypedDict):
    original_size: Tuple[int, int]
    scale_ratio: float
    pre_pad_size: Tuple[int, int]
    target_size: Tuple[int, int]
    padding: Dict[str, int]

def process_and_standardize_image(image_bytes: bytes, max_size: int = 1024) -> Tuple[bytes, Dict]:
    """預處理：修正 EXIF，轉為 RGB，等比縮放，並 Padding 至 64 的倍數"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            background = Image.new('RGB', image.size, (0, 0, 0))
            background.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
            image = background
        else:
            image = image.convert("RGB")
            
        original_width, original_height = image.size
        
        scale_ratio = 1.0
        if original_width > max_size or original_height > max_size:
            scale_ratio = max_size / max(original_width, original_height)
            new_w, new_h = int(original_width * scale_ratio), int(original_height * scale_ratio)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        current_w, current_h = image.size
        target_w = math.ceil(current_w / 64) * 64
        target_h = math.ceil(current_h / 64) * 64
        
        pad_w = target_w - current_w
        pad_h = target_h - current_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        
        if pad_w > 0 or pad_h > 0:
            image = ImageOps.expand(image, padding, fill=(0, 0, 0))
            
        output = io.BytesIO()
        image.save(output, format="PNG")
        
        metadata: ImageMetadata = {
            "original_size": (original_width, original_height),
            "scale_ratio": scale_ratio,
            "pre_pad_size": (current_w, current_h),
            "target_size": (target_w, target_h),
            "padding": {"left": pad_left, "top": pad_top, "right": pad_right, "bottom": pad_bottom}
        }
        return output.getvalue(), metadata
    except Exception as e:
        logger.error(f"圖片預處理失敗: {str(e)}")
        raise RuntimeError(f"圖片預處理失敗: {str(e)}")


def _unnormalize_bbox(bbox: List[int], width: int, height: int) -> Tuple[int, int, int, int]:
    """輔助函數：將 0-1000 座標反歸一化為實際像素"""
    return (
        int(bbox[0] * height / 1000), # y_min
        int(bbox[1] * width / 1000),  # x_min
        int(bbox[2] * height / 1000), # y_max
        int(bbox[3] * width / 1000)   # x_max
    )


def create_tier1_mask(img_width: int, img_height: int, target_bboxes: List[List[int]], protected_bboxes: List[List[int]] = None) -> bytes:
    """[Tier 1 專用] 局部修補的布林遮罩生成"""
    if protected_bboxes is None:
        protected_bboxes = []

    target_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    protected_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for bbox in target_bboxes:
        y1, x1, y2, x2 = _unnormalize_bbox(bbox, img_width, img_height)
        cv2.rectangle(target_mask, (x1, y1), (x2, y2), 255, -1)

    for bbox in protected_bboxes:
        y1, x1, y2, x2 = _unnormalize_bbox(bbox, img_width, img_height)
        cv2.rectangle(protected_mask, (x1, y1), (x2, y2), 255, -1)

    safe_binary_mask = cv2.bitwise_and(target_mask, cv2.bitwise_not(protected_mask))
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(safe_binary_mask, kernel, iterations=1)
    blurred_mask = cv2.GaussianBlur(dilated_mask, (11, 11), 0)
    final_mask = cv2.bitwise_and(blurred_mask, cv2.bitwise_not(protected_mask))

    is_success, buffer = cv2.imencode(".png", final_mask)
    if not is_success: raise ValueError("Tier 1 遮罩編碼失敗")
    return buffer.tobytes()


def apply_tier2_preblanking(image_bytes: bytes, protected_bboxes: List[List[int]]) -> Tuple[bytes, bytes]:
    """
    [藍圖 v6.0 - Tier 2 專用] 像素級底圖前置抹除 (Pre-Blanking)
    嚴格實行 Hard Clamp 與 QA 零污染檢測，摧毀 MiDaS 深度認知。
    回傳: (前置抹除後的底圖 Bytes, 單一真理 master_mask Bytes)
    """
    # 1. 讀取影像
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width = original_img.shape[:2]

    # 2. 絕對遮罩契約 (The Absolute Masking Contract)
    protected_mask = np.zeros((height, width), dtype=np.uint8)
    for bbox in protected_bboxes:
        y1, x1, y2, x2 = _unnormalize_bbox(bbox, width, height)
        cv2.rectangle(protected_mask, (x1, y1), (x2, y2), 255, -1)

    # 邊緣防禦：微幅膨脹保護區，吞噬現代銳利邊緣
    dilate_kernel = np.ones((3, 3), np.uint8)
    dilated_protected = cv2.dilate(protected_mask, dilate_kernel, iterations=1)
    
    # 單一真理：允許重繪區 = 255 - 保護區
    master_mask = cv2.bitwise_not(dilated_protected)

    # 3. 深度欺騙與高頻摧毀 (Depth Deception & High-Frequency Destruction)
    # 3.1 提取 Allowed 區域的 Mean Color 作為穩定基底
    mean_val = cv2.mean(original_img, mask=master_mask)
    mean_color_img = np.zeros_like(original_img)
    mean_color_img[:] = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))

    # 3.2 極限高斯模糊 (99x99)
    blurred_img = cv2.GaussianBlur(original_img, (99, 99), 30)

    # 3.3 混合模糊與均值，徹底壓平 3D 結構
    flat_field = cv2.addWeighted(blurred_img, 0.5, mean_color_img, 0.5, 0)

    # 3.4 高斯白噪聲注入 (破壞殘留的梯度連續性)
    noise = np.zeros_like(flat_field, dtype=np.int16)
    cv2.randn(noise, 0, 15)  # Mean 0, StdDev 15
    noisy_flat_field = cv2.addWeighted(flat_field, 1.0, noise, 1.0, 0, dtype=cv2.CV_8UC3)

    # 4. 影像合成與終極硬箝制 (Composition & Hard Clamp)
    preblanked_img = np.copy(original_img)
    
    # 覆寫允許重繪的區域
    preblanked_img[master_mask == 255] = noisy_flat_field[master_mask == 255]
    
    # ⚠️ 終極硬箝制：強制保護區像素 100% 絕對還原
    preblanked_img[dilated_protected == 255] = original_img[dilated_protected == 255]

    # 5. 內部 QA 指標：零污染檢測 (Verifiable QA Metric)
    diff = cv2.absdiff(preblanked_img, original_img)
    protected_diff = cv2.bitwise_and(diff, diff, mask=dilated_protected)
    contamination_score = np.sum(protected_diff)
    
    if contamination_score > 0:
        logger.error(f"🚨 [QA 致命錯誤] 檢測到保護區像素滲透！污染指數: {contamination_score}")
        raise RuntimeError("Pre-blanking 違反了絕對遮罩契約，保護區遭到污染。")
    else:
        logger.info("✅ [QA 驗證通過] Pre-blanking 成功：保護區像素實現絕對零污染。")

    # 6. 編碼輸出
    _, blanked_buf = cv2.imencode(".png", preblanked_img)
    _, mask_buf = cv2.imencode(".png", master_mask)

    return blanked_buf.tobytes(), mask_buf.tobytes()


def crop_to_original(image_bytes: bytes, metadata: Dict) -> bytes:
    """還原裁切與黑邊"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        pad = metadata["padding"]
        target_w, target_h = metadata["target_size"]
        
        if image.size != (target_w, target_h):
            return image_bytes
            
        crop_box = (pad["left"], pad["top"], target_w - pad["right"], target_h - pad["bottom"])
        cropped_image = image.crop(crop_box)
        
        orig_w, orig_h = metadata["original_size"]
        if cropped_image.size != (orig_w, orig_h):
            cropped_image = cropped_image.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
            
        output = io.BytesIO()
        cropped_image.save(output, format="PNG")
        return output.getvalue()
    except Exception as e:
        logger.error(f"還原裁切失敗: {str(e)}")
        return image_bytes