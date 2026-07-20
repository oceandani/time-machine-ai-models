import io
import math
from typing import Tuple, Dict, TypedDict
from PIL import Image, ImageOps
import numpy as np
from core.logging import logger

# --- 接口契約 (Pipeline Contract) ---
class ImageMetadata(TypedDict):
    original_size: Tuple[int, int]    # 原始使用者上傳尺寸
    scale_ratio: float                # 縮放比例
    pre_pad_size: Tuple[int, int]     # 縮放後、Padding 前的尺寸
    target_size: Tuple[int, int]      # 最終送入 ComfyUI/VLM 的 64 倍數尺寸
    padding: Dict[str, int]           # 補邊大小 (left, top, right, bottom)

from PIL import Image, ImageOps
import numpy as np
from core.logging import logger

def process_and_standardize_image(image_bytes: bytes, max_size: int = 1024) -> Tuple[bytes, Dict]:
    """
    將使用者上傳的圖片進行企業級預處理：
    1. 修復 EXIF 轉向問題
    2. 確保去背透明圖轉為純 RGB
    3. 等比例縮放至最大 max_size
    4. 進行 Padding，確保長寬皆為 64 的倍數
    
    回傳: (處理後的圖片 bytes, 包含裁切資訊的 metadata dict)
    """
    try:
        # 讀取圖片
        image = Image.open(io.BytesIO(image_bytes))
        
        # 1. 修正 EXIF 轉向 (防止手機相片打橫)
        image = ImageOps.exif_transpose(image)
        
        # 2. 強制轉換為 RGB (處理帶有 Alpha Channel 的 PNG)
        # 統一背景與 Padding 顏色為黑色 (0, 0, 0) 避免 VLM 邊緣誤判
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            background = Image.new('RGB', image.size, (0, 0, 0))
            background.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
            image = background
        else:
            image = image.convert("RGB")
            
        original_width, original_height = image.size
        logger.info(f"原始圖片尺寸: {original_width}x{original_height}")

        # 3. 限制最大邊長 (等比例縮放)
        scale_ratio = 1.0
        if original_width > max_size or original_height > max_size:
            scale_ratio = max_size / max(original_width, original_height)
            new_w, new_h = int(original_width * scale_ratio), int(original_height * scale_ratio)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            logger.info(f"已縮放至: {new_w}x{new_h}")
            
        current_w, current_h = image.size
        
        # 4. 計算 64 倍數的目標尺寸
        target_w = math.ceil(current_w / 64) * 64
        target_h = math.ceil(current_h / 64) * 64
        
        pad_w = target_w - current_w
        pad_h = target_h - current_h
        
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        
        # 5. 進行 Padding (將圖片置中，周圍補黑色)
        if pad_w > 0 or pad_h > 0:
            image = ImageOps.expand(image, padding, fill=(0, 0, 0))
            logger.info(f"Padding 完成，目標尺寸: {target_w}x{target_h} (64倍數)")
            
        # 轉回 Bytes 準備儲存或傳輸 (改為 PNG 避免 JPEG 壓縮失真)
        output = io.BytesIO()
        image.save(output, format="PNG")
        
        # 記錄 metadata (遵循 ImageMetadata 契約)
        metadata: ImageMetadata = {
            "original_size": (original_width, original_height),
            "scale_ratio": scale_ratio,
            "pre_pad_size": (current_w, current_h),
            "target_size": (target_w, target_h),
            "padding": {
                "left": pad_left,
                "top": pad_top,
                "right": pad_right,
                "bottom": pad_bottom
            }
        }
        
        return output.getvalue(), metadata
        
    except Exception as e:
        logger.error(f"圖片預處理失敗: {str(e)}")
        raise RuntimeError(f"圖片預處理失敗: {str(e)}")

def create_mask_from_bbox(width: int, height: int, bbox: list) -> bytes:
    """
    根據 VLM 辨識出的 Bounding Box 建立 Inpaint Mask
    注意：傳入的 width 與 height 必須是 metadata["target_size"] (即 64 的倍數)
    bbox 格式預期為 [x_min, y_min, x_max, y_max] (比例 0.0 ~ 1.0)
    """
    try:
        # 建立全黑背景的 Numpy 陣列 (0)
        mask_array = np.zeros((height, width), dtype=np.uint8)
        
        x_min = max(0, int(bbox[0] * width))
        y_min = max(0, int(bbox[1] * height))
        x_max = min(width, int(bbox[2] * width))
        y_max = min(height, int(bbox[3] * height))
        
        # 將錯誤區域塗白 (255)
        mask_array[y_min:y_max, x_min:x_max] = 255
        
        mask_image = Image.fromarray(mask_array, mode='L')
        output = io.BytesIO()
        mask_image.save(output, format="PNG")
        return output.getvalue()
    except Exception as e:
        logger.error(f"建立 Mask 失敗: {str(e)}")
        raise RuntimeError(f"建立 Mask 失敗: {str(e)}")

def crop_to_original(image_bytes: bytes, metadata: Dict) -> bytes:
    """
    將 ComfyUI 生成完的圖片，根據 Metadata 裁走黑邊，並還原成最原始的解析度。
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        pad = metadata["padding"]
        target_w, target_h = metadata["target_size"]
        
        # 驗證尺寸是否符合預期
        if image.size != (target_w, target_h):
            logger.warning(f"生成圖片尺寸 {image.size} 與預期 {target_w}x{target_h} 不符，跳過裁切")
            return image_bytes
            
        # 計算裁切框 (left, upper, right, lower)
        crop_box = (
            pad["left"],
            pad["top"],
            target_w - pad["right"],
            target_h - pad["bottom"]
        )
        
        cropped_image = image.crop(crop_box)
        
        # 還原至最原始尺寸
        orig_w, orig_h = metadata["original_size"]
        if cropped_image.size != (orig_w, orig_h):
            cropped_image = cropped_image.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
            
        output = io.BytesIO()
        # 最終出圖可以儲存為高品質 JPEG 或 PNG
        cropped_image.save(output, format="PNG")
        logger.info(f"成功裁除黑邊並還原圖片尺寸: {cropped_image.size}")
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"還原裁切失敗: {str(e)}")
        # 如果裁切失敗，至少把原圖回傳，不要導致整個任務失敗
        return image_bytes