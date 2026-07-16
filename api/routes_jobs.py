import uuid
import os
import traceback
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Response

# 引入核心生圖服務與企業級日誌
from services.inference_service import run_comfyui_inference
from core.logging import logger

router = APIRouter()

@router.post("/generate", summary="提交時光機生圖任務")
async def create_generation_job(
    year: str = Form(..., description="目標年份，例如: 1960s"),
    location: str = Form(..., description="地點，例如: Causeway Bay, Hong Kong"),
    image: UploadFile = File(...)
):
    """
    接收前端圖片與參數，呼叫 Inference Service，並返回結果圖片
    """
    job_id = str(uuid.uuid4())
    input_image_path = f"temp_input_{job_id}_{image.filename}"
    
    try:
        # 1. 儲存前端傳來嘅原圖
        with open(input_image_path, "wb") as f:
            f.write(await image.read())
            
        logger.info(f"[{job_id}] 收到 API 請求，準備處理 {location} 於 {year} 嘅圖片...")
        
        # 2. 呼叫核心生圖服務
        image_bytes = run_comfyui_inference(year=year, location=location, input_image_path=input_image_path)
        
        logger.info(f"[{job_id}] API 成功從 ComfyUI 獲取圖片數據，準備回傳給前端。")
        
        # 3. 回傳圖片 (使用 Response 確保 Swagger UI 能直接預覽，而非強迫下載)
        return Response(content=image_bytes, media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"[{job_id}] ❌ API 發生嚴重錯誤:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 4. 確保清理暫存圖
        if os.path.exists(input_image_path):
            os.remove(input_image_path)
            logger.info(f"[{job_id}] 暫存檔案 {input_image_path} 已清理。")