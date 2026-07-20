import uuid
import traceback
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Response

# 引入核心生圖服務、任務管理與企業級日誌
from core.logging import logger
from core.job_manager import JobManager
from services.agent_service import AgentService

router = APIRouter()

@router.post("/generate", summary="提交時光機生圖任務")
async def create_generation_job(
    year: str = Form(..., description="目標年份，例如: 1960s"),
    location: str = Form(..., description="地點，例如: Causeway Bay, Hong Kong"),
    image: UploadFile = File(...)
):
    """
    接收前端圖片與參數，交給 AgentService 進行預處理、生成與 VQA 評估
    """
    # 產生較短且易讀的 job_id
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    
    try:
        image_bytes = await image.read()
        logger.info(f"[{job_id}] 收到 API 請求，準備處理 {location} 於 {year} 嘅圖片...")
        
        # 初始化任務管理與大腦
        job_manager = JobManager()
        agent_service = AgentService(job_manager)
        
        # 將資料直接交給 Agentic Pipeline 處理，避免暫存檔殘留
        final_image_bytes = await agent_service.run_time_machine_pipeline(
            job_id=job_id,
            image_bytes=image_bytes,
            year=year,
            location=location
        )
        
        logger.info(f"[{job_id}] API 成功完成流程，準備回傳給前端。")
        
        # 回傳 PNG 圖片 (因 image_processor 已統一防損轉為 PNG)
        return Response(content=final_image_bytes, media_type="image/png")
        
    except Exception as e:
        logger.error(f"[{job_id}] ❌ API 發生嚴重錯誤:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))