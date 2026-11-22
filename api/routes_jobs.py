import uuid
import os
import traceback
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response
from dotenv import load_dotenv
from services.agent_service import AgentService
from core.job_manager import JobManager
from core.logging import logger

# 確保在啟動路由時載入環境變數
load_dotenv()

router = APIRouter()
job_manager = JobManager()

@router.post("/process", summary="提交時光機生圖任務")
async def process_image(
    image: UploadFile = File(...),
    year: str = Form(..., description="目標年份，例如: 1960s"),
    location: str = Form(..., description="地點，例如: Causeway Bay, Hong Kong")
):
    """
    接收圖片與參數，交給 AgentService 進行預處理、生成與 VQA 評估
    """
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    logger.info(f"[{job_id}] 📥 收到新請求: {image.filename} (目標: {location}, {year})")
    
    # 🛑 新增：起飛前檢查 (Pre-flight Check)
    # 根據你的建議，在浪費 GPU 資源生圖之前，先確保視覺大腦的 API Key 存在
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error(f"[{job_id}] 🚨 起飛前檢查失敗: 系統尚未配置 GEMINI_API_KEY，拒絕執行任務。")
        raise HTTPException(status_code=500, detail="系統配置缺失: 找不到 Gemini API Key，請檢查 .env 檔案。")
    
    try:
        # 直接將圖片讀取到記憶體
        image_bytes = await image.read()
        
        # 初始化 Agent 大腦 (注入 job_manager)
        agent = AgentService(job_manager)
        
        # 將任務交畀 Agent 處理 (傳入 year 同 location)
        final_image_bytes = await agent.run_time_machine_pipeline(
            job_id=job_id, 
            image_bytes=image_bytes, 
            year=year, 
            location=location
        )
        
        logger.info(f"[{job_id}] 📤 任務完成，回傳最終圖片")
        # 將最終無損 PNG bytes 回傳畀前端
        return Response(content=final_image_bytes, media_type="image/png")

    except HTTPException as http_exc:
        # P0 嚴格狀態管理：保留 AgentService 拋出之 422 驗收失敗報告或 503 狀態碼，直接回傳前端並記錄狀態
        job_manager.update_status(
            job_id,
            {
                "status": "qa_rejected" if http_exc.status_code == 422 else "failed",
                "status_code": http_exc.status_code,
                "detail": http_exc.detail,
            },
        )
        raise http_exc

    except Exception as e:
        logger.error(f"[{job_id}] ❌ 處理發生嚴重錯誤:\n{traceback.format_exc()}")
        # 更新任務狀態為失敗
        job_manager.update_status(job_id, {"status": "failed", "error": str(e)})
        raise HTTPException(status_code=500, detail=f"系統處理錯誤: {str(e)}")