from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
import os
import io
from PIL import Image
from api.routes_jobs import router as jobs_router
from core.logging import logger
from services import vlm_service

# 2. 插入 lifespan 函數

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 系統啟動中... 執行 VLM 全能力深度自檢...")
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.critical("❌ 嚴重警告: GEMINI_API_KEY 未設定！")
    else:
        client = genai.Client(api_key=api_key)
        # 獲取候選名單並更新至全域變數
        vlm_service.CANDIDATE_MODELS_QUEUE = await vlm_service.get_candidate_models(client)
        logger.info(f"📋 準備測試候選模型佇列: {vlm_service.CANDIDATE_MODELS_QUEUE}")
        
        # 💡 建立 1x1 白圖測試 Full Capability (Image Part + Schema)
        img = Image.new('RGB', (1, 1), color='white')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        test_image_bytes = img_byte_arr.getvalue()

        test_success = False
        # 淘汰賽機制
        while vlm_service.CANDIDATE_MODELS_QUEUE:
            test_model = vlm_service.CANDIDATE_MODELS_QUEUE[0]
            logger.info(f"🔍 正在測試模型全能力 (Image+Schema): {test_model}...")
            
            try:
                # 💡 升級：使用全新的 VLMResponse Pydantic 模型進行自檢
                response = await client.aio.models.generate_content(
                    model=test_model,
                    contents=[
                        "Generate a valid dummy edit plan with action 'preserve' to pass the schema test.", 
                        types.Part.from_bytes(data=test_image_bytes, mime_type="image/png")
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=vlm_service.VLMResponse,  # 🚀 正確引用新的 Pydantic 類別
                        temperature=0.0
                    )
                )
                logger.info(f"✅ VLM 自檢通過！鎖定安全模型: {test_model}")
                vlm_service.ACTIVE_MODEL = test_model
                test_success = True
                break  # 成功找到可用模型，跳出迴圈
            except Exception as e:
                logger.warning(f"❌ 模型 {test_model} 測試失敗或不支援該功能: {str(e)}")
                # 測試失敗，將其從佇列中移除，繼續測試下一個
                vlm_service.CANDIDATE_MODELS_QUEUE.pop(0)

        if not test_success:
            logger.critical("🚨 所有模型全能力測試失敗！生圖評估將進入降級模式。請檢查 API 配額或網絡狀況。")
            
    yield
    logger.info("🛑 系統正在關閉...")

# 初始化 FastAPI 應用程式
app = FastAPI(
    title="Time Machine AI Models API", 
    description="具備 VLM 視覺大腦與 ComfyUI Inpaint 能力的時光機系統",
    version="2.0.0",
    lifespan=lifespan
)

# 設定 CORS，容許前端跨網域呼叫
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境記得改為特定網域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛載 API 路由
app.include_router(jobs_router, prefix="/api/v1/jobs", tags=["Jobs"])

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 時光機 API 伺服器 (Phase 2 Agentic Pipeline) 啟動成功！")

@app.get("/health", tags=["System"])
def health_check():
    """系統健康檢查 Endpoint"""
    return {"status": "healthy", "version": "2.0.0"}