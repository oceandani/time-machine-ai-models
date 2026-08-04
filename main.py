from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes_jobs import router as jobs_router
from core.logging import logger

# 初始化 FastAPI 應用程式
app = FastAPI(
    title="Time Machine AI Models API", 
    description="具備 VLM 視覺大腦與 ComfyUI Inpaint 能力的時光機系統",
    version="2.0.0"
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