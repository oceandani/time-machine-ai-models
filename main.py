from fastapi import FastAPI
from api.routes_jobs import router as jobs_router

app = FastAPI(
    title="Time Machine API",
    description="全球通用時光機 App 後端核心",
    version="1.0.0"
)

# 將 routes_jobs 裡面嘅 API 掛載到 /api/v1/jobs 路徑下
app.include_router(jobs_router, prefix="/api/v1/jobs", tags=["Jobs"])

@app.get("/")
async def root():
    return {"message": "Time Machine API is running!"}