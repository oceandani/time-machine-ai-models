import os
import json
import asyncio
import random
from typing import Dict, Any
from core.logging import logger
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

# 🛑 匯入我們寫好的 GlobalLimiter 與 Exception
from core.limiter import GlobalLimiter, QueueTimeoutError

load_dotenv()

USE_MOCK_VLM = False

# ⚠️ 注意：改為 async def
async def evaluate_image(image_bytes: bytes, year: str, location: str) -> Dict[str, Any]:
    """
    評估圖片是否符合時代背景，整合 GlobalLimiter 與非同步重試機制。
    回傳格式: {"pass": bool, "reason": str, "bbox": list, "error_level": str}
    """
    if USE_MOCK_VLM:
        return {"pass": False, "reason": "[測試] 現代物品", "bbox": [0,0,0.5,0.5], "error_level": "CONTENT_REJECT"}

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("🚨 致命錯誤: 找不到 GEMINI_API_KEY！")
        return {"pass": False, "reason": "系統配置缺失", "bbox": [], "error_level": "SYSTEM_FATAL"}

    # 初始化 Client
    client = genai.Client(api_key=api_key)
    max_retries = 3
    
    system_instruction = f"""
    你是一個嚴格的歷史影像鑑定專家。檢查圖片是否完全符合 {year} 年的 {location}。
    若有現代物品，判定不通過 (pass: false)，並在 bbox 提供位置 [x_min, y_min, x_max, y_max]。
    必須且只能回傳 JSON 格式。
    """
    prompt = "請仔細檢查圖片，並以 JSON 格式回傳評估結果。"

    # 🔄 進入重試迴圈
    for attempt in range(max_retries):
        try:
            # 1. 獲取併發 Slot (設定 30 秒排隊超時)
            async with GlobalLimiter.acquire(queue_wait_timeout=30.0):
                logger.info(f"🧠 [Attempt {attempt+1}/{max_retries}] 正在呼叫 Gemini 視覺大腦評估圖片...")
                
                # 2. 呼叫 Async API (必須使用 client.aio)
                response = await client.aio.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/png")],
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                )
                
                raw_text = response.text.strip()
                if raw_text.startswith("```json"): raw_text = raw_text[7:-3].strip()
                elif raw_text.startswith("```"): raw_text = raw_text[3:-3].strip()
                    
                result = json.loads(raw_text)
                
                if "pass" not in result:
                    # 解析異常，拋出讓下方的 Exception 捕捉並重試
                    raise ValueError("模型回覆缺失 'pass' 欄位")
                    
                # 級別一：正常的業務邏輯判斷
                result["error_level"] = "NONE" if result["pass"] else "CONTENT_REJECT"
                logger.info(f"✅ VQA 評估完成。通過: {result['pass']} | 原因: {result.get('reason', '無')}")
                return result

        except QueueTimeoutError as e:
            # 排隊等太耐：直接觸發降級 (不重試)
            logger.warning(f"⚠️ {str(e)} -> 觸發降級")
            return {"pass": False, "reason": "排隊超時", "bbox": [], "error_level": "SYSTEM_WARNING_SKIP"}
            
        except APIError as e:
            error_msg = str(e).lower()
            # 級別三：帳號權限或計費問題 (不重試，直接 Fatal)
            if "403" in error_msg:
                logger.error(f"🚨 致命錯誤: API 權限遭拒或欠費: {str(e)}")
                return {"pass": False, "reason": "API 權限異常", "bbox": [], "error_level": "SYSTEM_FATAL"}
            
            # 級別二：降級處理 (Rate limit 429, 伺服器超載 503 等)
            logger.warning(f"⚠️ VLM 伺服器擁堵或限流: {str(e)}")
            if attempt == max_retries - 1:
                return {"pass": False, "reason": "API 限流或擁堵", "bbox": [], "error_level": "SYSTEM_WARNING_SKIP"}
                
        except Exception as e:
            # 其他諸如網路斷線、JSON 解析徹底失敗等
            logger.warning(f"⚠️ VLM 評估發生非致命異常: {str(e)}")
            if attempt == max_retries - 1:
                return {"pass": False, "reason": "網路或解析異常", "bbox": [], "error_level": "SYSTEM_WARNING_SKIP"}
        
        # 3. 指數退避 + Jitter (嚴格規範：必須在 async with 區塊外進行！)
        if attempt < max_retries - 1:
            backoff_time = (2 ** attempt) + random.uniform(0, 1) # 1s, 2s, 4s + Jitter
            logger.info(f"⏳ 準備重試 (Attempt {attempt+2})，釋放 Slot 並休眠 {backoff_time:.2f} 秒...")
            await asyncio.sleep(backoff_time)