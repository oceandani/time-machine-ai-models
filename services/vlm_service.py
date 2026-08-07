import os
import json
import asyncio
import random
import time
from typing import Dict, Any
from core.logging import logger
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

from core.limiter import GlobalLimiter, QueueTimeoutError, MaxRetriesExceededError

load_dotenv()

USE_MOCK_VLM = False

async def evaluate_image(image_bytes: bytes, year: str, location: str) -> Dict[str, Any]:
    """
    評估圖片是否符合時代背景。
    具備動態模型獲取 (Future-Proofing) 與嚴格的併發容錯機制。
    """
    if USE_MOCK_VLM:
        return {"pass": False, "reason": "[測試] 現代物品", "bbox": [0,0,0.5,0.5], "error_level": "CONTENT_REJECT"}

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("🚨 致命錯誤: 找不到 GEMINI_API_KEY！")
        return {"pass": False, "reason": "系統配置缺失", "bbox": [], "error_level": "SYSTEM_FATAL"}

    client = genai.Client(api_key=api_key)
    
    # ==========================================
    # 🌟 終極動態模型獲取機制 (Future-Proofing)
    # ==========================================
    candidate_models = []
    try:
        available_models = client.models.list()
        for m in available_models:
            if "flash" in m.name.lower() and "generateContent" in m.supported_generation_methods:
                clean_name = m.name.replace("models/", "")
                candidate_models.append(clean_name)
    except Exception as e:
        logger.warning(f"⚠️ 無法動態獲取模型名單: {str(e)}")
        
    if not candidate_models:
        candidate_models = ["gemini-3.5-flash", "gemini-1.5-flash"]
    
    MAX_RETRIES = 3
    MAX_TOTAL_TIME = 30.0
    REQUEST_TIMEOUT = 15.0
    
    system_instruction = f"""
    你是一個嚴格的歷史影像鑑定專家。檢查圖片是否完全符合 {year} 年的 {location}。
    若有現代物品，判定不通過 (pass: false)，並在 bbox 提供位置 [x_min, y_min, x_max, y_max]。
    必須且只能回傳 JSON 格式。
    """
    prompt = "請仔細檢查圖片，並以 JSON 格式回傳評估結果。"

    start_time = time.monotonic()

    try:
        for attempt in range(MAX_RETRIES):
            current_model = candidate_models[attempt % len(candidate_models)]
            
            try:
                async with GlobalLimiter.acquire(queue_wait_timeout=30.0) as slot:
                    logger.info(f"🧠 [Attempt {attempt+1}/{MAX_RETRIES}] 呼叫 VLM ({slot}) - 嘗試模型: {current_model}...")
                    
                    api_coroutine = client.aio.models.generate_content(
                        model=current_model,
                        contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/png")],
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            response_mime_type="application/json",
                            temperature=0.2
                        )
                    )
                    response = await asyncio.wait_for(api_coroutine, timeout=REQUEST_TIMEOUT)
                    
                    raw_text = response.text.strip()
                    if raw_text.startswith("```json"): raw_text = raw_text[7:-3].strip()
                    elif raw_text.startswith("```"): raw_text = raw_text[3:-3].strip()
                        
                    result = json.loads(raw_text)
                    if "pass" not in result:
                        raise ValueError("模型回覆缺失 'pass' 欄位")
                        
                    result["error_level"] = "NONE" if result["pass"] else "CONTENT_REJECT"
                    return result

            except QueueTimeoutError as e:
                logger.warning(f"⚠️ {str(e)} -> 觸發降級")
                return {"pass": False, "reason": "排隊超時", "bbox": [], "error_level": "SYSTEM_WARNING_SKIP"}
            
            except APIError as e:
                status_code = getattr(e, 'code', None)
                error_msg = str(e).lower()
                
                if status_code == 403 or "403" in error_msg:
                    logger.error(f"🚨 致命錯誤: API 權限遭拒或欠費: {error_msg}")
                    return {"pass": False, "reason": "API 權限異常", "bbox": [], "error_level": "SYSTEM_FATAL"}
                
                is_retryable = (status_code in [429, 503, 404]) or any(kw in error_msg for kw in ["429", "503", "quota", "too many requests", "404", "not found"])
                if not is_retryable:
                    logger.warning(f"⚠️ 遇到非重試型 API 錯誤 (Code: {status_code}): {error_msg} -> 觸發降級")
                    return {"pass": False, "reason": f"無法處理的 API 錯誤: {error_msg}", "bbox": [], "error_level": "SYSTEM_WARNING_SKIP"}
                
                logger.warning(f"⚠️ VLM 異常 (Code: {status_code}) 準備切換模型或退避: {error_msg}")
                
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ VLM API 請求超時 (>{REQUEST_TIMEOUT}s)")
                
            except Exception as e:
                logger.warning(f"⚠️ VLM 評估發生異常 (解析/網路): {str(e)}")

            elapsed_time = time.monotonic() - start_time
            if attempt == MAX_RETRIES - 1 or elapsed_time > MAX_TOTAL_TIME:
                raise MaxRetriesExceededError(f"達到重試上限 (嘗試: {attempt+1}/{MAX_RETRIES}, 耗時: {elapsed_time:.1f}s/{MAX_TOTAL_TIME}s)")
            
            backoff_time = (2 ** attempt) + random.uniform(0, 1)
            logger.info(f"⏳ 準備重試，釋放 Slot 並休眠 {backoff_time:.2f} 秒...")
            await asyncio.sleep(backoff_time)
            
    except MaxRetriesExceededError as e:
        logger.warning(f"🛑 {str(e)} -> 觸發最終降級")
        return {"pass": False, "reason": "超過最大重試限制", "bbox": [], "error_level": "SYSTEM_WARNING_SKIP"}