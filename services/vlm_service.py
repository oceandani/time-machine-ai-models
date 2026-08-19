import os
import json
import asyncio
import random
import time
import re
from typing import Dict, Any, List
from core.logging import logger
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

from core.limiter import GlobalLimiter, QueueTimeoutError, MaxRetriesExceededError

load_dotenv()

USE_MOCK_VLM = False

# 全域狀態：儲存經過驗證的模型佇列與當前啟用的模型
ACTIVE_MODEL = "gemini-flash" 
CANDIDATE_MODELS_QUEUE: List[str] = []

# 生產級 JSON Schema 定義
VLM_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "pass": {"type": "BOOLEAN"},
        "reason": {"type": "STRING"},
        "anachronisms": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "bbox": {
                        "type": "ARRAY",
                        "items": {"type": "INTEGER"}
                    }
                }
            }
        }
    },
    "required": ["pass", "reason", "anachronisms"]
}

async def get_candidate_models(client: genai.Client) -> List[str]:
    """
    獲取模型候選佇列，保證不包含不支援的變體，並將別名墊底。
    """
    env_model = os.getenv("VLM_MODEL", "auto")
    candidates = []
    
    if env_model.lower() != "auto":
        candidates.append(env_model)
    else:
        try:
            models = client.models.list()
            for m in models:
                name_lower = m.name.lower()
                if "gemini" in name_lower and "flash" in name_lower:
                    # 嚴格黑名單過濾
                    if not any(bad in name_lower for bad in ["image", "vision", "experimental", "embedding", "tts", "thinking"]):
                        candidates.append(m.name.replace("models/", ""))
            if candidates:
                candidates.sort(reverse=True)
        except Exception as e:
            logger.error(f"⚠️ 動態模型清單獲取失敗: {str(e)}")

    # 確保穩定別名必定存在於佇列最後
    if "gemini-flash" not in candidates:
        candidates.append("gemini-flash")
        
    # 移除重複項並保持順序
    seen = set()
    return [x for x in candidates if not (x in seen or seen.add(x))]

async def evaluate_image(image_bytes: bytes, year: str, location: str) -> Dict[str, Any]:
    global ACTIVE_MODEL, CANDIDATE_MODELS_QUEUE
    
    if USE_MOCK_VLM:
        return {"pass": False, "reason": "[測試] 現代物品", "bboxes": [[0,0,500,500]], "error_level": "CONTENT_REJECT"}

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("🚨 致命錯誤: 找不到 GEMINI_API_KEY！")
        return {"pass": False, "reason": "系統配置缺失", "bboxes": [], "error_level": "SYSTEM_FATAL"}

    client = genai.Client(api_key=api_key)

    MAX_NETWORK_RETRIES = 3   
    MAX_LOGIC_RETRIES = 1   
    network_attempts = 0
    logic_attempts = 0
    
    MAX_TOTAL_TIME = 60.0     
    REQUEST_TIMEOUT = 45.0    
    
    system_instruction = f"""
    你是一個嚴格的歷史影像鑑定專家。請檢查圖片是否完全符合 {year} 年的 {location}。
    如果發現任何現代物品，必須判定不通過 (pass: false)，並將所有錯誤區域記錄在 `anachronisms` 陣列中。
    BBOX 格式：必須使用 `[y_min, x_min, y_max, x_max]`，數值為 0 到 1000 的整數。
    天空涵蓋指令：若框選現代高樓大廈，請將 `y_min` 向上延伸包含天空，但避開前景電線或路燈。
    """
    prompt = "請仔細檢查圖片，並以 JSON 格式回傳評估結果。"

    start_time = time.monotonic()

    # 💡 獨立狀態機：清晰區分 Network 與 Logic 重試
    while network_attempts < MAX_NETWORK_RETRIES and logic_attempts <= MAX_LOGIC_RETRIES:
        elapsed_time = time.monotonic() - start_time
        if elapsed_time > MAX_TOTAL_TIME:
            logger.warning("🛑 總耗時超過限制，強制中斷")
            break

        current_model = ACTIVE_MODEL
        
        try:
            async with GlobalLimiter.acquire(queue_wait_timeout=30.0) as slot:
                logger.info(f"🧠 [Net:{network_attempts}|Logic:{logic_attempts}] 呼叫 VLM ({current_model})...")
                
                api_coroutine = client.aio.models.generate_content(
                    model=current_model,
                    contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/png")],
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=VLM_RESPONSE_SCHEMA,
                        temperature=0.2
                    )
                )
                response = await asyncio.wait_for(api_coroutine, timeout=REQUEST_TIMEOUT)
                
                # 💡 暴力正則提取與防禦校驗
                raw_text = response.text.strip()
                match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if not match:
                    raise ValueError(f"無法從回應中提取 JSON 結構: {raw_text[:50]}...")
                    
                result = json.loads(match.group(0))
                
                if not isinstance(result, dict) or "pass" not in result or "anachronisms" not in result:
                    raise ValueError("JSON 解析成功但缺少核心結構 (pass 或 anachronisms)")
                
                bboxes = []
                for item in result.get("anachronisms", []):
                    box = item.get("bbox")
                    if isinstance(box, list) and len(box) == 4:
                        bboxes.append(box)
                
                result["bboxes"] = bboxes
                result["error_level"] = "NONE" if result.get("pass") else "CONTENT_REJECT"
                return result

        except QueueTimeoutError as e:
            GlobalLimiter.record_degradation()
            logger.warning(f"⚠️ {str(e)} -> 觸發降級")
            return {"pass": False, "reason": "排隊超時", "bboxes": [], "error_level": "SYSTEM_WARNING_SKIP"}
        
        except APIError as e:
            status_code = getattr(e, 'code', None)
            error_msg = str(e).lower()
            
            if status_code == 403 or "403" in error_msg:
                logger.error(f"🚨 致命模型錯誤 (403): 權限遭拒")
                GlobalLimiter.record_degradation()
                return {"pass": False, "reason": "API 權限異常", "bboxes": [], "error_level": "SYSTEM_FATAL"}
            
            if status_code in [404, 400] or "404" in error_msg:
                logger.warning(f"⚠️ 模型 {current_model} 配置錯誤或失效 (Code: {status_code})。準備輪換...")
                # 💡 運行時輪換防禦
                if current_model in CANDIDATE_MODELS_QUEUE:
                    CANDIDATE_MODELS_QUEUE.remove(current_model)
                if CANDIDATE_MODELS_QUEUE:
                    ACTIVE_MODEL = CANDIDATE_MODELS_QUEUE[0]
                    logger.info(f"🔄 成功切換至備用模型: {ACTIVE_MODEL}，即時重試")
                    continue # 內部直接重試，不消耗 Network Attempts
                else:
                    logger.error("🚨 候補模型耗盡，全數失效，觸發降級")
                    GlobalLimiter.record_degradation()
                    return {"pass": False, "reason": "模型全數失效", "bboxes": [], "error_level": "SYSTEM_WARNING_SKIP"}
            
            # 處理 429, 503 等網絡異常
            logger.warning(f"⚠️ VLM 限流/伺服器繁忙 (Code: {status_code})")
            network_attempts += 1
            GlobalLimiter.record_retry()
            
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ VLM API 請求超時 (>{REQUEST_TIMEOUT}s)")
            network_attempts += 1
            GlobalLimiter.record_retry()
            
        except Exception as e:
            logger.warning(f"⚠️ VLM 邏輯/解析異常: {str(e)}")
            logic_attempts += 1
            GlobalLimiter.record_retry()

        # Exponential Backoff 計算
        if network_attempts < MAX_NETWORK_RETRIES and logic_attempts <= MAX_LOGIC_RETRIES:
            backoff_time = (2 ** max(network_attempts, logic_attempts)) + random.uniform(0, 1)
            logger.info(f"⏳ 休眠 {backoff_time:.2f} 秒後重試...")
            await asyncio.sleep(backoff_time)
            
    GlobalLimiter.record_degradation()
    logger.warning(f"🛑 達到重試上限 (Net:{network_attempts}, Logic:{logic_attempts})，觸發最終降級")
    return {"pass": False, "reason": "超過最大重試限制", "bboxes": [], "error_level": "SYSTEM_WARNING_SKIP"}