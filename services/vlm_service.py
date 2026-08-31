import asyncio
import json
import os
import random
import time
from enum import Enum
from typing import Dict, Any, List

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field, model_validator

from core.limiter import GlobalLimiter, QueueTimeoutError, MaxRetriesExceededError
from core.logging import logger

load_dotenv()

USE_MOCK_VLM = False
ACTIVE_MODEL = "gemini-flash"
CANDIDATE_MODELS_QUEUE: List[str] = []

class ActionType(str, Enum):
    PRESERVE = "preserve"
    REPLACE_STYLE = "replace_style"
    REMOVE_LOCAL_OBJECT = "remove_local_object"
    REPLACE_BUILDING_LOWRISE = "replace_building_lowrise"
    REMOVE_BUILDING_OR_MASS = "remove_building_or_mass"
    REBUILD_LANDFORM_OR_FILL = "rebuild_landform_or_fill"
    REBUILD_LARGE_BUILDING_GROUP = "rebuild_large_building_group"

class EvidenceType(str, Enum):
    STRUCTURAL_SCALE_VIOLATION = "STRUCTURAL_SCALE_VIOLATION"
    MODERN_MATERIAL = "MODERN_MATERIAL"
    LANDFORM_MISMATCH = "LANDFORM_MISMATCH"
    MODERN_OBJECT_OR_VEHICLE = "MODERN_OBJECT_OR_VEHICLE"
    HISTORICAL_CONTEXT_VALID = "HISTORICAL_CONTEXT_VALID"

class Region(BaseModel):
    action: ActionType = Field(description="Action to take. MUST be from ActionType enum.")
    evidence_type: EvidenceType = Field(description="Evidence justifying action. MUST be from EvidenceType enum.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    bbox: List[int] = Field(min_length=4, max_length=4, description="Format: [y_min, x_min, y_max, x_max]. Normalized coordinates.")

    @model_validator(mode='after')
    def validate_region_logic(self):
        for val in self.bbox:
            if not (0 <= val <= 1000):
                raise ValueError(f"BBox coordinate {val} is out of bounds (0-1000).")
        
        if self.action in [ActionType.REMOVE_BUILDING_OR_MASS, ActionType.REBUILD_LARGE_BUILDING_GROUP]:
            if self.evidence_type == EvidenceType.MODERN_OBJECT_OR_VEHICLE:
                raise ValueError("Logical Error: Cannot remove a massive building based on a small modern object/vehicle.")
        
        if self.action == ActionType.PRESERVE and self.evidence_type != EvidenceType.HISTORICAL_CONTEXT_VALID:
            raise ValueError("Logical Error: PRESERVE action must be paired with HISTORICAL_CONTEXT_VALID evidence.")
            
        return self

class VLMResponse(BaseModel):
    plan_id: str = Field(description="Generate a unique plan ID (e.g., PLAN-1234) for auditing.")
    edit_summary: str = Field(description="Concise summary of the reconstruction strategy.")
    regions: List[Region] = Field(max_length=8, description="List of regions. Max 8 regions allowed.")

async def get_candidate_models(client: genai.Client) -> List[str]:
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
                    if not any(bad in name_lower for bad in ["image", "vision", "experimental", "embedding", "tts", "thinking"]):
                        candidates.append(m.name.replace("models/", ""))
            if candidates:
                candidates.sort(reverse=True)
        except Exception as e:
            logger.error(f"⚠️ 動態模型清單獲取失敗: {str(e)}")

    if "gemini-flash" not in candidates:
        candidates.append("gemini-flash")
        
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
    
    system_instruction: str = f"""
    你是一個頂級的歷史街景重構架構師 (Historical Streetscape Reconstruction Architect)。
    你的任務是為這張圖片制定一份符合 {year} 年 {location} 真實地貌的「影像重構計畫 (Edit Plan)」。
    
    【絕對工程輸出守則】：
    1. 必須嚴格輸出符合 Schema 的純 JSON 格式，嚴禁任何 Markdown 標記。
    2. bbox 格式為 [y_min, x_min, y_max, x_max]，數值必須在 0 到 1000 之間 (圖片左上角為 0,0)。
    3. 你最多只能定義 8 個關鍵修改區域 (Regions)。
    
    【重構策略指南】：
    - 不要只是「找錯」，你要決定「如何修復」。
    - 如果發現現代巨型大廈 (違反年代高度限制)，action 應為 remove_building_or_mass，evidence 為 STRUCTURAL_SCALE_VIOLATION。
    - 如果是現代填海區或空地上的現代設施，應使用 rebuild_landform_or_fill 或 rebuild_large_building_group。
    - 準確評估 confidence (0.0-1.0)，若不確定請降低數值。
    """
    
    prompt = f"Formulate a historical reconstruction edit plan for this image based on the context of {year}, {location}."
    start_time = time.monotonic()

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
                        response_schema=VLMResponse,  
                        temperature=0.0               
                    )
                )
                response = await asyncio.wait_for(api_coroutine, timeout=REQUEST_TIMEOUT)
                
                raw_text = response.text.strip()
                parsed_json = json.loads(raw_text)
                
                bboxes = []
                has_critical_error = False
                
                for region in parsed_json.get("regions", []):
                    action = region.get("action")
                    if action != "preserve":
                        has_critical_error = True
                        if "bbox" in region:
                            bboxes.append(region["bbox"])
                
                plan_id = parsed_json.get("plan_id", "UNKNOWN_PLAN")
                logger.info(f"📝 [VLM Plan {plan_id}] Summary: {parsed_json.get('edit_summary')}")
                
                return {
                    "pass": not has_critical_error,
                    "reason": parsed_json.get("edit_summary", "需要修改地貌"),
                    "bboxes": bboxes, 
                    "error_level": "CONTENT_REJECT" if has_critical_error else "NONE",
                    "raw_semantic_data": parsed_json 
                }

        except QueueTimeoutError as e:
            GlobalLimiter.record_degradation()
            logger.warning(f"⚠️ {str(e)} -> 觸發降級")
            return {"pass": False, "reason": "排隊超時", "bboxes": [], "error_level": "SYSTEM_WARNING_SKIP"}
        
        except APIError as e:
            status_code = getattr(e, 'code', None)
            error_msg = str(e).lower()
            if status_code == 403 or "403" in error_msg:
                logger.error("🚨 致命模型錯誤 (403): 權限遭拒")
                GlobalLimiter.record_degradation()
                return {"pass": False, "reason": "API 權限異常", "bboxes": [], "error_level": "SYSTEM_FATAL"}
            if status_code in [404, 400] or "404" in error_msg:
                logger.warning(f"⚠️ 模型 {current_model} 配置錯誤或失效 (Code: {status_code})。準備輪換...")
                if current_model in CANDIDATE_MODELS_QUEUE:
                    CANDIDATE_MODELS_QUEUE.remove(current_model)
                if CANDIDATE_MODELS_QUEUE:
                    ACTIVE_MODEL = CANDIDATE_MODELS_QUEUE[0]
                    logger.info(f"🔄 成功切換至備用模型: {ACTIVE_MODEL}，即時重試")
                    continue 
                else:
                    logger.error("🚨 候補模型耗盡，全數失效，觸發降級")
                    GlobalLimiter.record_degradation()
                    return {"pass": False, "reason": "模型全數失效", "bboxes": [], "error_level": "SYSTEM_WARNING_SKIP"}
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

        if network_attempts < MAX_NETWORK_RETRIES and logic_attempts <= MAX_LOGIC_RETRIES:
            backoff_time = (2 ** max(network_attempts, logic_attempts)) + random.uniform(0, 1)
            logger.info(f"⏳ 休眠 {backoff_time:.2f} 秒後重試...")
            await asyncio.sleep(backoff_time)
            
    GlobalLimiter.record_degradation()
    logger.warning(f"🛑 達到重試上限 (Net:{network_attempts}, Logic:{logic_attempts})，觸發最終降級")
    return {"pass": False, "reason": "超過最大重試限制", "bboxes": [], "error_level": "SYSTEM_WARNING_SKIP"}