import asyncio
import json
import os
import random
import time
from enum import Enum
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field, model_validator, ValidationError

from core.limiter import GlobalLimiter, QueueTimeoutError, MaxRetriesExceededError
from core.logging import logger

load_dotenv()

USE_MOCK_VLM = False
ACTIVE_MODEL = "gemini-flash" 
CANDIDATE_MODELS_QUEUE: List[str] = []

# -----------------------------------------------------------------
# 商業級核心約束：世界歷史學家分類學 (v5.0 Macro-Historian Taxonomy)
# -----------------------------------------------------------------

class ActionType(str, Enum):
    ARCHETYPE_COMMERCIAL_STREET = "ARCHETYPE_COMMERCIAL_STREET"
    ARCHETYPE_INDUSTRIAL_PORT = "ARCHETYPE_INDUSTRIAL_PORT"
    ARCHETYPE_RURAL_VILLAGE = "ARCHETYPE_RURAL_VILLAGE"
    ARCHETYPE_RESIDENTIAL_DISTRICT = "ARCHETYPE_RESIDENTIAL_DISTRICT"
    PRESERVE_ALL = "PRESERVE_ALL"

class EvidenceType(str, Enum):
    ANACHRONISM_DETECTED = "ANACHRONISM_DETECTED"
    HISTORICAL_ACCURACY_VERIFIED = "HISTORICAL_ACCURACY_VERIFIED"

class ProtectedType(str, Enum):
    SKY_BACKGROUND = "SKY_BACKGROUND"
    WATER_BODY = "WATER_BODY"
    NATURAL_TERRAIN = "NATURAL_TERRAIN"
    PRESERVED_HERITAGE = "PRESERVED_HERITAGE"

class Region(BaseModel):
    action: ActionType = Field(description="Macro historical archetype decision for the entire image.")
    evidence_type: EvidenceType = Field(description="Evidence justifying the macro decision.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    # 全域穿越架構下，目標 bbox 預設涵蓋全圖
    bbox: List[int] = Field(default=[0, 0, 1000, 1000], min_length=4, max_length=4)

    @model_validator(mode='after')
    def validate_region_logic(self):
        for val in self.bbox:
            if not (0 <= val <= 1000):
                raise ValueError(f"BBox coordinate {val} out of bounds (0-1000).")
        if self.action == ActionType.PRESERVE_ALL and self.evidence_type != EvidenceType.HISTORICAL_ACCURACY_VERIFIED:
            raise ValueError("Logical Error: PRESERVE_ALL must pair with HISTORICAL_ACCURACY_VERIFIED.")
        return self

class ProtectedZone(BaseModel):
    category: ProtectedType = Field(description="The semantic category of the protected zone.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    bbox: List[int] = Field(min_length=4, max_length=4, description="Format: [y_min, x_min, y_max, x_max]. Normalized 0-1000.")

    @model_validator(mode='after')
    def validate_protected_bounds(self):
        for val in self.bbox:
            if not (0 <= val <= 1000):
                raise ValueError(f"Protected BBox coordinate {val} out of bounds (0-1000).")
        return self

class VLMResponse(BaseModel):
    # 💡 這裡定義了版本的單一真理來源
    schema_version: str = Field(default="v5", description="Schema version identifier.")
    plan_id: str = Field(description="Generate a unique plan ID (e.g., TIME-PLAN-1234).")
    edit_summary: str = Field(description="Geographic and historical assessment summary.")
    regions: List[Region] = Field(max_length=1, description="One single macro decision for the image.")
    protected_zones: Optional[List[ProtectedZone]] = Field(default_factory=list, max_length=6)

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
        return {"pass": False, "reason": "[測試] 現代物品", "bboxes": [[0,0,1000,1000]], "error_level": "CONTENT_REJECT"}

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("🚨 致命錯誤: 找不到 GEMINI_API_KEY！")
        return {"pass": False, "reason": "系統配置缺失", "bboxes": [], "error_level": "SYSTEM_FATAL"}

    client = genai.Client(api_key=api_key)

    MAX_NETWORK_RETRIES = 3   
    MAX_LOGIC_RETRIES = 2   
    network_attempts = 0
    logic_attempts = 0
    MAX_TOTAL_TIME = 60.0     
    REQUEST_TIMEOUT = 45.0    
    
    # 💡 動態讀取當前 Pydantic 模型的預設版本號，徹底消除硬編碼風險
    current_schema_version = VLMResponse.model_fields["schema_version"].default
    
    system_instruction: str = f"""
    你是一個擁有全知視角的「世界歷史學家與時空物理學家」。
    任務：鑑定這張照片是否 100% 完美符合 {year} 年 {location} 的真實地理風貌與建築水平。
    
    【核心時空法則 (Macro-Temporal Rules)】：
    1. 零容忍違和感：只要畫面中存在任何一處超越 {year} 年科技水平的特徵（現代摩天大樓、柏油路面、現代車輛、LED招牌），或者地貌完全改變，你必須立刻拒絕 PRESERVE_ALL，並選取最合適的 ARCHETYPE 指令。
    2. 拒絕材質欺騙：殘舊的石屎外牆不代表歷史悠久。一棟超乎當年法規與技術的高層大廈，即使再殘舊，也是時代錯誤的違規量體。
    
    【防禦型禁區劃分 (Defensive Masking - 極度重要)】：
    你的首要任務是找出時光機「絕對不能觸碰的永恆元素」並將其標記為 protected_zones：
    - SKY_BACKGROUND (天空): 只能框選完全無遮擋的純淨天空。若被違規高樓截斷，請放棄框選該交錯部分。
    - WATER_BODY (水體): 海洋、河流、湖泊。
    - NATURAL_TERRAIN (自然地形): 未被破壞的山脈。
    寧可少標記禁區，也絕不能將違規的現代建築標記為保護區。
    
    【絕對系統約束】：
    - 輸出版本約束：你的 schema_version 必須且只能是 "{current_schema_version}"。絕對禁止使用舊版的動作指令。
    """
    
    prompt = f"Formulate a precise historical macro-reconstruction plan for {year}, {location}. Determine the Archetype and strictly identify protected natural zones."
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
                
                # 💡 [藍圖 v5.0] 強制本地 Pydantic 驗證，攔截 VLM 幻覺
                try:
                    validated_response = VLMResponse.model_validate_json(raw_text)
                    parsed_json = validated_response.model_dump()
                except ValidationError as validation_error:
                    logger.warning(f"⚠️ VLM 輸出格式違規 (觸發幻覺防禦): 拒絕非 {current_schema_version} 的結構。")
                    raise ValueError(f"Schema Validation Failed: {validation_error}")
                
                bboxes = []
                has_critical_error = False
                
                for region in parsed_json.get("regions", []):
                    action = region.get("action")
                    if action != ActionType.PRESERVE_ALL.value:
                        has_critical_error = True
                        if "bbox" in region:
                            bboxes.append(region["bbox"])
                
                plan_id = parsed_json.get("plan_id", "UNKNOWN_PLAN")
                protected_count = len(parsed_json.get("protected_zones", []))
                logger.info(f"📝 [VLM Plan {plan_id}] Schema: {parsed_json['schema_version']} | Protected Zones: {protected_count} | Summary: {parsed_json.get('edit_summary')}")
                
                return {
                    "pass": not has_critical_error,
                    "reason": parsed_json.get("edit_summary", "需要時空重構"),
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
            if status_code == 403:
                return {"pass": False, "reason": "API 權限異常", "bboxes": [], "error_level": "SYSTEM_FATAL"}
            network_attempts += 1
            GlobalLimiter.record_retry()
            
        except asyncio.TimeoutError:
            network_attempts += 1
            GlobalLimiter.record_retry()
            
        except Exception as e:
            # 💡 捕捉到 Pydantic ValidationError 拋出的 ValueError，增加邏輯重試次數
            logger.warning(f"⚠️ VLM 邏輯/解析異常: {str(e)}")
            logic_attempts += 1
            GlobalLimiter.record_retry()

        if network_attempts < MAX_NETWORK_RETRIES and logic_attempts <= MAX_LOGIC_RETRIES:
            await asyncio.sleep((2 ** max(network_attempts, logic_attempts)))
            
    return {"pass": False, "reason": "超過最大重試限制", "bboxes": [], "error_level": "SYSTEM_WARNING_SKIP"}