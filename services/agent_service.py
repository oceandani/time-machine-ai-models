import io
from typing import List, Dict, Any

from PIL import Image
from fastapi import HTTPException

from core.logging import logger
# 💡 [藍圖 v6.0] 確保引入全新的遮罩與前置抹除函數
from core.image_processor import process_and_standardize_image, create_tier1_mask, apply_tier2_preblanking, crop_to_original
from core.job_manager import JobManager
from services.inference_service import run_comfyui_inference, run_comfyui_inpaint, run_comfyui_global_rebuild
from services.vlm_service import evaluate_image

class SemanticRouter:
    # 💡 [藍圖 v6.0] 歷史原型 Prompt 矩陣 (嚴禁硬編碼年份/地點，提供極致微觀細節)
    PROMPT_MATRIX = {
        "ARCHETYPE_COMMERCIAL_STREET": {
            "pos": "[YEAR] historic commercial streetscape of [LOCATION], highly accurate [YEAR]-appropriate architecture, dense low-rise to mid-rise buildings, authentic vintage shop signs overlapping the street, weathered plaster and concrete textures, ground-floor traditional storefronts, period-accurate road surface, masterpiece, highly detailed",
            "neg": "glass curtain wall, modern high-rise, skyscraper, LED screens, modern vehicles, hyper-dense mega-architecture, futuristic design, modern asphalt"
        },
        "ARCHETYPE_INDUSTRIAL_PORT": {
            "pos": "[YEAR] historic industrial port and dockyard of [LOCATION], period-accurate [YEAR] maritime infrastructure, vintage cranes, old warehouses, weathered brick and steel structures, natural coastline, masterpiece, highly detailed",
            "neg": "modern reclamation, concrete piers, modern seawall, luxury high-density apartments, glass towers, modern shipping containers"
        },
        "ARCHETYPE_RURAL_VILLAGE": {
            "pos": "[YEAR] historic rural village settlement of [LOCATION], authentic [YEAR] low-density period-appropriate houses, traditional roofing, natural unpaved paths, agricultural surroundings, masterpiece, highly detailed",
            "neg": "modern housing estates, hyper-dense blocks, megastructures, skyscrapers, modern paved roads, urban clutter"
        },
        "ARCHETYPE_RESIDENTIAL_DISTRICT": {
            "pos": "[YEAR] historic residential neighborhood of [LOCATION], authentic [YEAR] vernacular architecture, period-accurate tenement buildings, consistent floor-to-floor height, authentic film color, masterpiece, highly detailed",
            "neg": "modern luxury apartments, glass facades, skyscrapers, contemporary mega-structures, modern vehicles"
        },
        "DEFAULT": {
            "pos": "historic scenery of [LOCATION] in [YEAR], authentic [YEAR] architectural style, masterpiece, highly detailed",
            "neg": "modern buildings, modern vehicles, glass facade, LED screens, modern infrastructure"
        }
    }

    @classmethod
    def determine_route(cls, semantic_data: Dict[str, Any]) -> Dict[str, Any]:
        schema_version = semantic_data.get("schema_version", "v3")
        if schema_version != "v5":
            logger.warning(f"⚠️ [Router] 偵測到舊版 Schema快取 ({schema_version})。強制安全降級為 Tier 1。")
            return {
                "tier": 1,
                "target_bboxes": semantic_data.get("bboxes", []),
                "protected_bboxes": [],
                "strategy_name": "SAFE_FALLBACK_TIER_1",
                "prompts": cls.PROMPT_MATRIX["DEFAULT"]
            }

        regions = semantic_data.get("regions", [])
        protected_zones = semantic_data.get("protected_zones", [])
        protected_bboxes = [pz["bbox"] for pz in protected_zones]
        
        primary_action = "PRESERVE_ALL"
        if regions:
            primary_action = regions[0].get("action", "PRESERVE_ALL")

        # [藍圖 v6.0] 只要觸發歷史原型，預設傳遞全畫布 [0,0,1000,1000] 進入 Tier 2
        if primary_action != "PRESERVE_ALL" and primary_action in cls.PROMPT_MATRIX:
            logger.info(f"🧭 [Router] 觸發歷史原型: {primary_action}。鎖定為 Tier 2 全域穿越。")
            return {
                "tier": 2, 
                "target_bboxes": [[0, 0, 1000, 1000]], 
                "protected_bboxes": protected_bboxes, 
                "strategy_name": "GLOBAL_ARCHETYPE_REBUILD", 
                "prompts": cls.PROMPT_MATRIX[primary_action]
            }
        elif primary_action != "PRESERVE_ALL":
            return {
                "tier": 2,
                "target_bboxes": [[0, 0, 1000, 1000]],
                "protected_bboxes": protected_bboxes,
                "strategy_name": "GLOBAL_DEFAULT_REBUILD",
                "prompts": cls.PROMPT_MATRIX["DEFAULT"]
            }
        else:
            return {"tier": 0, "target_bboxes": [], "protected_bboxes": protected_bboxes, "strategy_name": "PRESERVE_ALL", "prompts": cls.PROMPT_MATRIX["DEFAULT"]}

class AgentService:
    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager

    async def run_time_machine_pipeline(self, job_id: str, year: str, location: str, image_bytes: bytes) -> bytes:
        logger.info(f"[{job_id}] 🚀 開始執行時光機 Agentic Pipeline...")
        
        processed_bytes, metadata = process_and_standardize_image(image_bytes)
        current_image_path = self.job_manager.save_image(job_id, 0, "processed_input.jpg", processed_bytes)
        
        logger.info(f"[{job_id}] 🎨 執行初始生圖 (Pass 0)...")
        current_image_bytes = run_comfyui_inference(year, location, current_image_path)
        
        MAX_CORRECTION_PASSES = 4
        previous_tier = None
        
        for pass_num in range(MAX_CORRECTION_PASSES):
            logger.info(f"[{job_id}] 🧠 進入 VQA 評估階段 (Pass {pass_num})...")
            eval_result = await evaluate_image(current_image_bytes, year, location)
            
            raw_data = eval_result.get("raw_semantic_data", {})
            plan_id = raw_data.get("plan_id", "UNKNOWN_PLAN")
            self.job_manager.save_metadata(job_id, pass_num, "qa_result.json", eval_result)
            
            error_level = eval_result.get("error_level", "SYSTEM_WARNING_SKIP")
            
            if error_level == "NONE":
                logger.info(f"[{job_id}] ✅ 內容評估 [通過]: 圖片符合歷史事實！ (Plan ID: {plan_id})")
                break
            elif error_level == "SYSTEM_WARNING_SKIP":
                break
            elif error_level == "SYSTEM_FATAL":
                raise HTTPException(status_code=500, detail="系統視覺大腦配置錯誤")
            elif error_level == "CONTENT_REJECT":
                if pass_num == MAX_CORRECTION_PASSES - 1:
                    logger.warning(f"[{job_id}] ⚠️ 達到最大修正次數，強制回傳最終相片。")
                    break
                    
                if raw_data:
                    routing_decision = SemanticRouter.determine_route(raw_data)
                else:
                    routing_decision = {"tier": 1, "target_bboxes": eval_result.get("bboxes", []), "protected_bboxes": [], "strategy_name": "LEGACY_FALLBACK", "prompts": SemanticRouter.PROMPT_MATRIX["DEFAULT"]}
                
                tier = routing_decision["tier"]
                target_bboxes = routing_decision["target_bboxes"]
                protected_bboxes = routing_decision["protected_bboxes"]
                prompts = routing_decision["prompts"]
                
                if tier == 0:
                    break

                # 💡 [藍圖 v6.0] 狀態機防禦：嚴格保留斷路器機制
                if previous_tier == 2 and tier == 2:
                    logger.warning(f"[{job_id}] ⚠️ 偵測到連續 Tier 2 評估失敗！強制降級為 Tier 1 (保留幾何特徵進行局部修補)。")
                    tier = 1
                previous_tier = tier

                # 💡 [藍圖 v6.0] 無損變數穿透
                pos_prompt = prompts["pos"].replace("[LOCATION]", location).replace("[YEAR]", year)
                neg_prompt = prompts["neg"].replace("[LOCATION]", location).replace("[YEAR]", year)

                # 💡 [藍圖 v6.0] 物理層分流處理 (Physical Routing)
                if tier == 2:
                    # 啟動像素級前置抹除
                    blanked_bytes, mask_bytes = apply_tier2_preblanking(current_image_bytes, protected_bboxes)
                    temp_base_path = self.job_manager.save_image(job_id, pass_num + 1, f"temp_base_blanked_{pass_num}.jpg", blanked_bytes)
                    mask_path = self.job_manager.save_image(job_id, pass_num + 1, "tier2_master_mask.png", mask_bytes)
                    
                    current_denoise = 0.85
                    logger.info(f"[{job_id}] 🌍 [Tier 2] 啟動前置抹除全域重構 | Plan: {plan_id} | Denoise: {current_denoise:.2f} | 永恆禁區: {len(protected_bboxes)}")
                    current_image_bytes = run_comfyui_global_rebuild(year, location, temp_base_path, mask_path, current_denoise, pos_prompt, neg_prompt)
                
                else:
                    # 降級 Tier 1：使用傳統局部遮罩，並保留原始圖片幾何
                    with Image.open(io.BytesIO(current_image_bytes)) as img:
                        img_width, img_height = img.size
                    safe_target_bboxes = target_bboxes if target_bboxes else [[0, 0, 1000, 1000]]
                    
                    mask_bytes = create_tier1_mask(img_width, img_height, safe_target_bboxes, protected_bboxes)
                    temp_base_path = self.job_manager.save_image(job_id, pass_num + 1, f"temp_base_original_{pass_num}.jpg", current_image_bytes)
                    mask_path = self.job_manager.save_image(job_id, pass_num + 1, "tier1_local_mask.png", mask_bytes)
                    
                    current_denoise = min(0.75 + (pass_num * 0.05), 0.85)
                    logger.info(f"[{job_id}] 🖌️ [Tier 1] 啟動局部修補防禦 | Plan: {plan_id} | Denoise: {current_denoise:.2f}")
                    current_image_bytes = run_comfyui_inpaint(year, location, temp_base_path, mask_path, current_denoise, pos_prompt, neg_prompt)
        
        logger.info(f"[{job_id}] 成功裁除黑邊並還原圖片尺寸: {metadata['original_size']}")
        return crop_to_original(current_image_bytes, metadata)