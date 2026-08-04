import asyncio
from typing import Optional
from core.logging import logger
from core.job_manager import JobManager
from core.image_processor import process_and_standardize_image, create_mask_from_bbox, crop_to_original
from services.vlm_service import evaluate_image
from services.inference_service import run_comfyui_inference

class AgentService:
    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
        self.max_retries = 3

    async def run_time_machine_pipeline(self, job_id: str, image_bytes: bytes, year: str, location: str) -> bytes:
        self.job_manager.init_job(job_id)
        
        processed_bytes, metadata = process_and_standardize_image(image_bytes)
        saved_image_path = self.job_manager.save_image(job_id, 0, "input_processed.jpg", processed_bytes)
        
        logger.info(f"[{job_id}] 執行初始 Base 生成...")
        current_image = run_comfyui_inference(year, location, saved_image_path) 
        
        for pass_num in range(self.max_retries):
            eval_result = await evaluate_image(current_image, year, location)
            self.job_manager.save_metadata(job_id, pass_num, "qa_result.json", eval_result)
            
            # 1. 成功通過 VQA (完美結局)
            if eval_result["pass"]:
                logger.info(f"[{job_id}] ✅ 評估結果 [通過]: {eval_result.get('reason', '無提供原因')}")
                return crop_to_original(current_image, metadata)
                
            # 2. 處理失敗情況
            error_level = eval_result.get("error_level", "SYSTEM_WARNING_SKIP")
            
            if error_level == "SYSTEM_FATAL":
                # 致命錯誤：直接中斷任務
                logger.error(f"[{job_id}] 🚨 檢測到致命系統錯誤，強制終止任務: {eval_result['reason']}")
                raise RuntimeError(f"VLM 系統異常: {eval_result['reason']}")
                
            elif error_level == "SYSTEM_WARNING_SKIP":
                # 優雅降級：記錄警告，略過修正循環，直接給用戶當前最好的結果
                logger.warning(f"[{job_id}] ⚠️ VLM 暫時不可用 ({eval_result['reason']})，觸發優雅降級，直接回傳相片。")
                return crop_to_original(current_image, metadata)
                
            elif error_level == "CONTENT_REJECT":
                # 正常業務退回：執行 Inpaint 修正
                logger.warning(f"[{job_id}] ❌ 內容評估 [失敗]: {eval_result.get('reason', '無提供原因')}")
                logger.info(f"[{job_id}] 準備執行修正 Pass {pass_num+1}...")
                
                # 安全獲取 bbox，避免報錯
                bbox = eval_result.get("bbox", [0.4, 0.4, 0.6, 0.6])
                if not bbox or len(bbox) != 4:
                    bbox = [0.4, 0.4, 0.6, 0.6] # Fallback 中心區域
                    
                mask_bytes = create_mask_from_bbox(metadata["target_size"][0], metadata["target_size"][1], bbox)
                self.job_manager.save_image(job_id, pass_num+1, "error_mask.png", mask_bytes)
                
                # 此處呼叫 run_comfyui_inpaint(..., mask_bytes)
                # current_image = run_comfyui_inpaint(...) 
            
        # 達到最大重試次數後，回傳最後一張結果
        logger.info(f"[{job_id}] ⚠️ 達到最大修正次數，強制回傳最終相片。")
        return crop_to_original(current_image, metadata)