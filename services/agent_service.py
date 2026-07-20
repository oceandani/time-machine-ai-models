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
        """
        核心狀態機邏輯：
        1. 預處理 -> 2. Base 生成 -> 3. VLM 評估 -> 4. 根據結果決定結束或 Inpaint
        """
        self.job_manager.init_job(job_id)
        
        # 1. 影像標準化
        processed_bytes, metadata = process_and_standardize_image(image_bytes)
        self.job_manager.save_image(job_id, 0, "input_processed.jpg", processed_bytes)
        
        # 2. 初始生成 (Pass 0)
        logger.info(f"[{job_id}] 執行初始 Base 生成...")
        current_image = run_comfyui_inference(year, location, "logs/jobs/input_processed.jpg") # 需對應 job_manager 路徑
        
        # 3. Agent 迴圈
        for pass_num in range(self.max_retries):
            # VLM 評估
            eval_result = evaluate_image(current_image, year, location)
            self.job_manager.save_metadata(job_id, pass_num, "qa_result.json", eval_result)
            
            if eval_result["pass"]:
                logger.info(f"[{job_id}] 評估通過，任務完成。")
                return crop_to_original(current_image, metadata)
            
            # 若失敗，進行 Inpaint
            logger.warning(f"[{job_id}] 檢測到異常，準備執行修正 Pass {pass_num+1}...")
            
            # 生成 Mask
            mask_bytes = create_mask_from_bbox(metadata["target_size"][0], metadata["target_size"][1], eval_result["bbox"])
            self.job_manager.save_image(job_id, pass_num+1, "error_mask.png", mask_bytes)
            
            # 呼叫 ComfyUI Inpaint (此處需 Workflow 配合，此處省略具體 Inference 參數細節)
            # current_image = run_comfyui_inpaint(...) 
            
        return crop_to_original(current_image, metadata)