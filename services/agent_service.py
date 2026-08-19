import os
from fastapi import HTTPException
from core.logging import logger
from core.image_processor import process_and_standardize_image, create_mask_from_bboxes, crop_to_original
from core.job_manager import JobManager

from services.inference_service import run_comfyui_inference, run_comfyui_inpaint
from services.vlm_service import evaluate_image

class AgentService:
    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager

    async def run_time_machine_pipeline(self, job_id: str, year: str, location: str, image_bytes: bytes) -> bytes:
        logger.info(f"[{job_id}] 🚀 開始執行時光機 Agentic Pipeline...")
        
        # 1. 影像前處理
        processed_bytes, metadata = process_and_standardize_image(image_bytes)
        current_image_path = self.job_manager.save_image(job_id, 0, "processed_input.jpg", processed_bytes)
        
        # 2. 初始生圖 (Pass 0)
        logger.info(f"[{job_id}] 🎨 執行初始生圖 (Pass 0)...")
        current_image_bytes = run_comfyui_inference(year, location, current_image_path)
        
        MAX_CORRECTION_PASSES = 3
        
        # 3. VQA 自我修正迴圈
        for pass_num in range(MAX_CORRECTION_PASSES):
            logger.info(f"[{job_id}] 🧠 進入 VQA 評估階段 (Pass {pass_num})...")
            eval_result = await evaluate_image(current_image_bytes, year, location)
            
            # 💡 終極修正：補回 filename 參數 "qa_result.json"，完美對齊 JobManager 的要求
            self.job_manager.save_metadata(job_id, pass_num, "qa_result.json", eval_result)
            
            error_level = eval_result.get("error_level", "SYSTEM_WARNING_SKIP")
            
            if error_level == "NONE":
                logger.info(f"[{job_id}] ✅ 內容評估 [通過]: 圖片符合歷史事實！")
                break
                
            elif error_level == "SYSTEM_WARNING_SKIP":
                logger.warning(f"[{job_id}] ⚠️ VLM 暫時不可用，觸發優雅降級，直接回傳相片。")
                break
                
            elif error_level == "SYSTEM_FATAL":
                logger.error(f"[{job_id}] 🚨 處理發生嚴重錯誤: API 權限異常")
                raise HTTPException(status_code=500, detail="系統視覺大腦配置錯誤")
                
            elif error_level == "CONTENT_REJECT":
                logger.warning(f"[{job_id}] ❌ 內容評估 [失敗]: {eval_result.get('reason', '無提供原因')}")
                
                if pass_num == MAX_CORRECTION_PASSES - 1:
                    logger.info(f"[{job_id}] ⚠️ 達到最大修正次數，強制回傳最終相片。")
                    break
                    
                logger.info(f"[{job_id}] 準備執行修正 Pass {pass_num+1}...")
                
                # 💡 智能提取：優先獲取多區塊 bboxes 陣列
                bboxes = eval_result.get("bboxes", [])
                
                if not bboxes:
                    # 舊版向下兼容與 Fallback
                    single_bbox = eval_result.get("bbox")
                    if single_bbox and len(single_bbox) == 4:
                        bboxes = [single_bbox]
                    else:
                        logger.warning(f"[{job_id}] ⚠️ 找不到有效的 bbox，啟用 Fallback 預設置中區塊")
                        bboxes = [[0.4, 0.4, 0.6, 0.6]]
                    
                # 呼叫新的多區塊羽化引擎
                mask_bytes = create_mask_from_bboxes(metadata["target_size"][0], metadata["target_size"][1], bboxes)
                
                # 儲存 Mask 與當前 Base 圖片
                mask_path = self.job_manager.save_image(job_id, pass_num+1, "error_mask.png", mask_bytes)
                base_image_path = self.job_manager.save_image(job_id, pass_num, f"temp_base_pass_{pass_num}.jpg", current_image_bytes)
                
                # 呼叫 Inpaint 進行局部重繪
                current_image_bytes = run_comfyui_inpaint(year, location, base_image_path, mask_path)
        
        # 4. 後處理 (裁切還原)
        logger.info(f"[{job_id}] 成功裁除黑邊並還原圖片尺寸: {metadata['original_size']}")
        final_bytes = crop_to_original(current_image_bytes, metadata)
        
        logger.info(f"[{job_id}] 📤 任務完成，回傳最終圖片")
        return final_bytes