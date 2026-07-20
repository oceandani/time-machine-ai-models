import os
import json
import tempfile
from datetime import datetime
from typing import Any, Dict, Union
from core.logging import logger

class JobJSONEncoder(json.JSONEncoder):
    """處理複雜資料型別轉型"""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "tolist"):  # 處理 numpy 或 torch tensor
            return obj.tolist()
        return super().default(obj)

class JobManager:
    """
    優化後的 JobManager，具備原子寫入、安全性防護與自動狀態管理。
    """
    def __init__(self, base_dir="logs/jobs"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_safe_path(self, job_id: str, sub_dir: str = None) -> str:
        """嚴格的路徑生成，防止目錄遍歷"""
        safe_job_id = os.path.basename(job_id)
        path = os.path.join(self.base_dir, safe_job_id)
        if sub_dir:
            path = os.path.join(path, str(sub_dir))
        return path

    def _atomic_write(self, target_path: str, content: Union[str, bytes]):
        """原子寫入確保檔案完整性"""
        dirname = os.path.dirname(target_path)
        os.makedirs(dirname, exist_ok=True)
        
        # 統一處理為 bytes
        if isinstance(content, str):
            content = content.encode("utf-8")
            
        with tempfile.NamedTemporaryFile(dir=dirname, delete=False, mode="wb") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        os.replace(tmp_path, target_path)

    def init_job(self, job_id: str):
        """初始化任務，包含必要的 Manifest 文件"""
        job_path = self._get_safe_path(job_id)
        os.makedirs(job_path, exist_ok=True)
        
        # 初始化預設狀態
        self.update_status(job_id, {"status": "initialized", "created_at": datetime.now().isoformat()})
        logger.info(f"[{job_id}] 🚀 任務已初始化目錄與 Manifest。")
        return job_path

    def update_status(self, job_id: str, status_dict: Dict):
        """更新任務狀態文件"""
        file_path = os.path.join(self._get_safe_path(job_id), "status.json")
        self._atomic_write(file_path, json.dumps(status_dict, cls=JobJSONEncoder, indent=2))

    def save_image(self, job_id: str, pass_num: int, filename: str, content: bytes) -> str:
        if not isinstance(pass_num, int) or pass_num < 0:
            raise ValueError("pass_num 必須為非負整數")
            
        pass_dir = self._get_safe_path(job_id, f"pass{pass_num}")
        file_path = os.path.join(pass_dir, os.path.basename(filename))
        self._atomic_write(file_path, content)
        logger.info(f"[{job_id}] 🖼️ 原子儲存圖片: {file_path}")
        return file_path

    def save_metadata(self, job_id: str, pass_num: int, filename: str, data: dict) -> str:
        if not isinstance(pass_num, int) or pass_num < 0:
            raise ValueError("pass_num 必須為非負整數")
            
        pass_dir = self._get_safe_path(job_id, f"pass{pass_num}")
        file_path = os.path.join(pass_dir, os.path.basename(filename))
        json_str = json.dumps(data, cls=JobJSONEncoder, ensure_ascii=False, indent=2)
        
        self._atomic_write(file_path, json_str)
        logger.info(f"[{job_id}] 📄 原子儲存 Metadata: {file_path}")
        return file_path