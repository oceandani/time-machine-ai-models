import os
import json
import tempfile
import re
from datetime import datetime
from typing import Any, Dict, Union

from core.logging import logger

class JobJSONEncoder(json.JSONEncoder):
    """處理 datetime 等無法直接 JSON serialize 的物件"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class JobManager:
    """
    優化後的 JobManager，具備原子寫入、安全性防護與自動狀態管理。
    """
    def __init__(self, base_dir="logs/jobs"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_safe_path(self, job_id: str, sub_dir: str = None) -> str:
        """確保路徑安全，只對 job_id 做 basename，保留固定的 sub_dir 結構"""
        safe_job_id = os.path.basename(job_id).strip()
        job_path = os.path.join(self.base_dir, safe_job_id)
        if sub_dir:
            return os.path.join(job_path, str(sub_dir))
        return job_path

    def _validate_filename(self, filename: str) -> str:
        """過濾並驗證檔名，防止惡意注入或空檔名"""
        if not filename or not isinstance(filename, str):
            raise ValueError("檔名不能為空")
        safe_name = os.path.basename(filename).strip()
        # 僅允許英數字母、底線、橫線、點號與空格
        if not safe_name or not re.match(r'^[\w\-. ]+$', safe_name):
            raise ValueError(f"不合法的檔名: {filename}")
        return safe_name

    def _atomic_write(self, target_path: str, content: Union[str, bytes]):
        """原子寫入確保檔案完整性，統一處理 bytes 防止模式混搭"""
        dirname = os.path.dirname(target_path)
        os.makedirs(dirname, exist_ok=True)
        
        # 如果傳入的是字串，強制轉為 bytes
        if isinstance(content, str):
            content = content.encode("utf-8")
            
        with tempfile.NamedTemporaryFile(dir=dirname, delete=False, mode="wb") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        os.replace(tmp_path, target_path)

    def _write_json(self, target_path: str, data: Any):
        """統一 JSON 寫入格式 (UTF-8, indent=2, 無 ASCII 限制)"""
        json_str = json.dumps(data, cls=JobJSONEncoder, ensure_ascii=False, indent=2)
        self._atomic_write(target_path, json_str)

    def init_job(self, job_id: str):
        """初始化任務，包含必要的 Manifest 與 Progress 文件"""
        job_path = self._get_safe_path(job_id)
        os.makedirs(job_path, exist_ok=True)
        
        now_str = datetime.now().isoformat()
        
        # 1. 寫入狀態
        self.update_status(job_id, {"status": "initialized", "created_at": now_str})
        
        # 2. 建立空 Manifest
        manifest_path = os.path.join(job_path, "manifest.json")
        self._write_json(manifest_path, {"job_id": job_id, "passes": [], "created_at": now_str})
        
        # 3. 建立進度記錄 Progress
        progress_path = os.path.join(job_path, "progress.json")
        self._write_json(progress_path, {"step": 0, "pct": 0, "msg": "Job initialized"})
        
        logger.info(f"[{job_id}] 🚀 任務已初始化目錄、Manifest、Status 與 Progress。")
        return job_path

    def update_status(self, job_id: str, status_dict: Dict):
        """更新任務狀態文件"""
        file_path = os.path.join(self._get_safe_path(job_id), "status.json")
        self._write_json(file_path, status_dict)

    def save_image(self, job_id: str, pass_num: int, filename: str, content: bytes) -> str:
        """儲存圖片"""
        if not isinstance(pass_num, int) or pass_num < 0:
            raise ValueError("pass_num 必須為非負整數")
            
        safe_filename = self._validate_filename(filename)
        pass_dir = self._get_safe_path(job_id, f"pass{pass_num}")
        file_path = os.path.join(pass_dir, safe_filename)
        
        self._atomic_write(file_path, content)
        logger.info(f"[{job_id}] 🖼️ 原子儲存圖片: {file_path}")
        return file_path

    def save_metadata(self, job_id: str, pass_num: int, filename: str, data: dict) -> str:
        """儲存 Metadata"""
        if not isinstance(pass_num, int) or pass_num < 0:
            raise ValueError("pass_num 必須為非負整數")
            
        safe_filename = self._validate_filename(filename)
        pass_dir = self._get_safe_path(job_id, f"pass{pass_num}")
        file_path = os.path.join(pass_dir, safe_filename)
        
        self._write_json(file_path, data)
        logger.info(f"[{job_id}] 📄 原子儲存 Metadata: {file_path}")
        return file_path