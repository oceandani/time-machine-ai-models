from pydantic import BaseModel
from typing import Optional

# 定義 API 成功提交任務後嘅標準回覆格式
class JobSubmitResponse(BaseModel):
    status: str
    message: str
    job_id: str
    used_prompt: Optional[str] = None

# (日後可以加更多，例如 JobStatusResponse 等等)