import asyncio
import os
from contextlib import asynccontextmanager
from core.logging import logger

# ==========================================
# 1. 錯誤類型定義 (Exceptions Contract)
# ==========================================
class QueueTimeoutError(Exception):
    """當排隊等待 Slot 超時時拋出 (對應降級映射)"""
    pass

class MaxRetriesExceededError(Exception):
    """當重試次數或總耗時達到上限時拋出 (防止無窮等待)"""
    pass

# ==========================================
# 2. 全局限流器介面與 Phase 1 實作 (GlobalLimiter)
# ==========================================
class GlobalLimiter:
    """
    VQA 併發控制限流器。
    Phase 1 採用 In-Memory 的 asyncio.Semaphore 實作。
    Phase 2 將透過依賴注入或條件判斷切換為 Redis 實作。
    """
    _semaphore: asyncio.Semaphore = None
    
    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        """懶加載初始化 Semaphore"""
        if cls._semaphore is None:
            # 初始值設定點：從環境變數讀取。
            # 預設為 1 (嚴格串行)，用以保護開發期的免費 API Key。
            max_slots = int(os.getenv("VQA_MAX_CONCURRENCY", "1"))
            cls._semaphore = asyncio.Semaphore(max_slots)
            logger.info(f"🚧 GlobalLimiter 初始化完成，當前最大併發 Slot 數: {max_slots}")
        return cls._semaphore

    @classmethod
    @asynccontextmanager
    async def acquire(cls, queue_wait_timeout: float = 30.0):
        """
        獲取併發 Slot 的 Context Manager (強制釋放語意)。
        
        Args:
            queue_wait_timeout: 最大排隊等待時間 (秒)。超過此時間未排到隊伍，將拋出異常。
            
        Raises:
            QueueTimeoutError: 排隊超時，觸發系統降級。
            
        Usage:
            async with GlobalLimiter.acquire(queue_wait_timeout=30.0):
                await call_vlm_api()
        """
        sem = cls._get_semaphore()
        
        # 1. 嘗試排隊獲取鎖 (Acquire)
        try:
            # 使用 asyncio.wait_for 包裝 acquire 動作，賦予超時防死鎖能力
            await asyncio.wait_for(sem.acquire(), timeout=queue_wait_timeout)
        except asyncio.TimeoutError:
            logger.warning(f"⏳ [GlobalLimiter] 排隊等候 VQA Slot 逾時 ({queue_wait_timeout}s)")
            raise QueueTimeoutError(f"排隊等候超過 {queue_wait_timeout} 秒，無法取得 Slot")

        # 2. 進入臨界區 (Critical Section)
        try:
            # 將控制權交還給 async with 區塊內的業務代碼
            yield "slot_acquired"
        
        # 3. 強制釋放語意 (Release Semantics)
        finally:
            # 確保無論 API 呼叫是成功、失敗、甚至是進程崩潰邊緣，
            # 只要離開 async with 區塊，Slot 就必定會被釋放歸還。
            sem.release()
            # logger.debug("[GlobalLimiter] Slot 已安全釋放")