import asyncio
import time
from contextlib import asynccontextmanager
from core.logging import logger

class QueueTimeoutError(Exception):
    pass

class MaxRetriesExceededError(Exception):
    pass

class GlobalLimiter:
    _semaphore = asyncio.Semaphore(1)
    
    # 💡 實作四大 Metrics 統計
    metrics = {
        "total_requests": 0,
        "queue_wait_times": [],
        "slot_hold_times": [],
        "retry_counts": 0,
        "degradation_rate_events": 0
    }

    @classmethod
    @asynccontextmanager
    async def acquire(cls, queue_wait_timeout: float = 30.0):
        enter_time = time.monotonic()
        cls.metrics["total_requests"] += 1
        
        try:
            # 記錄 Queue Wait Time
            await asyncio.wait_for(cls._semaphore.acquire(), timeout=queue_wait_timeout)
            wait_duration = time.monotonic() - enter_time
            cls.metrics["queue_wait_times"].append(wait_duration)
        except asyncio.TimeoutError:
            cls.metrics["degradation_rate_events"] += 1
            raise QueueTimeoutError(f"排隊超時 (> {queue_wait_timeout}s)")

        slot_start_time = time.monotonic()
        try:
            yield "slot_acquired"
        finally:
            cls._semaphore.release()
            # 記錄 Slot Hold Time
            hold_duration = time.monotonic() - slot_start_time
            cls.metrics["slot_hold_times"].append(hold_duration)

    @classmethod
    def record_retry(cls):
        """記錄發生了重試"""
        cls.metrics["retry_counts"] += 1

    @classmethod
    def record_degradation(cls):
        """記錄發生了系統降級"""
        cls.metrics["degradation_rate_events"] += 1
        
    @classmethod
    def print_metrics_report(cls):
        """列印當前 Metrics 報告"""
        total = cls.metrics["total_requests"]
        degrades = cls.metrics["degradation_rate_events"]
        deg_rate = (degrades / total * 100) if total > 0 else 0
        logger.info(f"📊 [Limiter Metrics] 請求:{total} | 降級:{degrades} ({deg_rate:.1f}%) | 重試:{cls.metrics['retry_counts']}")