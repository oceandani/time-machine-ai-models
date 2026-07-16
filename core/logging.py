import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    """
    設定企業級日誌系統，將 Log 輸出至 Terminal 及檔案
    """
    # 確保 logs 資料夾存在
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "app.log")

    # 設定 Log 格式: [時間] [層級] [模組] - 訊息
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. 設定輸出至 Terminal (Console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 2. 設定輸出至檔案 (File)，最大 5MB，保留 3 個備份
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # 設定 Root Logger
    root_logger = logging.getLogger()
    # 先清除已有的 handlers，避免重複 print
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return root_logger

# 建立一個全域嘅 logger 實例供其他檔案使用
logger = logging.getLogger("TimeMachine")
# 喺模組載入時自動初始化
setup_logging()