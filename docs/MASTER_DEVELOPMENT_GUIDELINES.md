🚀 國際化 AI 街景時光機 App - 核心開發準則與藍圖總覽 (V1.0)
1. 商業目標與產品定位
核心功能：即時環境生成手機 App。用戶拍攝或上傳街景，系統即時輸出對應 1960s / 1970s / 1980s 風貌的寫實影像。

視覺要求：嚴格保留原始構圖與視角，將現代物件（車輛、招牌、LED 等）精準替換為歷史外觀。

輸出規格：單張 1080p 或 2K 寫實圖像，支援「寫實還原」與「風格化（膠片/黑白/色彩）」模式。

全球化擴展 (Global Ready)：架構需具備通用性，從香港 SOGO 到倫敦大笨鐘皆可無縫處理，無需手動建立貼圖庫。  

2. 核心技術棧 (Tech Stack)
後端 API：Python (FastAPI) - 負責影像處理、AI API 封裝及異步排隊邏輯。  
+ 1

前端應用：Flutter (Dart) - 單一程式碼庫編譯 iOS/Android，專注於相機調用與 Before/After 滑桿展示。  

AI 繪圖引擎：ComfyUI 架構，配合 SDXL 或 Flux.1 等先進模型。

伺服器基建：Serverless GPU (如 RunPod、Replicate) - 實現按需計費，極大化降低開發與營運成本。  

視覺大腦 (VQA)：Google Gemini 2.0 Flash - 負責歷史邏輯糾錯與 Bounding Box 標記。  
+ 1

3. 開發策略：漸進式填充 (Progressive Implementation)
為確保「一人開發」的資源集中度，採取「後端絕對優先」的解耦開發策略：  
+ 1

Phase 1 (MVP 核心)：優先固化 AI 管道（ComfyUI JSON 動態注入）、建立基礎 API (routes_jobs.py) 與生圖對接 (inference_service.py)。  
+ 1

Phase 2 (Agentic 質量控制)：引入視覺大腦 (VLM) 進行 QA 評估與自我修正循環 (qa_service.py / vlm_service.py)，確保出圖具備商業價值。  

Phase 3 (非同步擴展)：上線前引入 Celery Task Workers 處理高併發排隊機制。  

4. 國際化 RAG for Graphics 數據流架構
全面捨棄手動 Photoshop 貼圖，改用全自動化檢索與注入：  

地理定位與自動 Mask：前端傳送 GPS 經緯度，後端透過 OpenStreetMap 逆向編碼識別地標（如：Big Ben），並交由 YOLOv8-seg 自動生成 Inpaint 遮罩。  

歷史圖像檢索鏈：根據地標與年代（例如 "Big Ben 1960s"），後端自動向 Wikimedia/Flickr 抓取真實歷史照片。  

動態參數鎖定：將抓取的歷史圖片餵入 IP-Adapter。鎖死關鍵參數以防變形：Global KSampler Denoise 鎖定於 0.20（保留骨架）；Inpaint Denoise 鎖定於 0.75-0.80（替換歷史材質）。  

5. VQA 併發控制與容錯規範 (Resilience First)
系統必須將 VLM 的不穩定性（如限流、超時）視為常態，嚴格遵守以下降級與限流機制：

A. 併發與限流器 (GlobalLimiter)
Slot 語意：必須統一透過 GlobalLimiter (Phase 1 為 Semaphore，Phase 2 為 Redis) 獲取 In-flight 資源。  
+ 1

防死鎖：所有鎖的獲取與持有必須具備 Timeout 機制，絕對禁止在 async with 區塊內部執行 sleep。  

B. 三級錯誤降級機制 (Graceful Degradation)
Tier 1 (CONTENT_REJECT)：正常業務邏輯。VLM 發現現代物品，進入 Inpaint 修正循環。  

Tier 2 (SYSTEM_WARNING_SKIP)：遇到 429 限流、503 擁堵、Timeout 或 JSON 解析失敗。觸發優雅降級，直接繞過 VQA 審查，回傳當前 Base 圖，絕對不允許出現白畫面。  

Tier 3 (SYSTEM_FATAL)：致命配置錯誤（如 403 無權限或 API Key 遺失）。立即中斷任務並返回 HTTP 500。  

C. 止血條款 (Stop-Bleeding)
重試機制必須同時受到「最大重試次數」（如 3 次）與「總耗時上限」（如 30 秒）的雙重約束。  

必須使用 Exponential Backoff + Jitter 進行退避等待。  

6. 可觀測性與 Debug 追蹤 (Observability)
為確保後續除錯效率，每個生成任務必須建立獨立的 Artifacts 目錄 (logs/jobs/job_id/)，並記錄每一個 Pass 的完整生命週期：  

必存產物：原始處理圖、VLM 輸出的 Error Mask (.png)、JSON 決策日誌、最終輸出圖。  

可重現性：嚴禁黑盒隨機性，所有 Seed、Prompt 與 Denoise 數值必須寫入 Metadata 留存。  