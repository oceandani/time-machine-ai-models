import json
import uuid
import random
import time
import requests
import os
from fastapi import HTTPException
from websocket import create_connection

# 引入我哋啱啱建立嘅企業級日誌系統
from core.logging import logger

COMFYUI_HTTP_URL = "http://127.0.0.1:8188"
COMFYUI_WS_URL = "ws://127.0.0.1:8188/ws"

def upload_image_to_comfyui(image_path: str) -> str:
    upload_url = f"{COMFYUI_HTTP_URL}/upload/image"
    try:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f)}
            data = {"overwrite": "true"}
            response = requests.post(upload_url, files=files, data=data)
            response.raise_for_status()
            return response.json()["name"]
    except Exception as e:
        logger.error(f"上傳圖片至 ComfyUI 失敗: {e}")
        raise HTTPException(status_code=500, detail=f"上傳圖片失敗: {e}")

def download_image(filename: str, subfolder: str, image_type: str) -> bytes:
    view_url = f"{COMFYUI_HTTP_URL}/view?filename={filename}&subfolder={subfolder}&type={image_type}"
    img_response = requests.get(view_url)
    img_response.raise_for_status()
    return img_response.content

def run_comfyui_inference(year: str, location: str, input_image_path: str) -> bytes:
    client_id = str(uuid.uuid4())
    
    logger.info(f"[{client_id}] 1. 正在上傳圖片至 ComfyUI...")
    comfy_image_name = upload_image_to_comfyui(input_image_path)
    logger.info(f"[{client_id}] 2. 圖片上傳成功，ComfyUI 檔名: {comfy_image_name}")

    try:
        with open("workflows/workflow_api.json", "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
    except FileNotFoundError:
        logger.error("找不到 workflows/workflow_api.json 檔案")
        raise HTTPException(status_code=500, detail="系統缺少工作流檔案")

    # 替換參數
    if "11" in prompt_data: prompt_data["11"]["inputs"]["seed"] = random.randint(1, 1000000000)
    if "5" in prompt_data: prompt_data["5"]["inputs"]["text"] = f"{year} vintage photography of {location}"
    if "2" in prompt_data: prompt_data["2"]["inputs"]["image"] = comfy_image_name

    try:
        ws = create_connection(f"{COMFYUI_WS_URL}?clientId={client_id}")
        # 關鍵：設定 WebSocket 1 秒超時，防止死等
        ws.settimeout(1.0)
    except Exception as e:
        logger.error(f"無法連接 ComfyUI WebSocket: {e}")
        raise HTTPException(status_code=500, detail=f"無法連接 WebSocket: {e}")

    payload = {"prompt": prompt_data, "client_id": client_id}
    # 💡 修復：正確獲取 prompt_id (任務 ID)
    response = requests.post(f"{COMFYUI_HTTP_URL}/prompt", json=payload)
    response.raise_for_status()
    prompt_id = response.json().get("prompt_id")
    
    logger.info(f"[{client_id}] 3. 任務已提交 (ID: {prompt_id})，開始監控生圖進度...")

    # 監控邏輯
    final_image_info = None
    start_time = time.time()
    
    while time.time() - start_time < 180:
        # 防彈機制一：主動查 History (用正確嘅 prompt_id)
        try:
            hist_resp = requests.get(f"{COMFYUI_HTTP_URL}/history/{prompt_id}", timeout=1)
            if hist_resp.status_code == 200:
                hist = hist_resp.json()
                if prompt_id in hist:
                    outputs = hist[prompt_id].get("outputs", {})
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            logger.info(f"[{client_id}] ✅ 從歷史紀錄中確認任務生成完畢！")
                            final_image_info = node_output["images"][0]
                            break
                    if final_image_info:
                        break # 搵到圖片即刻中斷死等
        except Exception:
            pass # 查唔到就繼續
            
        # 防彈機制二：聽 WebSocket
        try:
            out = ws.recv()
            if isinstance(out, str):
                msg = json.loads(out)
                if msg.get("type") == "progress":
                    logger.info(f"[{client_id}] ⏳ KSampler 處理中: {msg['data']['value']} / {msg['data']['max']}")
                elif msg.get("type") == "executed":
                    logger.info(f"[{client_id}] ✅ WebSocket 報告任務生成完畢！")
                    final_image_info = msg["data"]["output"]["images"][0]
                    break
        except Exception:
            # 1 秒 timeout 觸發，安全略過，進入下一次迴圈
            pass
    
    ws.close()
    
    if not final_image_info: 
        logger.error(f"[{client_id}] ❌ 生圖超時 (超過 180 秒) 或未能獲取圖片")
        raise HTTPException(status_code=500, detail="生圖超時")
        
    logger.info(f"[{client_id}] 4. 正在下載並回傳最終相片: {final_image_info['filename']}")
    return download_image(final_image_info["filename"], "", "output")