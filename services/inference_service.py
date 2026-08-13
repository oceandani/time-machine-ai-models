import json
import uuid
import random
import time
import requests
import os
from fastapi import HTTPException
from websocket import create_connection

# 引入企業級日誌系統
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

    try:
        with open("workflows/workflow_api.json", "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="系統缺少工作流檔案")

    if "11" in prompt_data: prompt_data["11"]["inputs"]["seed"] = random.randint(1, 1000000000)
    if "5" in prompt_data: prompt_data["5"]["inputs"]["text"] = f"vintage photography of {location} in {year}, highly detailed, empty street, focus on architecture, no crowds, high quality"
    if "2" in prompt_data: prompt_data["2"]["inputs"]["image"] = comfy_image_name

    return _execute_comfyui_workflow(client_id, prompt_data)

def run_comfyui_inpaint(year: str, location: str, base_image_path: str, mask_image_path: str) -> bytes:
    """
    執行 ComfyUI 局部重繪 (Inpaint) 任務。
    讀取 inpaint_workflow_api.json，並將 Node 2 設為底圖，Node 999 設為 Mask。
    """
    client_id = str(uuid.uuid4())
    logger.info(f"[{client_id}] 🎨 開始執行 Inpaint 修正...")
    
    # 1. 必須將兩張圖片 (底圖 + 遮罩) 都上傳到 ComfyUI 伺服器
    base_img_name = upload_image_to_comfyui(base_image_path)
    mask_img_name = upload_image_to_comfyui(mask_image_path)

    try:
        with open("configs/inpaint_workflow_api.json", "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="缺少 inpaint_workflow_api.json")

    # 2. 動態替換 Inpaint 專用的節點參數
    if "2" in prompt_data: prompt_data["2"]["inputs"]["image"] = base_img_name
    if "999" in prompt_data: prompt_data["999"]["inputs"]["image"] = mask_img_name
    if "23" in prompt_data: prompt_data["23"]["inputs"]["seed"] = random.randint(1, 1000000000)
    if "5" in prompt_data: prompt_data["5"]["inputs"]["text"] = f"vintage photography of {location} in {year}, highly detailed, focus on architecture"

    return _execute_comfyui_workflow(client_id, prompt_data)

def _execute_comfyui_workflow(client_id: str, prompt_data: dict) -> bytes:
    """共用的 ComfyUI 執行與監聽邏輯"""
    try:
        ws = create_connection(f"{COMFYUI_WS_URL}?clientId={client_id}")
        ws.settimeout(1.0)
    except Exception as e:
        logger.error(f"無法連接 WebSocket: {e}")
        raise HTTPException(status_code=500, detail=f"無法連接 WebSocket: {e}")

    payload = {"prompt": prompt_data, "client_id": client_id}
    response = requests.post(f"{COMFYUI_HTTP_URL}/prompt", json=payload)
    response.raise_for_status()
    prompt_id = response.json().get("prompt_id")
    
    logger.info(f"[{client_id}] 任務已提交 (ID: {prompt_id})，開始監控...")

    final_image_info = None
    start_time = time.time()
    TIMEOUT_SECONDS = 600 
    
    while time.time() - start_time < TIMEOUT_SECONDS:
        try:
            hist_resp = requests.get(f"{COMFYUI_HTTP_URL}/history/{prompt_id}", timeout=1)
            if hist_resp.status_code == 200:
                hist = hist_resp.json()
                if prompt_id in hist:
                    outputs = hist[prompt_id].get("outputs", {})
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            final_image_info = node_output["images"][0]
                            break
                    if final_image_info: break
        except Exception:
            pass 
            
        try:
            out = ws.recv()
            if isinstance(out, str):
                msg = json.loads(out)
                if msg.get("type") == "executed":
                    final_image_info = msg["data"]["output"]["images"][0]
                    break
        except Exception:
            pass
    
    ws.close()
    
    if not final_image_info: 
        raise HTTPException(status_code=500, detail="生圖超時或失敗")
        
    logger.info(f"[{client_id}] 成功獲取圖片: {final_image_info['filename']}")
    return download_image(final_image_info["filename"], "", "output")