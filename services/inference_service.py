import json
import os
import random
import time
import uuid

import requests
from fastapi import HTTPException
from websocket import create_connection

from core.logging import logger

COMFYUI_HTTP_URL = "http://127.0.0.1:8188"
COMFYUI_WS_URL = "ws://127.0.0.1:8188/ws"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "configs")

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

def inject_prompts(prompt_data: dict, year: str, location: str):
    for node_id, node_info in prompt_data.items():
        if node_info.get("class_type") == "CLIPTextEncode":
            original_text = node_info["inputs"].get("text", "")
            if "[YEAR]" in original_text or "[LOCATION]" in original_text:
                new_text = original_text.replace("[YEAR]", year).replace("[LOCATION]", location)
                node_info["inputs"]["text"] = new_text
                logger.info(f"📝 成功將 Node {node_id} 的 Prompt 佔位符替換為: {year}, {location}")

def inject_dynamic_prompts(prompt_data: dict, positive_prompt: str, negative_prompt: str):
    """[藍圖 v4.1] 根據 _meta.title 精準注入動態路由派發的正負面 Prompt"""
    for node_id, node_info in prompt_data.items():
        if node_info.get("class_type") == "CLIPTextEncode":
            title = node_info.get("_meta", {}).get("title", "")
            if "Positive" in title:
                node_info["inputs"]["text"] = positive_prompt
                logger.info(f"📝 [Node {node_id}] 注入動態正向 Prompt")
            elif "Negative" in title:
                node_info["inputs"]["text"] = negative_prompt
                logger.info(f"📝 [Node {node_id}] 注入動態負向 Prompt")

def get_save_image_node_id(prompt_data: dict) -> str:
    for node_id, node_info in prompt_data.items():
        if node_info.get("class_type") == "SaveImage":
            return node_id
    raise ValueError("❌ 找不到 SaveImage 節點，請檢查 JSON 檔案！")

def run_comfyui_inference(year: str, location: str, input_image_path: str) -> bytes:
    client_id = str(uuid.uuid4())
    logger.info(f"[{client_id}] 1. 正在上傳圖片至 ComfyUI...")
    comfy_image_name = upload_image_to_comfyui(input_image_path)

    json_path = os.path.join(CONFIG_DIR, "pass0_workflow_api.json")
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ 找不到工作流檔案: {json_path}")
        raise HTTPException(status_code=500, detail="系統缺少 pass0_workflow_api.json")

    target_node_id = get_save_image_node_id(prompt_data)
    inject_prompts(prompt_data, year, location)

    image_injected = False
    for node_id, node_info in prompt_data.items():
        if node_info.get("class_type") == "LoadImage":
            node_info["inputs"]["image"] = comfy_image_name
            image_injected = True
        elif node_info.get("class_type") == "KSampler":
            node_info["inputs"]["seed"] = random.randint(1, 1000000000)
            
    if not image_injected:
         logger.warning("⚠️ 找不到 LoadImage 節點，無法注入原圖！")

    return _execute_comfyui_workflow(client_id, prompt_data, target_node_id=target_node_id)

def run_comfyui_inpaint(year: str, location: str, base_image_path: str, mask_image_path: str, denoise: float, pos_prompt: str, neg_prompt: str) -> bytes:
    """[藍圖 v4.1] Tier 1 局部修補修正 (整合動態 Prompt)"""
    client_id = str(uuid.uuid4())
    logger.info(f"[{client_id}] 🎨 開始執行 Tier 1 局部 Inpaint 修正...")
    
    base_img_name = upload_image_to_comfyui(base_image_path)
    mask_img_name = upload_image_to_comfyui(mask_image_path)
    json_path = os.path.join(CONFIG_DIR, "inpaint_workflow_api.json")
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ 找不到工作流檔案: {json_path}")
        raise HTTPException(status_code=500, detail="系統缺少 inpaint_workflow_api.json")

    target_node_id = get_save_image_node_id(prompt_data)
    
    # 💡 [藍圖 v4.1] 替換為動態 Prompt 注入
    inject_dynamic_prompts(prompt_data, pos_prompt, neg_prompt)

    base_injected = False
    mask_injected = False
    
    for node_id, node_info in prompt_data.items():
        if node_info.get("class_type") == "LoadImage":
            title = node_info.get("_meta", {}).get("title", "")
            if "Base" in title:
                node_info["inputs"]["image"] = base_img_name
                base_injected = True
                logger.info(f"🔗 成功將底圖注入 Node {node_id}")
            elif "Mask" in title:
                node_info["inputs"]["image"] = mask_img_name
                mask_injected = True
                logger.info(f"🔗 成功將遮罩注入 Node {node_id}")
        elif node_info.get("class_type") == "KSampler":
            node_info["inputs"]["seed"] = random.randint(1, 1000000000)
            node_info["inputs"]["denoise"] = denoise
            logger.info(f"⚙️ 成功將 Denoise ({denoise:.2f}) 注入 KSampler (Node {node_id})")

    if not base_injected or not mask_injected:
        logger.error("🚨 致命錯誤：無法在 JSON 中找到標題為 'Base Image' 或 'Mask Image' 的節點！")
        raise RuntimeError("JSON 缺少正確標題的底圖或遮罩節點")

    return _execute_comfyui_workflow(client_id, prompt_data, target_node_id=target_node_id)


def run_comfyui_global_rebuild(year: str, location: str, base_image_path: str, mask_image_path: str, denoise: float, pos_prompt: str, neg_prompt: str) -> bytes:
    """[藍圖 v4.1] Tier 2 全域受控重構：結合 Depth 引導與 VAE Inpaint Mask"""
    client_id = str(uuid.uuid4())
    logger.info(f"[{client_id}] 🌍 開始執行 Tier 2 全域受控地貌重構...")
    
    base_img_name = upload_image_to_comfyui(base_image_path)
    mask_img_name = upload_image_to_comfyui(mask_image_path)

    json_path = os.path.join(CONFIG_DIR, "global_rebuild_workflow_api.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="系統缺少 global_rebuild_workflow_api.json")

    target_node_id = get_save_image_node_id(prompt_data)
    inject_dynamic_prompts(prompt_data, pos_prompt, neg_prompt)

    base_injected, mask_injected = False, False
    
    for node_id, node_info in prompt_data.items():
        class_type = node_info.get("class_type")
        title = node_info.get("_meta", {}).get("title", "")
        
        if class_type == "LoadImage":
            if "Base" in title:
                node_info["inputs"]["image"] = base_img_name
                base_injected = True
            elif "Mask" in title:
                node_info["inputs"]["image"] = mask_img_name
                mask_injected = True
        elif class_type == "KSampler":
            node_info["inputs"]["seed"] = random.randint(1, 1000000000)
            node_info["inputs"]["denoise"] = denoise

    if not base_injected or not mask_injected:
        raise RuntimeError("Tier 2 JSON 缺少標題為 'Base Image' 或 'Mask Image' 的節點！")

    return _execute_comfyui_workflow(client_id, prompt_data, target_node_id=target_node_id)


def _execute_comfyui_workflow(client_id: str, prompt_data: dict, target_node_id: str = None) -> bytes:
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
                    if target_node_id and target_node_id in outputs:
                        if "images" in outputs[target_node_id]:
                            final_image_info = outputs[target_node_id]["images"][0]
                            break
                    elif not target_node_id:
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
                    data = msg.get("data", {})
                    if data.get("prompt_id") == prompt_id:
                        if target_node_id and data.get("node") != target_node_id:
                            continue
                        if "images" in data.get("output", {}):
                            final_image_info = data["output"]["images"][0]
                            break
        except Exception:
            pass
    
    ws.close()
    
    if not final_image_info: 
        raise HTTPException(status_code=500, detail="生圖超時或失敗")
        
    logger.info(f"[{client_id}] 成功獲取圖片: {final_image_info['filename']}")
    return download_image(final_image_info["filename"], "", "output")