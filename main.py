from fastapi import FastAPI, File, UploadFile, Form
import json
import requests
import random
import os

app = FastAPI(title="Time Machine API")

# 假設你目前嘅 ComfyUI 係喺本地 Mac 運行
COMFYUI_URL = "http://127.0.0.1:8188/prompt" 

@app.post("/generate-vintage")
async def generate_vintage_photo(
    year: str = Form(...),            # 接收年份 (例如: "1960s")
    location: str = Form(...),        # 接收地點 (例如: "Causeway Bay, Hong Kong")
    image: UploadFile = File(...)     # 接收用戶上傳嘅圖片
):
    # 1. 儲存用戶上傳嘅圖片 (供 ComfyUI 讀取)
    input_image_path = f"temp_input_{image.filename}"
    with open(input_image_path, "wb") as f:
        f.write(await image.read())
        
    # 2. 讀取 ComfyUI 嘅 JSON 說明書
    with open("workflows/workflow_api.json", "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
        
    # ==========================================
    # 3. 動態修改 JSON 內嘅參數 (核心魔法)
    # ==========================================
    
    # [重點解說 A]: 替換 Seed 確保每次生成都有啲唔同
    # 假設 KSampler_Inpaint 嘅節點 ID 係 "13" (你需要自己打開 JSON 確認真實 ID)
    if "23" in prompt_data:
        prompt_data["23"]["inputs"]["seed"] = random.randint(1, 1000000000000000)
        
    # [重點解說 B]: 動態替換 Global Prompt (加入年份同地點)
    # 假設 Global Positive Prompt 嘅 CLIP Text Encode 節點 ID 係 "6"
    if "5" in prompt_data:
        dynamic_prompt = f"{year} vintage photography of {location}, historical street view, highly detailed"
        prompt_data["5"]["inputs"]["text"] = dynamic_prompt
        
    # [重點解說 C]: 替換 Load Image 嘅路徑為用戶剛上傳嘅圖片
    # 假設 Load Image 嘅節點 ID 係 "10"
    if "2" in prompt_data:
        # 注意: 實際商用時，呢度可能係轉成 Base64 或上傳到雲端嘅 URL
        prompt_data["2"]["inputs"]["image"] = input_image_path

    # ==========================================
    # 4. 將修改好嘅 Payload 發送給 ComfyUI 伺服器
    # ==========================================
    payload = {"prompt": prompt_data}
    
    try:
        response = requests.post(COMFYUI_URL, json=payload)
        response.raise_for_status()
        
        # 提取 ComfyUI 返回嘅任務 ID (prompt_id)
        prompt_id = response.json().get("prompt_id")
        
        # 為了 Prototype 簡單化，目前先直接返回任務 ID
        # 稍後我哋會寫 WebSocket 來實時監聽生圖完成狀態並返回圖片 URL
        return {
            "status": "success", 
            "message": "生圖任務已成功提交", 
            "prompt_id": prompt_id,
            "used_prompt": dynamic_prompt
        }
        
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"無法連接 ComfyUI: {str(e)}"}