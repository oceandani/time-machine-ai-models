import os
import json
import base64
import requests
from typing import Dict, Any
from core.logging import logger

def _encode_image_to_base64(image_bytes: bytes) -> str:
    """將圖片 bytes 轉換為 base64 字串，以便傳送給 OpenAI API"""
    return base64.b64encode(image_bytes).decode('utf-8')

def _build_evaluation_prompt(year: str, location: str) -> str:
    """構建給 VLM 的 System Prompt，嚴格要求輸出 JSON 格式"""
    return f"""
    You are an expert historical photography QA AI.
    Your task is to evaluate an image that is supposed to depict {location} in the {year}.
    
    Check for any severe anachronisms (items that definitely did not exist in {year}), 
    such as modern smartphones, contemporary cars (e.g., Tesla), modern clothing, or highly modern architecture.
    
    You MUST return your evaluation in strict JSON format with the following keys:
    - "pass" (boolean): true if the image is historically plausible, false if there are obvious modern anachronisms.
    - "reason" (string): Briefly explain why it passed or failed.
    - "bbox" (list of floats): If pass is false, provide a bounding box of the MOST obvious modern item. 
      Format: [x_min, y_min, x_max, y_max] where values are relative coordinates between 0.0 and 1.0. 
      (e.g., [0.1, 0.2, 0.4, 0.5]). If pass is true, this should be an empty list [].
    - "correction_prompt" (string): If pass is false, provide a short text prompt describing what should replace the anachronism to fit the {year} era (e.g., "vintage 1960s car", "empty street"). If pass is true, leave empty.
    
    Output ONLY valid JSON. No markdown formatting or code blocks.
    """

def evaluate_image(image_bytes: bytes, year: str, location: str) -> Dict[str, Any]:
    """
    呼叫 OpenAI GPT-4o 模型對圖片進行時代邏輯評估。
    回傳格式化的 JSON 字典。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("缺少 OPENAI_API_KEY 環境變數")
        raise RuntimeError("系統尚未配置 OpenAI API Key，無法進行 VLM 評估。")

    base64_image = _encode_image_to_base64(image_bytes)
    prompt = _build_evaluation_prompt(year, location)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 使用 gpt-4o 作為視覺模型，並強制開啟 JSON 回應模式
    payload = {
        "model": "gpt-4o",
        "response_format": { "type": "json_object" },
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low" # 使用 low detail 節省 token 成本與時間，對抓大錯已經足夠
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    try:
        logger.info(f"🧠 正在呼叫 GPT-4o 評估圖片 (目標: {location}, {year})...")
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result_data = response.json()
        content = result_data['choices'][0]['message']['content']
        
        # 解析 JSON 結果
        evaluation_result = json.loads(content)
        
        # 確保回傳值包含必要欄位，做基本防護
        final_result = {
            "pass": bool(evaluation_result.get("pass", True)),
            "reason": str(evaluation_result.get("reason", "No reason provided")),
            "bbox": evaluation_result.get("bbox", []),
            "correction_prompt": str(evaluation_result.get("correction_prompt", ""))
        }
        
        if final_result["pass"]:
            logger.info(f"✅ VQA 評估通過: {final_result['reason']}")
        else:
            logger.warning(f"❌ VQA 評估失敗: {final_result['reason']} (建議修正: {final_result['correction_prompt']})")
            
        return final_result

    except requests.exceptions.Timeout:
        logger.error("VLM API 請求超時")
        raise RuntimeError("VLM 評估超時，請檢查網絡連線。")
    except json.JSONDecodeError:
        logger.error(f"VLM 回傳了非法的 JSON 格式: {content}")
        raise RuntimeError("VLM 評估結果解析失敗。")
    except Exception as e:
        logger.error(f"VLM 評估發生未預期錯誤: {str(e)}")
        raise RuntimeError(f"VLM 評估失敗: {str(e)}")