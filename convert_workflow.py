import json
import os

def convert_to_inpaint_workflow(input_json_path: str, output_json_path: str):
    """
    將 PoC 的全流程 workflow_api.json 轉換為專職 Inpaint 的工作流。
    """
    if not os.path.exists(input_json_path):
        print(f"❌ 找不到來源檔案: {input_json_path}")
        return

    with open(input_json_path, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    # 1. 刪除與 Base 生圖相關的冗餘節點 (節省 GPU 算力)
    # 包括：Base KSampler (11), VAE Decode (13), VAE Encode (34), 以及 ControlNet/MiDaS 相關節點 (7, 9, 10, 15)
    nodes_to_delete = ["7", "9", "10", "11", "13", "15", "34"]
    for node_id in nodes_to_delete:
        if node_id in workflow:
            del workflow[node_id]

    # 2. 改裝 Node 2：原本食 User 原圖，依家改為食 VQA 退回的 Base Image
    if "2" in workflow:
        workflow["2"]["_meta"]["title"] = "Load Base Image"
        workflow["2"]["inputs"]["image"] = "temp_base_image.jpg"

    # 3. 新增 Node 999：專門用來讀取 VQA 生成的 Error Mask
    workflow["999"] = {
        "inputs": {
            "image": "error_mask.png",
            "channel": "red" # 讀取圖片的紅色通道作為黑白遮罩
        },
        "class_type": "LoadImageMask",
        "_meta": {
            "title": "Load Error Mask"
        }
    }

    # 4. 重新接駁 Node 31 (VAEEncodeForInpaint)
    if "31" in workflow:
        workflow["31"]["inputs"]["pixels"] = ["2", 0] # 圖片來源改為 Node 2 (Load Base Image)
        workflow["31"]["inputs"]["mask"] = ["999", 0] # 遮罩來源改為 Node 999 (Load Error Mask)

    # 5. 重新接駁 Node 32 (InpaintModelConditioning)
    if "32" in workflow:
        workflow["32"]["inputs"]["pixels"] = ["2", 0] # 圖片來源改為 Node 2 (Load Base Image)
        workflow["32"]["inputs"]["mask"] = ["999", 0] # 遮罩來源改為 Node 999 (Load Error Mask)

    # 儲存為新的 JSON 檔案
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 成功將工作流轉換並儲存為: {output_json_path}")
    print("👉 請檢查並確保 services/inference_service.py 中的 Node ID 設定為 '2' (底圖) 與 '999' (遮罩)")

if __name__ == "__main__":
    # 設定輸入與輸出的檔案路徑
    input_file = "workflows/workflow_api.json"
    output_file = "configs/inpaint_workflow_api.json"
    
    # 確保 configs 資料夾存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    convert_to_inpaint_workflow(input_file, output_file)