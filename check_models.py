import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("🚨 找不到 GEMINI_API_KEY，請檢查 .env 檔案")
        return

    client = genai.Client(api_key=api_key)
    
    print("🔍 正在向 Google 查詢你個 API Key 專屬嘅可用模型名單...\n")
    
    try:
        # 修正：新版 SDK 正確嘅呼叫方法係 list()
        models = client.models.list()
        valid_models = []
        
        for m in models:
            # 直接篩選並印出包含 'flash' 關鍵字嘅模型
            if "flash" in m.name.lower():
                valid_models.append(m.name)
                print(f"✅ 發現可用 Flash 模型: {m.name}")
                
        if not valid_models:
            print("⚠️ 找不到任何帶有 'flash' 的模型！以下為所有可用模型：")
            for m in client.models.list():
                print(f"- {m.name}")
        else:
            print("\n💡 解決方案：請將上方其中一個模型名稱（如果帶有 'models/' 前綴請略過，只需保留後段如 'gemini-1.5-flash-002'）")
            print("複製並完整貼上到 vlm_service.py 裡的 CANDIDATE_MODELS 陣列中！")
            
    except Exception as e:
        print(f"❌ 查詢失敗: {str(e)}")

if __name__ == "__main__":
    main()