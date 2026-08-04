import os
from google import genai
from dotenv import load_dotenv

# 自動載入 .env 檔案中的環境變數
load_dotenv()

def check_available_models():
    """
    連線至 Google Gemini API 並列出目前帳號/API Key 支援的所有模型名稱。
    包含地區限制與 Key 缺失的異常排查提示。
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ 錯誤: 找不到 GEMINI_API_KEY，請確認 .env 檔案設定。")
        return

    print("🔑 API Key 已載入，正在連線 Google 伺服器...\n")

    try:
        # 使用全新 google-genai SDK 寫法
        client = genai.Client(api_key=api_key)

        print("🔍 你目前可以使用的模型列表:")
        print("-" * 50)

        # 列出所有模型
        models = client.models.list()
        found = False

        for m in models:
            print(f"✅ {m.name}")
            found = True

        if not found:
            print("⚠️ 找不到任何模型。可能是 API 權限或地區問題。")

        print("-" * 50)
        # 💡 已修正：不再誤導尋找 1.5-flash，改為檢查 2.0-flash
        print("💡 請檢查上面的列表，看看是否有 'gemini-2.0-flash' 等版本。")

    except Exception as e:
        print(f"\n❌ 連線或獲取列表失敗: {e}")
        error_msg = str(e).lower()
        if "location is not supported" in error_msg or "400" in error_msg or "403" in error_msg:
            print("\n🚨🚨🚨 發現地區限制！🚨🚨🚨")
            print("Google Gemini API 目前不支援香港地區直連。")
            print("👉 請確保你的 VPN 已開啟，且 Terminal 有成功繞過地區限制（可嘗試設定 http_proxy 環境變數）。")

if __name__ == "__main__":
    check_available_models()