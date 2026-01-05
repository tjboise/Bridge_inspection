import google.generativeai as genai

# 填入您的 Google Key
GOOGLE_API_KEY = "AIzaSyDBt4A8UHgyk_QhSTX70GUTDVXRLX1bm6M"
genai.configure(api_key=GOOGLE_API_KEY)

print("正在查询可用模型列表...\n")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"查询失败: {e}")