import os
import cv2
import numpy as np
import google.generativeai as genai
from core_logic import load_model_core, process_vision_core, ask_gemini_planner_core

# ============================
# 🔥 1. 在这里填 Key，确保环境没问题
# ============================
MY_API_KEY = "AIzaSyDMLr1ohvRxzcahRm6-vClKH7fcc1cGqzo"
os.environ["GOOGLE_API_KEY"] = MY_API_KEY
genai.configure(api_key=MY_API_KEY)

print("🚀 Starting Ours Pipeline Debug...")

# 1. 找一张图
IMG_DIR = os.path.join("data", "JPEGImages")
img_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
if not img_files:
    print("❌ No images found!")
    exit()
img_path = os.path.join(IMG_DIR, img_files[0])
print(f"📸 Testing on image: {img_path}")

# 2. 加载模型
print("⏳ Loading Model...")
hrnet = load_model_core()
if not hrnet:
    print("❌ Model failed to load.")
    exit()

# 3. 测试 Planner (最关键的一步！)
prompt = "Segment the Bearing"
print(f"\n🧠 Asking Planner (Ours): '{prompt}'")
print("   (If this crashes, it means Key or Model Name is wrong!)")

# 🔥 这里去掉了 try-except，让错误直接爆出来
plan = ask_gemini_planner_core(prompt)

print(f"\n✅ Planner Output:\n{plan}")

if not plan or not plan.get("target_layers"):
    print("\n❌ PROBLEM FOUND: Planner returned empty plan!")
    print("   -> Check your Model Name in core_logic.py (Must be 'models/gemini-2.0-flash')")
    print("   -> Check if API Key is hardcoded in core_logic.py")
else:
    print("\n⚡ Running Vision Process...")
    mask = process_vision_core(hrnet, img_path, plan)
    print(f"   Output Mask Sum: {np.sum(mask)}")

    if np.sum(mask) > 0:
        print("\n🎉 SUCCESS! Ours pipeline is working!")
    else:
        print("\n❌ Vision returned empty mask (ID mismatch?)")