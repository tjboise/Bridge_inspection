
import os
import cv2
import numpy as np
import google.generativeai as genai
import torch
import json
from core_logic import load_model_core, ask_gemini_planner_core, process_vision_core

# 配置
DATA_ROOT = "data"
IMG_DIR = os.path.join(DATA_ROOT, "JPEGImages")

# 1. 检查 API Key
api_key = os.environ.get("GOOGLE_API_KEY")
print(f"🔑 API Key set: {bool(api_key)}")
if api_key:
    print(f"   Key prefix: {api_key[:4]}...")

# 2. 检查图片
print("\n1️⃣ Checking Image...")
img_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
if not img_files:
    print("❌ No images found!")
    exit()
test_img_id = img_files[0]
test_img_path = os.path.join(IMG_DIR, test_img_id)
print(f"   Using image: {test_img_path}")
img = cv2.imread(test_img_path)
if img is None:
    print("❌ Failed to read image.")
else:
    print(f"   Image shape: {img.shape}")

# 3. 检查模型 (Model)
print("\n2️⃣ Checking Model Inference...")
hrnet = load_model_core()
if hrnet:
    try:
        mask_e, mask_d = hrnet.get_raw_masks(img)
        print(f"   Mask E (Elements) range: {mask_e.min()} - {mask_e.max()}")
        print(f"   Mask D (Defects) range: {mask_d.min()} - {mask_d.max()}")
        if np.sum(mask_e) == 0 and np.sum(mask_d) == 0:
            print("❌ WARNING: Model returned ALL ZERO masks! (Model issue)")
        else:
            print("✅ Model inference successful (Non-zero output).")
    except Exception as e:
        print(f"❌ Model inference failed: {e}")
else:
    print("❌ Failed to load model.")

# 4. 检查 Planner (API)
print("\n3️⃣ Checking Gemini Planner API...")
test_prompt = "Segment the Bearing"
print(f"   Sending prompt: '{test_prompt}'")

try:
    plan = ask_gemini_planner_core(test_prompt)
    print(f"   Planner Output: {json.dumps(plan, indent=2)}")

    if not plan or not plan.get("target_layers"):
        print("❌ Planner returned empty or invalid plan.")
    else:
        print("✅ Planner working correctly.")

        # 5. 检查综合流程
        print("\n4️⃣ Checking Full Pipeline...")
        if hrnet:
            final_mask = process_vision_core(hrnet, test_img_path, plan)
            print(f"   Final Mask Sum: {np.sum(final_mask)}")
            if np.sum(final_mask) > 0:
                print("✅ Full pipeline SUCCESS!")
            else:
                print("❌ Full pipeline produced ZERO mask.")

except Exception as e:
    print(f"❌ Planner API failed: {e}")
