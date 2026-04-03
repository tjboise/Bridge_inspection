import os
import pandas as pd
import numpy as np
from PIL import Image
import google.generativeai as genai
import time
import json

# ==========================================
# 1. 配置区
# ==========================================
GOOGLE_API_KEY = "AIzaSyDMLr1ohvRxzcahRm6-vClKH7fcc1cGqzo"
genai.configure(api_key=GOOGLE_API_KEY)

BASE_PATH = "VOCdevkit/VOC2007"
IMAGE_FOLDER = os.path.join(BASE_PATH, "JPEGImages")
MASK_FOLDER = os.path.join(BASE_PATH, "SegmentationClass", "element")
PDF_PATH = "standard/AASHTO-bridge_element_guide_manual__05092010.pdf"
INPUT_EXCEL = "Bridge_Elements_List.xlsx"
OUTPUT_EXCEL = "Bridge_Inspection_Report_CS.xlsx"

# 颜色映射 (RGB)
COLOR_MAP = {
    "Bearing": (128, 0, 0), "Bracing": (0, 128, 0), "Deck": (128, 128, 0),
    "Floor Beam": (0, 0, 128), "Girder": (128, 0, 128), "Pier": (0, 128, 128)
}


# ==========================================
# 2. 核心逻辑
# ==========================================

def get_component_crop(img_name, element_name):
    """根据 Mask 裁剪构件局部图 (带边缘缓冲)"""
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    mask_name = os.path.splitext(img_name)[0] + ".png"
    mask_path = os.path.join(MASK_FOLDER, mask_name)

    if not os.path.exists(mask_path): return None

    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("RGB")
    mask_np = np.array(mask)

    target_rgb = COLOR_MAP.get(element_name)
    if not target_rgb: return None

    match = np.all(mask_np == target_rgb, axis=-1)
    coords = np.argwhere(match)
    if coords.size == 0: return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    pad = 20  # 增加一点视野，方便 AI 看清周围环境
    left, top = max(0, x_min - pad), max(0, y_min - pad)
    right, bottom = min(img.width, x_max + pad), min(img.height, y_max + pad)

    return img.crop((left, top, right, bottom))


def evaluate_cs_with_llm(model, pdf_file, crop_img, element_name):
    """调用 Gemini 返回结构化 JSON 评价"""
    prompt = f"""
    Role: You are a Senior Bridge Inspector (AASHTO Certified).
    Task: Evaluate the Condition State (CS) of the following {element_name} based on the provided image and AASHTO manual.

    Instructions:
    1. Check for defects like corrosion, cracking, spalling, or section loss.
    2. Map the severity to AASHTO CS1 (Good), CS2 (Fair), CS3 (Poor), or CS4 (Severe).
    3. Output the result ONLY in the following JSON format:
    Example: 
    {{
        "cs": "CS1/CS2/CS3/CS4",
        "reason": "One clear sentence explaining the physical evidence found in the image."
    }}
    """

    try:
        # 使用 Gemini 2.0 Flash 处理图片和 PDF
        response = model.generate_content([pdf_file, crop_img, prompt])

        # 清洗返回内容中的 markdown 标签 (如 ```json)
        raw_text = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(raw_text)
        return data.get("cs", "Unknown"), data.get("reason", "No reason provided")
    except Exception as e:
        return "Error", str(e)


# ==========================================
# 3. 执行循环
# ==========================================

def main():
    print("📢 正在初始化：上传 AASHTO 手册...")
    pdf_handle = genai.upload_file(path=PDF_PATH)
    model = genai.GenerativeModel('gemini-2.0-flash')

    df = pd.read_excel(INPUT_EXCEL)
    cs_list = []
    reason_list = []

    print(f"🚀 开始批量检测，总计 {len(df)} 项任务\n" + "=" * 60)

    for index, row in df.iterrows():
        img_name = row['Image Name']
        el_name = row['Elements']

        if el_name == "None (Background Only)":
            cs_list.append("N/A")
            reason_list.append("No structural element detected.")
            continue

        # 裁剪图片
        crop = get_component_crop(img_name, el_name)

        if crop:
            cs, reason = evaluate_cs_with_llm(model, pdf_handle, crop, el_name)
            # --- 实时打印 ---
            print(f"[{index + 1}/{len(df)}] 图片: {img_name} | 构件: {el_name}")
            print(f"   ▶ 判定结果: {cs}")
            print(f"   ▶ 评价理由: {reason}")
            print("-" * 30)

            cs_list.append(cs)
            reason_list.append(reason)
            time.sleep(1.5)  # 避开频率限制
        else:
            cs_list.append("Error")
            reason_list.append("Failed to crop image or missing mask.")

    # 写入两列
    df['cs-llm'] = cs_list
    df['analysis-comment'] = reason_list

    df.to_excel(OUTPUT_EXCEL, index=False)
    print("=" * 60 + f"\n✅ 检测任务全部完成！\n💾 最终报告已保存至: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()