import cv2
import numpy as np
from PIL import Image
import os
import sys
import torch
import json
from AECIF_Net import HRnet_Segmentation
from openai import OpenAI

# ==========================================
# 1. 配置区域 (Groq 免费方案)
# ==========================================
# 🔑 请在这里填入你的 Groq Key (以 gsk_ 开头)
API_KEY = "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"

# Groq 的配置 (不要改)
BASE_URL = "https://api.groq.com/openai/v1"
# MODEL_NAME = "llama3-8b-8192"  # 速度极快的 Llama 3 模型
MODEL_NAME = "llama-3.3-70b-versatile"

# 文件路径
WEIGHT_FILE = 'model_data/best_epoch_weights.pth'
TEST_IMAGE = 'img/1.jpg'


# ==========================================
# 2. 这里的 Prompt 是核心：定义了 LLM 的世界观
# ==========================================
def ask_ai_universal(user_query):
    print(f"\n[AI Brain] 分析用户意图: '{user_query}' ...")

    if "gsk_" not in API_KEY and "sk-" not in API_KEY:
        print("❌ 错误: 请填入 API Key")
        return None

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # 🌟 核心升级：让 LLM 成为逻辑分析师
    system_prompt = """
    You are an intelligent Bridge Inspection Agent. 
    You have access to a segmented image with the following available data layers (Masks):

    【Data Layers】
    - ELEMENTS: Bearing(1), Bracing(2), Deck(3), Floor Beam(4), Girder(5), Pier(6)
    - DEFECTS: Rust(1)

    【Your Job】
    Translate the user's natural language query into a strict JSON execution plan.
    You must decide logically how to combine these layers to answer the user.

    【Logic Types】
    1. "Show me X": Visualize specific targets.
    2. "Show X on Y" (Intersection): You want to see defects (X) ONLY within the area of an element (Y).
    3. "How much X?" / "Is it serious?": Calculate the area percentage.
    4. "Is there any X?": Check if the mask area > 0.

    【Output JSON Schema】
    {
        "intent": "visualize" | "analyze", 
        "target_layers": [{"type": "elements"|"defects", "id": int, "name": str}, ...],
        "constraint_layer": {"type": "elements", "id": int, "name": str} | null,
        "description": "Short explanation of the logic (e.g., 'Calculating rust coverage on the girder')"
    }

    【Examples】
    - User: "Where is the girder?" 
      -> {"intent": "visualize", "target_layers": [{"type": "elements", "id": 5, "name": "Girder"}], "constraint_layer": null, "description": "Locating girder"}

    - User: "Show me the rust on the floor beam" 
      -> {"intent": "visualize", "target_layers": [{"type": "defects", "id": 1, "name": "Rust"}], "constraint_layer": {"type": "elements", "id": 4, "name": "Floor Beam"}, "description": "Filtering rust on floor beam"}

    - User: "How bad is the corrosion on the pier?" 
      -> {"intent": "analyze", "target_layers": [{"type": "defects", "id": 1, "name": "Rust"}], "constraint_layer": {"type": "elements", "id": 6, "name": "Pier"}, "description": "Calculating rust percentage on pier"}

    - User: "Is the bridge safe?" (Implies checking for defects)
      -> {"intent": "analyze", "target_layers": [{"type": "defects", "id": 1, "name": "Rust"}], "constraint_layer": null, "description": "Checking total corrosion amount"}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            temperature=0.1,
        )
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)

    except Exception as e:
        print(f"❌ AI 解析失败: {e}")
        return None


# ==========================================
# 3. 智能加载器
# ==========================================
def load_model_smartly():
    print(f"🚀 初始化 AECIF-Net...")
    try:
        if torch.cuda.is_available():
            dummy = torch.tensor([1.0]).cuda()
            res = dummy + 1.0
            print("✅ GPU 模式启动！")
            return HRnet_Segmentation(model_path=WEIGHT_FILE, cuda=True)
    except RuntimeError:
        pass
    print("🔄 使用 CPU 模式...")
    return HRnet_Segmentation(model_path=WEIGHT_FILE, cuda=False)


# ==========================================
# 4. 执行引擎 (Python Logic)
# ==========================================
def execute_plan(hrnet, image, plan):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]

    print(f"⚙️ [Executor] 执行逻辑: {plan['description']}")

    # 1. 获取所有原始 Mask
    mask_e, mask_d = hrnet.get_raw_masks(image)

    # 2. 构建【目标集合】 (Target Mask)
    # 逻辑：多个目标之间取并集 (Union)
    target_mask = np.zeros((h, w), dtype=np.uint8)
    target_names = []

    for item in plan['target_layers']:
        target_names.append(item['name'])
        if item['type'] == 'elements':
            target_mask = cv2.bitwise_or(target_mask, (mask_e == item['id']).astype(np.uint8))
        elif item['type'] == 'defects':
            target_mask = cv2.bitwise_or(target_mask, (mask_d == item['id']).astype(np.uint8))

    # 3. 构建【约束集合】 (Constraint Mask)
    # 逻辑：如果有约束，取交集 (Intersection)
    roi_mask = None
    if plan['constraint_layer']:
        c_item = plan['constraint_layer']
        print(f"   -> 施加空间约束: 仅限 {c_item['name']} 区域")
        if c_item['type'] == 'elements':
            roi_mask = (mask_e == c_item['id']).astype(np.uint8)
        else:
            roi_mask = (mask_d == c_item['id']).astype(np.uint8)

        # 核心：目标 AND 约束
        target_mask = cv2.bitwise_and(target_mask, roi_mask)

    # 4. 分析与计算 (Analysis)
    pixel_count = np.sum(target_mask > 0)
    report_text = ""

    if plan['intent'] == 'analyze':
        if pixel_count == 0:
            report_text = "Analysis Result: None detected."
        else:
            # 如果有约束区域，计算相对比例
            if roi_mask is not None:
                roi_pixels = np.sum(roi_mask > 0)
                if roi_pixels > 0:
                    ratio = (pixel_count / roi_pixels) * 100
                    report_text = f"Severity: {ratio:.2f}% of {plan['constraint_layer']['name']} is affected."
                else:
                    report_text = "Constraint area not found."
            else:
                # 否则计算全图比例
                ratio = (pixel_count / (h * w)) * 100
                report_text = f"Coverage: {ratio:.2f}% of total image."

        print(f"📊 [Report] {report_text}")

    elif plan['intent'] == 'visualize':
        if pixel_count > 0:
            report_text = f"Visualizing: {', '.join(target_names)}"
        else:
            report_text = f"Not Found: {', '.join(target_names)}"

    # 5. 可视化渲染
    if pixel_count > 0:
        overlay = img_cv.copy()

        # 渲染目标 (红色高亮)
        overlay[target_mask > 0] = [0, 0, 255]

        # 如果有约束区域，把约束区域也画个淡淡的轮廓（比如蓝色），方便对比
        if roi_mask is not None:
            contours_roi, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_cv, contours_roi, -1, (255, 0, 0), 1)  # 蓝色细线表示约束范围

        # 混合
        res_img = cv2.addWeighted(img_cv, 0.7, overlay, 0.3, 0)

        # 画目标轮廓 (黄色)
        contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(res_img, contours, -1, (0, 255, 255), 2)

        # 在图上写报告
        cv2.rectangle(res_img, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(res_img, report_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Bridge Analysis Agent", res_img)
        print("✅ 结果展示中，按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"⚠️ 分析结果: 未在指定区域检测到相关目标。")


# ==========================================
# 5. 主入口
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(WEIGHT_FILE):
        sys.exit("❌ 权重文件丢失")
    if not os.path.exists(TEST_IMAGE):
        sys.exit("❌ 图片丢失")

    hrnet = load_model_smartly()
    image = Image.open(TEST_IMAGE)

    print("\n💡 尝试问我各种问题:")
    print(" - 'Is the pier rusted?' (判断)")
    print(" - 'How bad is the corrosion on the girder?' (定量分析)")
    print(" - 'Where are the bearings?' (定位)")
    print(" - 'Show me rust and cracks' (LLM会知道crack不在数据库里)")

    while True:
        query = input("\n💬 Bridge Agent (You): ")
        if query.lower() in ['q', 'quit']: break

        plan = ask_ai_universal(query)
        if plan:
            execute_plan(hrnet, image, plan)