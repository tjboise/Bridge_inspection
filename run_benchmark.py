print("🚀 Script is starting...")
import os
import time

# ============================
# 0. 配置 (必填)
# ============================
MY_API_KEY = "AIzaSyDMLr1ohvRxzcahRm6-vClKH7fcc1cGqzo"  # <--- 🛑 填 Key
os.environ["GOOGLE_API_KEY"] = MY_API_KEY

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import itertools
from PIL import Image
import google.generativeai as genai

genai.configure(api_key=MY_API_KEY)
from core_logic import load_model_core, process_vision_core, ask_gemini_planner_core

# 路径
DATA_ROOT = "data"
IMG_DIR = os.path.join(DATA_ROOT, "JPEGImages")
GT_DEFECT_DIR = os.path.join(DATA_ROOT, "SegmentationClass", "defect")
GT_ELEMENT_DIR = os.path.join(DATA_ROOT, "SegmentationClass", "element")
VIS_DIR = "benchmark_visualizations"
if not os.path.exists(VIS_DIR): os.makedirs(VIS_DIR)

# 映射
ELEMENT_MAP = {1: "Bearing", 2: "Bracing", 3: "Deck", 4: "Floor Beam", 5: "Girder", 6: "Pier"}
DEFECT_MAP = {1: "Rust"}
COLORS_ELEMENTS = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128)]
COLORS_DEFECTS = [(128, 128, 128), (128, 64, 0), (0, 0, 0)]


# ============================
# 1. 辅助函数
# ============================

def sanitize_plan(plan):
    """确保 VLM 输出包含正确的 ID"""
    if not isinstance(plan, dict): return {"target_layers": []}

    def clean(layers):
        if not isinstance(layers, list): return []
        cleaned = []
        for item in layers:
            if isinstance(item, str): item = {"name": item}
            if not isinstance(item, dict): continue

            # 补全 ID
            if "id" not in item and "name" in item:
                name = item["name"].lower()
                for eid, ename in ELEMENT_MAP.items():
                    if ename.lower() in name:
                        item["id"] = eid;
                        item["type"] = "elements";
                        break
                if "id" not in item and "rust" in name:
                    item["id"] = 1;
                    item["type"] = "defects"
            if "id" in item:
                try:
                    item["id"] = int(item["id"])
                except:
                    pass
                cleaned.append(item)
        return cleaned

    plan["target_layers"] = clean(plan.get("target_layers", []))
    plan["constraint_layers"] = clean(plan.get("constraint_layers", []))
    return plan


def read_mask_by_id(img_id, type_, id_, h, w):
    path = GT_DEFECT_DIR if type_ == "defect" else GT_ELEMENT_DIR
    m_bgr = cv2.imread(os.path.join(path, f"{img_id}.png"))
    if m_bgr is None: return np.zeros((h, w), dtype=np.uint8)
    if m_bgr.shape[:2] != (h, w): m_bgr = cv2.resize(m_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    m_rgb = cv2.cvtColor(m_bgr, cv2.COLOR_BGR2RGB)

    if type_ == "defect":
        target = COLORS_DEFECTS[1]
        mask = cv2.inRange(m_rgb, np.array(target), np.array(target))
        if np.sum(mask) == 0:
            not_bg = np.logical_not(np.all(m_rgb == [0, 0, 0], axis=-1))
            not_gy = np.logical_not(np.all(m_rgb == [128, 128, 128], axis=-1))
            mask = np.logical_and(not_bg, not_gy).astype(np.uint8) * 255
    else:
        try:
            target = COLORS_ELEMENTS[id_]
        except:
            return np.zeros((h, w), dtype=np.uint8)
        mask = cv2.inRange(m_rgb, np.array(target), np.array(target))
    return (mask > 0).astype(np.uint8)


def scan_image_contents(img_id, h, w):
    elems, defects = [], []
    for eid in ELEMENT_MAP:
        if np.sum(read_mask_by_id(img_id, "element", eid, h, w)) > 0: elems.append(eid)
    if np.sum(read_mask_by_id(img_id, "defect", 1, h, w)) > 0: defects.append(1)
    return elems, defects


def generate_dynamic_questions(existing_elements, existing_defects):
    questions = []
    # Level 1
    for eid in existing_elements:
        questions.append({
            "id": f"L1_{ELEMENT_MAP[eid]}", "level": 1,
            "prompt": f"Segment the {ELEMENT_MAP[eid]}",
            "logic": "single", "target_type": "element", "target_ids": [eid]
        })
    # Level 2 (Union)
    if len(existing_elements) >= 2:
        pairs = list(itertools.combinations(existing_elements, 2))[:1]
        for e1, e2 in pairs:
            questions.append({
                "id": f"L2_{ELEMENT_MAP[e1]}_{ELEMENT_MAP[e2]}", "level": 2,
                "prompt": f"Show me the {ELEMENT_MAP[e1]} and the {ELEMENT_MAP[e2]}",
                "logic": "union", "target_type": "element", "target_ids": [e1, e2]
            })
    # Level 3 (Rust)
    if 1 in existing_defects:
        for eid in existing_elements:
            questions.append({
                "id": f"L3_Rust_{ELEMENT_MAP[eid]}", "level": 3,
                "prompt": f"Show me the rust on the {ELEMENT_MAP[eid]}",
                "logic": "intersection", "target_type": "defect", "target_ids": [1],
                "constraint_type": "element", "constraint_id": eid
            })
    return questions


def ask_gemini_baseline_polygon(prompt, image_shape, img_path):
    h, w = image_shape
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    try:
        pil_img = Image.open(img_path)
    except:
        return np.zeros((h, w), dtype=np.uint8)
    sys_prompt = "Task: Instance Segmentation. OUTPUT JSON ONLY: {\"polygons\": [[[y1,x1],...]]}. Norm 0-100 or 0-1000."
    try:
        response = model.generate_content([sys_prompt, f"Query: {prompt}", pil_img])
        text = response.text
        mask = np.zeros((h, w), dtype=np.uint8)
        clean_text = text.replace("```json", "").replace("```", "").strip()
        start = clean_text.find('{')
        end = clean_text.rfind('}')
        if start != -1 and end != -1:
            data = json.loads(clean_text[start:end + 1])
            if "polygons" in data and isinstance(data["polygons"], list):
                for poly in data["polygons"]:
                    pts = []
                    flat = [c for p in poly for c in p if isinstance(c, (int, float))]
                    scale = 1000.0 if flat and max(flat) > 100 else 100.0
                    for p in poly:
                        if len(p) >= 2: pts.append(
                            [int(min(max(p[1], 0), scale) / scale * w), int(min(max(p[0], 0), scale) / scale * h)])
                    if len(pts) >= 3: cv2.fillPoly(mask, np.array([pts], dtype=np.int32), 1)
        return mask
    except:
        return np.zeros((h, w), dtype=np.uint8)


def calculate_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union > 0 else 0.0


def save_visualization(img_id, q_id, gt, ours, base):
    cv2.imwrite(os.path.join(VIS_DIR, f"{img_id}_{q_id}_GT.png"), gt * 255)
    cv2.imwrite(os.path.join(VIS_DIR, f"{img_id}_{q_id}_Ours.png"), ours * 255)
    cv2.imwrite(os.path.join(VIS_DIR, f"{img_id}_{q_id}_Base.png"), base * 255)


# ============================
# 3. 主流程 (带详细日志)
# ============================
def run_tests():
    print("⏳ Loading HRNet...")
    hrnet = load_model_core()

    all_images = [f.replace(".jpg", "") for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    test_ids = all_images[:20]

    results = []
    print(f"🚀 Benchmarking {len(test_ids)} images...")
    print(f"📄 Logs will be printed for each failure case.\n")

    for img_id in tqdm(test_ids):
        img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
        temp_img = cv2.imread(img_path)
        if temp_img is None: continue
        h, w = temp_img.shape[:2]

        elems, defects = scan_image_contents(img_id, h, w)
        questions = generate_dynamic_questions(elems, defects)

        for q in questions:
            # 1. GT
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            if q['logic'] == 'single':
                gt_mask = read_mask_by_id(img_id, q['target_type'], q['target_ids'][0], h, w)
            elif q['logic'] == 'union':
                for tid in q['target_ids']:
                    gt_mask = cv2.bitwise_or(gt_mask, read_mask_by_id(img_id, q['target_type'], tid, h, w))
            elif q['logic'] == 'intersection':
                t_m = read_mask_by_id(img_id, q['target_type'], q['target_ids'][0], h, w)
                c_m = read_mask_by_id(img_id, q['constraint_type'], q['constraint_id'], h, w)
                gt_mask = cv2.bitwise_and(t_m, c_m)

            if np.sum(gt_mask) == 0: continue

            # 2. Ours
            try:
                plan_raw = ask_gemini_planner_core(q['prompt'])
                time.sleep(0.3)
            except:
                plan_raw = {}

            plan_clean = sanitize_plan(plan_raw)
            mask_ours = process_vision_core(hrnet, img_path, plan_clean)
            iou_ours = calculate_iou(mask_ours, gt_mask)

            # 3. Baseline
            try:
                mask_base = ask_gemini_baseline_polygon(q['prompt'], (h, w), img_path)
                time.sleep(0.3)
            except:
                mask_base = np.zeros((h, w), dtype=np.uint8)
            iou_base = calculate_iou(mask_base, gt_mask)

            save_visualization(img_id, q['id'], gt_mask, mask_ours, mask_base)

            # 🔥🔥🔥 打印诊断信息 (Diagnostic Print) 🔥🔥🔥
            print(f"\n[{img_id}] Q: {q['prompt']}")
            print(f"   🎯 GT Pixels: {np.sum(gt_mask)}")
            print(f"   🧠 VLM Plan: {plan_clean.get('target_layers', [])}")
            print(f"   👁️ Ours Mask: {np.sum(mask_ours)} (IoU: {iou_ours:.2f})")
            print(f"   🤖 Base Mask: {np.sum(mask_base)} (IoU: {iou_base:.2f})")

            if iou_ours == 0:
                print("   ⚠️ FAIL REASON CHECK: Planner selected wrong ID? Or CNN missed it?")

            results.append({
                "Image": img_id, "Level": q['level'], "Question": q['prompt'],
                "IoU_Ours": iou_ours, "IoU_Baseline": iou_base
            })

    df = pd.DataFrame(results)
    if not df.empty:
        print("\n🏆 Results Summary:")
        print(df.groupby("Level")[["IoU_Ours", "IoU_Baseline"]].mean())
        df.to_csv("benchmark_final_debug.csv", index=False)


if __name__ == "__main__":
    run_tests()