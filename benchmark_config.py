print("🚀 Script is starting...")
import os
import time
import random  # <--- 新增 import

# ============================
# 0. 自动配置 API Key
# ============================
API_KEY_FILE = "api_key.txt"
MY_API_KEY = None

if os.path.exists(API_KEY_FILE):
    try:
        with open(API_KEY_FILE, "r", encoding="utf-8") as f:
            MY_API_KEY = f.read().strip()
    except:
        pass

if not MY_API_KEY:
    MY_API_KEY = os.getenv("GOOGLE_API_KEY")

if not MY_API_KEY:
    print("\n❌ CRITICAL ERROR: API Key not found! Please check 'api_key.txt'.")
    exit(1)

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

# 路径配置
DATA_ROOT = "data"
IMG_DIR = os.path.join(DATA_ROOT, "JPEGImages")
GT_DEFECT_DIR = os.path.join(DATA_ROOT, "SegmentationClass", "defect")
GT_ELEMENT_DIR = os.path.join(DATA_ROOT, "SegmentationClass", "element")
TEST_LIST_PATH = os.path.join(DATA_ROOT, "ImageSets", "Segmentation", "test.txt")

VIS_DIR = "benchmark_visualizations"
if not os.path.exists(VIS_DIR): os.makedirs(VIS_DIR)

# 映射配置
ELEMENT_MAP = {1: "Bearing", 2: "Bracing", 3: "Deck", 4: "Floor Beam", 5: "Girder", 6: "Pier"}
DEFECT_MAP = {1: "Rust"}
# 颜色配置 (用于可视化 CNN 原始输出)
COLORS_ELEMENTS = np.array([
    [0, 0, 0],  # Background
    [0, 0, 128],  # Bearing
    [0, 128, 0],  # Bracing
    [128, 0, 0],  # Deck
    [128, 128, 0],  # Floor Beam
    [0, 0, 128],  # Girder
    [128, 0, 128],  # Pier
    [0, 128, 128]  # Extra
], dtype=np.uint8)

COLORS_DEFECTS = np.array([
    [0, 0, 0],  # Background
    [0, 0, 255]  # Rust
], dtype=np.uint8)


# ============================
# 1. 辅助函数
# ============================
def sanitize_plan(plan):
    if not isinstance(plan, dict): return {"target_layers": []}

    def clean(layers):
        if not isinstance(layers, list): return []
        cleaned = []
        for item in layers:
            if isinstance(item, str): item = {"name": item}
            if not isinstance(item, dict): continue

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
        not_bg = np.logical_not(np.all(m_rgb == [0, 0, 0], axis=-1))
        return not_bg.astype(np.uint8)
    else:
        COLORS_GT = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128)]
        try:
            target = COLORS_GT[id_]
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
        text = response.text.replace("```json", "").replace("```", "").strip()
        start = text.find('{');
        end = text.rfind('}')
        if start != -1 and end != -1:
            data = json.loads(text[start:end + 1])
            mask = np.zeros((h, w), dtype=np.uint8)
            if "polygons" in data:
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
        pass
    return np.zeros((h, w), dtype=np.uint8)


def calculate_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union > 0 else 0.0


def save_visualization_enhanced(img_id, q_id, gt, ours, base, hrnet_raw_e, hrnet_raw_d):
    cv2.imwrite(os.path.join(VIS_DIR, f"{img_id}_{q_id}_1_GT.png"), gt * 255)
    cv2.imwrite(os.path.join(VIS_DIR, f"{img_id}_{q_id}_2_Ours.png"), ours * 255)
    cv2.imwrite(os.path.join(VIS_DIR, f"{img_id}_{q_id}_3_Base.png"), base * 255)

    # Save Raw CNN
    h, w = hrnet_raw_e.shape
    vis_e = np.zeros((h, w, 3), dtype=np.uint8)
    for id_ in range(len(COLORS_ELEMENTS)):
        if id_ < 8: vis_e[hrnet_raw_e == id_] = COLORS_ELEMENTS[id_]
    cv2.imwrite(os.path.join(VIS_DIR, f"{img_id}_CNN_Raw_Element.png"), vis_e)

    vis_d = np.zeros((h, w, 3), dtype=np.uint8)
    vis_d[hrnet_raw_d == 1] = COLORS_DEFECTS[1]
    cv2.imwrite(os.path.join(VIS_DIR, f"{img_id}_CNN_Raw_Defect.png"), vis_d)


# ============================
# 3. 主流程
# ============================
def run_tests():
    print("⏳ Loading HRNet...")
    hrnet = load_model_core()

    test_ids = []
    if os.path.exists(TEST_LIST_PATH):
        with open(TEST_LIST_PATH, 'r') as f:
            full_list = [line.strip() for line in f.readlines() if line.strip()]

        # 🔥🔥🔥 随机抽样 30 张 🔥🔥🔥
        SAMPLE_SIZE = 30
        if len(full_list) > SAMPLE_SIZE:
            print(f"🎲 Found {len(full_list)} images. Randomly sampling {SAMPLE_SIZE} for testing...")
            test_ids = random.sample(full_list, SAMPLE_SIZE)
        else:
            print(f"📄 Found {len(full_list)} images. Using all of them.")
            test_ids = full_list
    else:
        all_images = [f.replace(".jpg", "") for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
        test_ids = all_images[:20]

    results = []
    print(f"🚀 Benchmarking {len(test_ids)} images: {test_ids}\n")

    for img_id in tqdm(test_ids):
        img_path = os.path.join(IMG_DIR, f"{img_id}.jpg")
        if not os.path.exists(img_path): continue

        try:
            from PIL import Image
            pil_img = Image.open(img_path).convert('RGB')
            raw_mask_e, raw_mask_d = hrnet.get_raw_masks(pil_img)
            h, w = raw_mask_e.shape[:2]
        except Exception as e:
            print(f"❌ HRNet Error on {img_id}: {e}")
            continue

        elems, defects = scan_image_contents(img_id, h, w)
        questions = generate_dynamic_questions(elems, defects)

        for q in questions:
            # GT
            gt_mask = np.zeros((h, w), dtype=np.uint8)
            if q['logic'] == 'single':
                gt_mask = read_mask_by_id(img_id, q['target_type'], q['target_ids'][0], h, w)
            if np.sum(gt_mask) == 0: continue

            # Ours
            try:
                plan_raw = ask_gemini_planner_core(q['prompt'])
                time.sleep(0.5)
            except:
                plan_raw = {}
            plan_clean = sanitize_plan(plan_raw)
            mask_ours = process_vision_core(hrnet, pil_img, plan_clean)
            iou_ours = calculate_iou(mask_ours, gt_mask)

            # Baseline
            try:
                mask_base = ask_gemini_baseline_polygon(q['prompt'], (h, w), img_path)
                time.sleep(0.5)
            except:
                mask_base = np.zeros((h, w), dtype=np.uint8)
            iou_base = calculate_iou(mask_base, gt_mask)

            save_visualization_enhanced(img_id, q['id'], gt_mask, mask_ours, mask_base, raw_mask_e, raw_mask_d)

            print(f"\n[{img_id}] Q: {q['prompt']}")
            print(f"   🔍 JSON Plan: {json.dumps(plan_clean, ensure_ascii=False)}")
            print(f"   🎯 GT Pixels: {np.sum(gt_mask)}")
            print(f"   👁️ Ours Mask: {np.sum(mask_ours)} (IoU: {iou_ours:.2f})")
            if np.sum(mask_ours) == 0 and len(plan_clean.get('target_layers', [])) > 0:
                tid = plan_clean['target_layers'][0]['id']
                ttype = plan_clean['target_layers'][0]['type']
                print(
                    f"   ⚠️ Diagnosis: Planner asked for ID {tid} ({ttype}), but CNN raw output for this ID was empty!")

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