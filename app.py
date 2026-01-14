import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import json
import os
import google.generativeai as genai
from AECIF_Net import HRnet_Segmentation

# ==========================================
# 1. Page Setup
# ==========================================
st.set_page_config(
    page_title="Bridge Inspection Dashboard",
    page_icon="🌉",
    layout="wide"
)

# ==========================================
# 0. SYSTEM CONFIGURATION
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("❌ Google API Key missing! Set it in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# 🎨 Sidebar
with st.sidebar:
    st.header("⚙️ Architecture")
    st.success("🗺️ Planner: Gemini 2.5")
    st.success("🧠 Reasoning: Gemini 2.5")
    st.info("👁️ Vision: AECIF-Net")

    st.divider()
    debug_mode = st.checkbox("🔬 Diagnostic Mode", value=True)
    st.caption("Beta Version v10.1")

# ==========================================
# 2. Backend Logic
# ==========================================

ELEMENT_MAP = {1: "Bearing", 2: "Bracing", 3: "Deck", 4: "Floor Beam", 5: "Girder", 6: "Pier"}
DEFECT_MAP = {1: "Rust"}


@st.cache_resource
def load_model():
    weight_path = 'model_data/best_epoch_weights.pth'
    if not os.path.exists(weight_path): return None, "Weight file missing"
    print("🔒 Forcing CPU mode...")
    model = HRnet_Segmentation(model_path=weight_path, cuda=False)
    return model, "CPU"


def get_best_model():
    return [
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite',
        'gemini-1.5-flash',
        'gemini-3-flash',
        'gemma-3-12b'
    ]


def clean_json_string(text):
    text = text.replace("```json", "").replace("```", "").strip()
    s = text.find('{')
    e = text.rfind('}')
    if s != -1 and e != -1:
        return text[s:e + 1]
    return text


def business_logic_refine(plan, query):
    """
    🏢 业务逻辑兜底
    """
    q = query.lower()

    # 规则 1: Overview 必须查 Rust (为了让 Reasoning 知道有锈，但具体的 Prompt 会控制不说数字)
    if any(k in q for k in ["overview", "defect", "summary", "check", "condition", "describe"]):
        if plan['intent'] != 'detect_defects': plan['intent'] = 'detect_defects'
        targets = plan.get('target_layers', [])
        if not targets: targets = []
        if not any(t.get('name') == 'Rust' for t in targets):
            targets.append({"type": "defects", "id": 1, "name": "Rust"})
        plan['target_layers'] = targets

    # 规则 2: Show/Segment -> Visualize
    if any(k in q for k in ["show", "see", "visual", "segment", "highlight", "draw"]):
        plan['intent'] = 'visualize'

    # 规则 3: 实体 ID 强行锁定
    detected_elements = []
    for eid, name in ELEMENT_MAP.items():
        if name.lower() in q or name.lower() + "s" in q:
            detected_elements.append({"type": "elements", "id": eid, "name": name})
    if "rust" in q or "corrosion" in q:
        detected_elements.append({"type": "defects", "id": 1, "name": "Rust"})

    if detected_elements and not ("all" in q and ("element" in q or "part" in q)):
        plan['target_layers'] = detected_elements
        if plan['intent'] != 'detect_defects': plan['intent'] = 'visualize'

    # 规则 4: 全量显示
    has_all = "all" in q or "every" in q or "whole" in q
    has_part = "element" in q or "part" in q or "component" in q
    if has_all and has_part:
        plan['intent'] = 'visualize'
        plan['target_layers'] = [{"type": "elements", "id": i, "name": name} for i, name in ELEMENT_MAP.items()]
        plan['constraint_layers'] = []

        # 规则 5: 约束清洗
    spatial_prepositions = [" on ", " in ", " within ", " inside ", " atop "]
    has_spatial = any(prep in f" {q} " for prep in spatial_prepositions)

    if has_spatial and detected_elements:
        has_defects = any(d['type'] == 'defects' for d in detected_elements)
        has_elems = any(d['type'] == 'elements' for d in detected_elements)

        if has_defects and has_elems:
            plan['target_layers'] = [d for d in detected_elements if d['type'] == 'defects']
            plan['constraint_layers'] = [d for d in detected_elements if d['type'] == 'elements']
        elif not has_spatial:
            plan['constraint_layers'] = []
    elif not has_spatial:
        plan['constraint_layers'] = []

    return plan


def ask_gemini_planner(query):
    models = get_best_model()
    base_prompt = """
    Role: Bridge Inspection Planner. Task: Convert user query to JSON.
    Output JSON Schema: {"intent": "visualize"|"detect_defects"|"chat", "reply": "str", "target_layers": [], "constraint_layers": []}
    """
    log_buffer = ""
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(base_prompt + f"\nUser Query: {query}")
            draft_text = clean_json_string(response.text)
            try:
                plan = json.loads(draft_text)
                if "intent" not in plan: raise ValueError("Missing 'intent'")
                final_plan = business_logic_refine(plan, query)
                log_buffer += f"✅ Planner ({model_name}) succeeded.\n"
                return final_plan, log_buffer
            except Exception as e:
                log_buffer += f"⚠️ {model_name} format error. Retrying...\n"
                continue
        except Exception as e:
            continue
    return keyword_rescue(query, log_buffer)


def keyword_rescue(query, previous_logs):
    q = query.lower()
    log = previous_logs + "\n💀 Planner failed. Using Keyword Rescue."
    dummy = {"intent": "visualize", "target_layers": [], "constraint_layers": []}
    rescued = business_logic_refine(dummy, query)
    if rescued['target_layers']: return rescued, log
    return {"intent": "detect_defects", "target_layers": [{"type": "defects", "id": 1, "name": "Rust"}],
            "constraint_layers": []}, log


def generate_reasoning_response(query, stats, image, intent):
    """
    Step 3: Reasoning (核心修改部分)
    """
    models = get_best_model()
    q_lower = query.lower()

    # 判断是否为 General Overview 类问题
    is_general_overview = any(
        k in q_lower for k in ["overview", "describe", "what is this", "explain the image", "general"])

    if intent == 'visualize':
        prompt = f"""
        User Query: "{query}"
        [Visual Analysis Data]: {stats}

        [TASK]
        User wants visualization. 
        1. Confirm highlighted areas.
        2. Brief (Max 2 sentences).
        3. NO mention of "crops".
        """
    elif is_general_overview:
        # 🔥 针对 Overview 的特殊 Prompt：禁止读数
        prompt = f"""
        User Query: "{query}"
        [AI Detection Data]: {stats} (Use this for internal confirmation ONLY)

        [TASK]
        Act as a Senior Bridge Inspector providing a general site overview.
        1. Describe the structure type and context (e.g., bridge over railway).
        2. Mention visible conditions **qualitatively** (e.g., "signs of corrosion", "paint delamination").
        3. ⛔️ RESTRICTION: Do NOT quote specific percentage numbers or mention "AI data/Sensor data" in the output. Keep it natural and professional.
        4. NO mention of "crops".
        """
    else:
        # 🔥 针对具体评估 (Assessment) 的 Prompt：允许读数
        prompt = f"""
        User Query: "{query}"
        [AI Detection Data]: {stats}

        [TASK]
        Senior Bridge Inspector Reasoning.
        1. Direct Answer to the specific question.
        2. You MAY quote the percentage coverage from the [AI Detection Data] to support your assessment of severity.
        3. Use professional engineering terminology.
        4. NO mention of "crops".
        """

    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            res = model.generate_content([prompt, image])
            return res.text
        except:
            continue
    return "Reasoning Error: All models failed."


def process_vision_smart(hrnet, image_pil, plan, debug=False):
    if plan.get('intent') == 'chat': return None, "", []

    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    mask_e, mask_d = hrnet.get_raw_masks(image_pil)

    res_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    canvas = np.zeros_like(res_img)
    mask_bool = np.zeros((h, w), dtype=bool)

    targets = plan.get('target_layers', [])
    constraints = plan.get('constraint_layers', [])

    # 1. 约束层
    roi_mask = None
    if constraints:
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        for c in constraints:
            cid = c.get('id')
            if cid is None: continue
            curr = mask_d if c.get('type') == 'defects' else mask_e
            roi_mask = cv2.bitwise_or(roi_mask, (curr == cid).astype(np.uint8))

        if np.sum(roi_mask) > 0:
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res_img, contours, -1, (255, 255, 255), 1)

    # 2. 目标层 + 面积计算
    found_info = []
    legend = []

    total_area = np.sum(roi_mask) if roi_mask is not None else (h * w)
    if total_area == 0: total_area = 1

    for item in targets:
        tid = item.get('id')
        if tid is None: continue

        correct_name = item.get('name', 'Unknown')
        ttype = item.get('type')

        if tid in DEFECT_MAP and (ttype == 'defects' or 'rust' in correct_name.lower()):
            ttype = 'defects'
            correct_name = DEFECT_MAP[tid]
        elif tid in ELEMENT_MAP:
            ttype = 'elements'
            correct_name = ELEMENT_MAP[tid]

        curr = mask_d if ttype == 'defects' else mask_e
        curr_mask = (curr == tid).astype(np.uint8)

        raw_pixels = np.sum(curr_mask)

        if roi_mask is not None:
            curr_mask = cv2.bitwise_and(curr_mask, roi_mask)

        pixel_count = np.sum(curr_mask)
        rgb = hrnet.colors_1[tid] if ttype == 'defects' else hrnet.colors[tid]

        ratio = (pixel_count / total_area) * 100

        if debug and raw_pixels > 0:
            status = f"✅ {ratio:.2f}%" if pixel_count > 0 else "❌ BLOCKED"
            st.sidebar.text(f"{correct_name}: {raw_pixels}px -> {status}")

        if pixel_count > 0:
            # 这里的文本只给 LLM 看，用户看不到，所以保留数字供 LLM 参考
            found_info.append(f"{correct_name}: {ratio:.2f}% coverage")
            canvas[curr_mask > 0] = rgb
            mask_bool = np.logical_or(mask_bool, curr_mask > 0)
            if correct_name not in [l[0] for l in legend]: legend.append((correct_name, rgb))

    if legend:
        mask_u8 = mask_bool.astype(np.uint8) * 255
        blended = cv2.addWeighted(res_img, 0.6, canvas, 0.4, 0)
        res_img[mask_bool] = blended[mask_bool]
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(res_img, contours, -1, (255, 255, 255), 2)

    stats = f"Detected: {', '.join(found_info)}" if found_info else "No targets detected."
    return res_img, stats, legend


def render_legend(legend):
    html = ""
    for name, rgb in legend:
        html += f"<span style='display:inline-block;width:12px;height:12px;background:rgb({rgb[0]},{rgb[1]},{rgb[2]});margin-right:5px;'></span>{name} "
    return html


# ==========================================
# 3. Frontend
# ==========================================
st.title("🌉 Bridge Inspection Dashboard")

with st.spinner("Loading Vision Model..."):
    hrnet, _ = load_model()
if not hrnet: st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    up_file = st.file_uploader("Image", type=["jpg", "png"], label_visibility="collapsed")
    if up_file:
        img = Image.open(up_file)
        st.session_state['img'] = img
        st.image(img, use_container_width=True)

with col2:
    if 'history' not in st.session_state: st.session_state['history'] = []

    chat_box = st.container(height=500)
    for msg in st.session_state['history']:
        with chat_box.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("img") is not None: st.image(msg["img"])
            if msg.get("log"):
                with st.expander("🛠️ Log"): st.text(msg["log"])

    if up_file and (query := st.chat_input("Ex: Give me an overview")):
        st.session_state['history'].append({"role": "user", "content": query})
        with chat_box.chat_message("user"):
            st.markdown(query)

        with chat_box.chat_message("assistant"):
            status = st.empty()

            # Step 1: Planner
            status.markdown("🧠 *Planner is Thinking...*")
            plan, log = ask_gemini_planner(query)

            with st.expander("🛠️ Planner Log"):
                st.text(log)

            if plan['intent'] == 'chat':
                reply = plan.get('reply', 'Hello!')
                status.markdown(reply)
                st.session_state['history'].append({"role": "assistant", "content": reply, "log": log})
            else:
                # Step 2: Vision
                status.markdown("👁️ *Vision System Scanning...*")
                res_img, stats, legend = process_vision_smart(hrnet, st.session_state['img'], plan, debug=debug_mode)

                # Step 3: Reasoning
                status.markdown("📝 *Reasoning Engine Analyzing...*")
                reply = generate_reasoning_response(query, stats, st.session_state['img'], plan['intent'])

                status.markdown(reply)

                # Display Logic
                show_img = (plan['intent'] == 'visualize')
                if show_img:
                    st.image(res_img)
                    if legend: st.markdown(render_legend(legend), unsafe_allow_html=True)

                st.session_state['history'].append({
                    "role": "assistant",
                    "content": reply,
                    "img": res_img if show_img else None,
                    "log": log
                })