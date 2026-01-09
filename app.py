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
    st.success("🧠 Planner: Gemini Auto-Switch")
    st.success("🛠️ Refiner: Self-Correction")
    st.info("👁️ Vision: AECIF-Net")

    st.divider()
    debug_mode = st.checkbox("🔬 Diagnostic Mode", value=True)
    st.caption("v9.1 - User Experience Optimized")

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
    q = query.lower()

    # 规则 1：Overview 必须查 Rust (用于给 Expert 提供数据，但 Intent 保持 detect_defects)
    if any(k in q for k in ["overview", "defect", "summary", "check", "condition"]):
        if plan['intent'] != 'detect_defects':
            plan['intent'] = 'detect_defects'

        targets = plan.get('target_layers', [])
        if not targets: targets = []
        if not any(t.get('name') == 'Rust' for t in targets):
            targets.append({"type": "defects", "id": 1, "name": "Rust"})
        plan['target_layers'] = targets

    # 规则 2：All elements / Show me -> Visualize
    # 只要用户用了 "Show", "Segment", "Visual" 这种词，或者 "All elements"，就是 visualize
    if any(k in q for k in ["show", "see", "visual", "segment", "highlight", "draw"]):
        plan['intent'] = 'visualize'

    # 规则 3：All elements 必须清空约束并列出所有
    has_all = "all" in q or "every" in q or "whole" in q
    has_part = "element" in q or "part" in q or "component" in q

    if has_all and has_part:
        plan['intent'] = 'visualize'
        plan['target_layers'] = [{"type": "elements", "id": i, "name": name} for i, name in ELEMENT_MAP.items()]
        plan['constraint_layers'] = []

    return plan


def ask_gemini_plan_with_retry(query):
    models = get_best_model()

    base_prompt = """
    Role: Bridge Inspection Orchestrator.
    Task: Convert user query to JSON.

    Logic:
    1. "visualize": User wants to SEE/LOCATE/HIGHLIGHT. (Keywords: Show, Segment, Where is)
    2. "detect_defects": User wants REPORT/TEXT ONLY. (Keywords: Overview, Summary, Report)

    Output JSON Schema:
    {
      "intent": "visualize" | "detect_defects" | "chat",
      "reply": "str",
      "target_layers": [{"type": "elements"|"defects", "id": int, "name": "str"}],
      "constraint_layers": [{"type": "elements"|"defects", "id": int}]
    }
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
                log_buffer += f"✅ Model {model_name} succeeded.\n"
                return final_plan, log_buffer
            except Exception as parse_error:
                log_buffer += f"⚠️ {model_name} format error. Retrying...\n"
                # Self-correction logic omitted for brevity, swapping to next model usually faster
                continue
        except Exception as e:
            continue

    return keyword_rescue(query, log_buffer)


def keyword_rescue(query, previous_logs):
    q = query.lower()
    log = previous_logs + "\n💀 AI failed. Using Keyword Rescue."

    # 1. Visualize Keywords
    if any(k in q for k in ["show", "see", "segment", "highlight", "draw"]):
        # Check for specific elements
        for eid, name in ELEMENT_MAP.items():
            if name.lower() in q:
                return {"intent": "visualize", "target_layers": [{"type": "elements", "id": eid, "name": name}],
                        "constraint_layers": []}, log
        # Default visualize all if "all" is present
        if "all" in q:
            return {"intent": "visualize",
                    "target_layers": [{"type": "elements", "id": i, "name": n} for i, n in ELEMENT_MAP.items()],
                    "constraint_layers": []}, log

    # 2. Defects/Overview
    return {
        "intent": "detect_defects",
        "target_layers": [{"type": "defects", "id": 1, "name": "Rust"}],
        "constraint_layers": []
    }, log


def generate_expert_response(query, stats, image, intent):
    """Step 3: Expert (根据意图区分 Prompt)"""
    models = get_best_model()

    if intent == 'visualize':
        # 🌟 模式 A：看图模式 (极简)
        prompt = f"""
        User Query: "{query}"
        CNN Findings: {stats}

        [TASK]
        The user wants to see the visualization. 
        1. Confirm what is highlighted in the image.
        2. Be extremely BRIEF (Max 2 sentences). 
        3. DO NOT describe the bridge structure or history. 
        4. DO NOT reference "crops" or "patches".

        Example Output: "I have highlighted all detected bridge elements, including the Girder, Pier, and Deck."
        """
    else:
        # 🌟 模式 B：报告模式 (详细)
        prompt = f"""
        User Query: "{query}"
        [Sensor Data]: {stats}

        [TASK]
        Act as a Senior Bridge Inspector. Provide a comprehensive overview.
        1. Direct Answer to the user.
        2. Integrate visual observations with sensor data naturally.
        3. ⛔️ CRITICAL: Do NOT mention "crops", "patches", or specific image file names. Treat the image as one whole scene.
        """

    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            res = model.generate_content([prompt, image])
            return res.text
        except:
            continue
    return "Expert Error: All models failed."


def process_vision_smart(hrnet, image_pil, plan, debug=False):
    if plan.get('intent') == 'chat': return None, "", []

    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    mask_e, mask_d = hrnet.get_raw_masks(image_pil)

    if debug:
        unique_e = np.unique(mask_e)
        st.sidebar.warning(f"🔎 Raw IDs: {unique_e}")

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

    # 2. 目标层
    found = []
    legend = []

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

        if roi_mask is not None:
            curr_mask = cv2.bitwise_and(curr_mask, roi_mask)

        pixel_count = np.sum(curr_mask)
        rgb = hrnet.colors_1[tid] if ttype == 'defects' else hrnet.colors[tid]

        if pixel_count > 0:
            found.append(correct_name)
            canvas[curr_mask > 0] = rgb
            mask_bool = np.logical_or(mask_bool, curr_mask > 0)
            if correct_name not in [l[0] for l in legend]: legend.append((correct_name, rgb))

    if found:
        mask_u8 = mask_bool.astype(np.uint8) * 255
        blended = cv2.addWeighted(res_img, 0.6, canvas, 0.4, 0)
        res_img[mask_bool] = blended[mask_bool]
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(res_img, contours, -1, (255, 255, 255), 2)

    stats = f"Detected: {', '.join(found)}" if found else "No targets found."
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
                with st.expander("🛠️ Correction Log"): st.text(msg["log"])

    if up_file and (query := st.chat_input("Ex: Give me an overview")):
        st.session_state['history'].append({"role": "user", "content": query})
        with chat_box.chat_message("user"):
            st.markdown(query)

        with chat_box.chat_message("assistant"):
            status = st.empty()

            # Step 1: Plan
            status.markdown("🧠 *Planning...*")
            plan, log = ask_gemini_plan_with_retry(query)

            with st.expander("🛠️ Correction Log"):
                st.text(log)

            if plan['intent'] == 'chat':
                reply = plan.get('reply', 'Hello!')
                status.markdown(reply)
                st.session_state['history'].append({"role": "assistant", "content": reply, "log": log})
            else:
                # Step 2: Vision
                status.markdown("👁️ *Scanning...*")
                res_img, stats, legend = process_vision_smart(hrnet, st.session_state['img'], plan, debug=debug_mode)

                # Step 3: Expert
                status.markdown("📝 *Writing Report...*")
                reply = generate_expert_response(query, stats, st.session_state['img'], plan['intent'])

                status.markdown(reply)

                # 🌟 核心修改 1：严格控制图片显示
                # 只有当 intent 是 visualize 时才显示图片。
                # 即使 detect_defects 在后台调用了 CNN 查锈，前台也不显示。
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