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
    st.success("🧠 Planner: Gemini 2.5 Flash")
    st.success("🕵️ Expert: Gemini 2.5 Flash")
    st.info("👁️ Vision: AECIF-Net")

    st.divider()
    debug_mode = st.checkbox("🔬 Diagnostic Mode", value=True, help="See internal logs")
    st.caption("v7.1 - Fix Constraint Conflict")

# ==========================================
# 2. Backend Logic
# ==========================================

ELEMENT_MAP = {
    1: "Bearing",
    2: "Bracing",
    3: "Deck",
    4: "Floor Beam",
    5: "Girder",
    6: "Pier"
}

DEFECT_MAP = {
    1: "Rust"
}


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
        'gemini-3-flash',
        'gemini-2.5-flash-lite',
        'gemini-1.5-flash'
    ]


def refine_plan(query, plan):
    """
    🚑 规则修正器 v2
    """
    q = query.lower()

    # 规则 1：Overview 必须查 Rust
    keywords = ["overview", "defect", "summary", "problem", "condition", "check"]
    if any(k in q for k in keywords):
        if plan['intent'] != 'detect_defects':
            plan['intent'] = 'detect_defects'

        targets = plan.get('target_layers', [])
        has_rust = any(t.get('name') == 'Rust' for t in targets)
        if not has_rust:
            targets.append({"type": "defects", "id": 1, "name": "Rust"})
            plan['target_layers'] = targets

    # 规则 2：All elements 展开 + 🔥 强制清空约束
    # 既然用户想看“全部”，就绝对不能有 Constraint 限制！
    if "all element" in q or "everything" in q:
        plan['intent'] = 'visualize'
        plan['target_layers'] = [{"type": "elements", "id": i, "name": name} for i, name in ELEMENT_MAP.items()]
        plan['constraint_layers'] = []  # 👈 关键修复：把“紧箍咒”摘掉！

    return plan


def ask_gemini_plan(query):
    """Step 1: 使用 Gemini 2.5 进行规划"""
    model_candidates = get_best_model()

    system_prompt = """
    Translate user queries into JSON.
    1. "visualize": User wants to SEE/LOCATE.
    2. "detect_defects": User wants REPORT. ALWAYS include Rust (ID 1).
    Output JSON ONLY.
    Schema: {
      "intent": "visualize" | "detect_defects" | "chat",
      "reply": "Only for chat",
      "target_layers": [{"type": "elements"|"defects", "id": int, "name": str}],
      "constraint_layers": [{"type": "elements"|"defects", "id": int}]
    }
    """

    for model_name in model_candidates:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(system_prompt + f"\nUser Query: {query}")
            content = response.text.replace("```json", "").replace("```", "").strip()
            s = content.find('{');
            e = content.rfind('}')
            if s != -1 and e != -1:
                raw_plan = json.loads(content[s:e + 1])
                final_plan = refine_plan(query, raw_plan)
                return final_plan, f"**Gemini Plan:**\n{json.dumps(final_plan, indent=2)}"
        except:
            continue

    fallback = {"intent": "detect_defects", "target_layers": [{"type": "defects", "id": 1, "name": "Rust"}]}
    return fallback, "⚠️ Plan Error, defaulting to Rust check."


def generate_expert_response(query, stats, image, intent):
    """Step 3: Gemini 2.5 生成最终报告"""
    model_candidates = get_best_model()

    if intent == 'visualize':
        prompt = f"User asked: '{query}'. CNN found: {stats}. Briefly confirm location (1 sentence). No full report."
    else:
        prompt = f"""
        User Query: "{query}"
        [Sensor Data Reference]: {stats}
        [Task]: Act as a Senior Bridge Inspector.
        - Direct Answer.
        - Holistic View.
        - Integrate Sensor Data NATURALLY.
        """

    for model_name in model_candidates:
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

    # 🔬 详细诊断日志
    debug_info = []
    if debug:
        unique_e = np.unique(mask_e)
        st.sidebar.warning(f"🔎 Raw IDs in Elements Mask: {unique_e}")
        debug_info.append(f"Map raw: {unique_e}")

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
            if c.get('type') == 'defects':
                roi_mask = cv2.bitwise_or(roi_mask, (mask_d == cid).astype(np.uint8))
            else:
                roi_mask = cv2.bitwise_or(roi_mask, (mask_e == cid).astype(np.uint8))

        # 调试：如果有约束，显示出来
        if debug and np.sum(roi_mask) > 0:
            st.sidebar.info(f"🔒 Constraint Mask Active! Area: {np.sum(roi_mask)} px")

        if np.sum(roi_mask) > 0:
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res_img, contours, -1, (255, 255, 255), 1)

    # 2. 目标层
    found = []
    legend = []
    target_names_checked = []

    for item in targets:
        tid = item.get('id')
        if tid is None: continue

        # 强制修正名称
        correct_name = "Unknown"
        ttype = item.get('type')

        if tid in DEFECT_MAP and (ttype == 'defects' or 'rust' in item.get('name', '').lower()):
            ttype = 'defects'
            correct_name = DEFECT_MAP[tid]
        elif tid in ELEMENT_MAP:
            ttype = 'elements'
            correct_name = ELEMENT_MAP[tid]

        target_names_checked.append(correct_name)

        if ttype == 'elements':
            curr_mask = (mask_e == tid).astype(np.uint8)
            rgb = hrnet.colors[tid] if tid < len(hrnet.colors) else (200, 200, 200)
        else:
            curr_mask = (mask_d == tid).astype(np.uint8)
            rgb = hrnet.colors_1[tid] if tid < len(hrnet.colors_1) else (0, 0, 255)

        # 🌟 核心调试：记录遮罩前的像素数
        raw_pixels = np.sum(curr_mask)

        if roi_mask is not None:
            curr_mask = cv2.bitwise_and(curr_mask, roi_mask)

        final_pixels = np.sum(curr_mask)

        if debug and raw_pixels > 0:
            status = "✅ Kept" if final_pixels > 0 else "❌ Filtered by Constraint"
            st.sidebar.text(f"Checking {correct_name} (ID {tid}): {raw_pixels}px -> {status}")

        if final_pixels > 0:
            found.append(f"{correct_name}")
            canvas[curr_mask > 0] = rgb
            mask_bool = np.logical_or(mask_bool, curr_mask > 0)
            if correct_name not in [l[0] for l in legend]: legend.append((correct_name, rgb))

    if found:
        mask_u8 = mask_bool.astype(np.uint8) * 255
        blended = cv2.addWeighted(res_img, 0.6, canvas, 0.4, 0)
        res_img[mask_bool] = blended[mask_bool]
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(res_img, contours, -1, (255, 255, 255), 2)

    if found:
        stats = f"CNN Detected: {', '.join(found)}"
    else:
        if target_names_checked:
            stats = f"CNN scanned for [{', '.join(target_names_checked)}] but found 0 pixels."
        else:
            stats = "CNN: No specific targets were requested."

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
            if msg.get("img") is not None:
                st.image(msg["img"])
            if msg.get("log"):
                with st.expander("🧠 Thought Process"): st.code(msg["log"], language="json")

    if up_file and (query := st.chat_input("Ex: Show me all elements")):
        st.session_state['history'].append({"role": "user", "content": query})
        with chat_box.chat_message("user"):
            st.markdown(query)

        with chat_box.chat_message("assistant"):
            status = st.empty()

            # Step 1: Gemini Plan
            status.markdown("🧠 *Planning...*")
            plan, log = ask_gemini_plan(query)

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

                show_img = (plan['intent'] == 'visualize' or len(legend) > 0)
                if show_img:
                    st.image(res_img)
                    if legend: st.markdown(render_legend(legend), unsafe_allow_html=True)

                st.session_state['history'].append({
                    "role": "assistant",
                    "content": reply,
                    "img": res_img if show_img else None,
                    "log": log + f"\n\nCNN Stats: {stats}"
                })