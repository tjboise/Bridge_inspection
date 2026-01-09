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
# 1. Page Setup (老大，必须排第一)
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
    st.caption("v6.5 - Natural Report Style")


# ==========================================
# 2. Backend Logic
# ==========================================

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
    🚑 规则修正器：防止 LLM 漏检关键病害
    """
    q = query.lower()

    # 规则 1：如果问 "Overview" 或 "Defects"，必须查 Rust
    keywords = ["overview", "defect", "summary", "problem", "condition", "check"]
    if any(k in q for k in keywords):
        if plan['intent'] != 'detect_defects':
            plan['intent'] = 'detect_defects'

        targets = plan.get('target_layers', [])
        has_rust = any(t.get('name') == 'Rust' for t in targets)
        if not has_rust:
            targets.append({"type": "defects", "id": 1, "name": "Rust"})
            plan['target_layers'] = targets

    # 规则 2：如果问 "All elements"，必须展开列表
    if "all element" in q or "everything" in q:
        plan['intent'] = 'visualize'
        plan['target_layers'] = [
            {"type": "elements", "id": 1, "name": "Bearing"},
            {"type": "elements", "id": 2, "name": "Bracing"},
            {"type": "elements", "id": 3, "name": "Deck"},
            {"type": "elements", "id": 4, "name": "Floor Beam"},
            {"type": "elements", "id": 5, "name": "Girder"},
            {"type": "elements", "id": 6, "name": "Pier"}
        ]

    return plan


def ask_gemini_plan(query):
    """Step 1: 使用 Gemini 2.5 进行规划"""
    model_candidates = get_best_model()

    system_prompt = """
    You are the orchestration brain. Translate user queries into JSON.

    **Logic:**
    1. "visualize": User wants to SEE/LOCATE.
    2. "detect_defects": User wants REPORT/ASSESSMENT.
       - "Overview", "Summary" -> intent: detect_defects.
       - ALWAYS include Rust (ID 1) in targets.

    **Output JSON ONLY.**
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
                # 🌟 调用修正器
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
        # 🌟 核心修改：让回答更自然，不要强制分段
        prompt = f"""
        User Query: "{query}"

        [Sensor Data Reference]: {stats}
        (Note: If CNN found nothing, it might be a false negative. Use your vision.)

        [Task]: Act as a Senior Bridge Inspector.
        - **Direct Answer**: Address the user's specific question directly.
        - **Holistic View**: Provide a comprehensive visual assessment of the bridge condition (check for rust, cracks, spalling, etc.).
        - **Data Integration**: Incorporate the "Sensor Data" NATURALLY into your narrative to support your observations. 
          - ❌ DO NOT create a separate "Sensor Data" section unless specifically asked for measurements.
          - ✅ Example: "Visual inspection reveals significant corrosion on the girder, which is confirmed by sensors detecting 5% rust coverage."
        """

    for model_name in model_candidates:
        try:
            model = genai.GenerativeModel(model_name)
            res = model.generate_content([prompt, image])
            return res.text
        except:
            continue
    return "Expert Error: All models failed."


def process_vision_smart(hrnet, image_pil, plan):
    if plan.get('intent') == 'chat': return None, "", []

    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    mask_e, mask_d = hrnet.get_raw_masks(image_pil)

    res_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    canvas = np.zeros_like(res_img)
    mask_bool = np.zeros((h, w), dtype=bool)

    targets = plan.get('target_layers', [])
    constraints = plan.get('constraint_layers', [])

    # 1. 约束层处理
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

        if np.sum(roi_mask) > 0:
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res_img, contours, -1, (255, 255, 255), 1)

    # 2. 目标层处理
    found = []
    legend = []
    target_names_checked = []

    for item in targets:
        tid = item.get('id')
        name = item.get('name', 'Unknown')
        if tid is None: continue

        target_names_checked.append(name)

        ttype = item.get('type')
        if not ttype:
            ttype = 'defects' if name.lower() in ['rust', 'corrosion'] else 'elements'

        if ttype == 'elements':
            curr_mask = (mask_e == tid).astype(np.uint8)
            rgb = hrnet.colors[tid] if tid < len(hrnet.colors) else (200, 200, 200)
        else:
            curr_mask = (mask_d == tid).astype(np.uint8)
            rgb = hrnet.colors_1[tid] if tid < len(hrnet.colors_1) else (0, 0, 255)

        if roi_mask is not None:
            curr_mask = cv2.bitwise_and(curr_mask, roi_mask)

        pixel_count = np.sum(curr_mask)
        if pixel_count > 0:
            found.append(f"{name}")
            canvas[curr_mask > 0] = rgb
            mask_bool = np.logical_or(mask_bool, curr_mask > 0)
            if name not in [l[0] for l in legend]: legend.append((name, rgb))

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

    if up_file and (query := st.chat_input("Ex: Give me an overview")):
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
                res_img, stats, legend = process_vision_smart(hrnet, st.session_state['img'], plan)

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