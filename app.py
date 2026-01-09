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
# 0. SYSTEM CONFIGURATION (全 Gemini 版)
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("❌ Google API Key missing! Set it in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 1. Page Setup
# ==========================================
st.set_page_config(page_title="Bridge AI (All-Gemini)", page_icon="🌉", layout="wide")

with st.sidebar:
    st.header("⚙️ Architecture")
    st.success("🧠 Planner: Gemini 1.5 Flash")
    st.success("🕵️ Expert: Gemini 1.5 Flash")
    st.info("👁️ Vision: AECIF-Net (CPU)")
    st.divider()
    st.caption("v6.0 - Smarter, Simpler, No Groq")


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


def ask_gemini_plan(query):
    """
    Step 1: 使用 Gemini 进行规划 (它比 Llama 聪明得多，能听懂 'All', 'And', 'On')
    """
    # 使用 Flash 模型做规划，速度快且逻辑够用
    model = genai.GenerativeModel('gemini-1.5-flash')

    system_prompt = """
    You are the orchestration brain of a Bridge Inspection System.
    Your goal is to translate user queries into executable JSON instructions for a CNN.

    **Capabilities:**
    - Elements: 1:Bearing, 2:Bracing, 3:Deck, 4:Floor Beam, 5:Girder, 6:Pier.
    - Defects: 1:Rust.

    **Logic Rules:**
    1. **"visualize"**: User wants to SEE/LOCATE parts.
       - "Show all elements" -> List ALL IDs [1,2,3,4,5,6] in target_layers.
       - "Show rust on bearing" -> target: Rust(1), constraint: Bearing(1).
       - "Show rust on bearing AND deck" -> target: Rust(1), constraint: [Bearing(1), Deck(3)]. (Gemini can handle logic lists!)
    2. **"detect_defects"**: User wants ASSESSMENT/REPORT.
       - "Overview", "Summary", "Check defects" -> intent: detect_defects.
       - ALWAYS include Rust(1) in targets for defect checks unless specified otherwise.
    3. **"scan"**: Inventory check.

    **Output JSON ONLY.** Schema:
    {
      "intent": "visualize" | "detect_defects" | "chat",
      "reply": "Only for chat",
      "target_layers": [{"type": "elements"|"defects", "id": int, "name": str}],
      "constraint_layers": [{"type": "elements"|"defects", "id": int}] (List is allowed now!)
    }
    """

    try:
        response = model.generate_content(system_prompt + f"\nUser Query: {query}")
        content = response.text.replace("```json", "").replace("```", "").strip()
        s = content.find('{');
        e = content.rfind('}')
        if s != -1 and e != -1:
            return json.loads(content[s:e + 1]), f"**Gemini Plan:**\n{content}"
        else:
            raise ValueError("No JSON")
    except Exception as e:
        # 极简兜底：如果 Gemini 真的挂了（极少发生），默认为全量检测
        return {
            "intent": "detect_defects",
            "target_layers": [{"type": "defects", "id": 1, "name": "Rust"}],
            "constraint_layers": []
        }, f"⚠️ Plan Error: {e}"


def generate_expert_response(query, stats, image, intent):
    """Step 3: Gemini 生成最终报告"""
    model = genai.GenerativeModel('gemini-1.5-flash')

    if intent == 'visualize':
        prompt = f"User asked: '{query}'. CNN found: {stats}. Briefly confirm location (1 sentence). No full report."
    else:
        prompt = f"User asked: '{query}'. CNN found: {stats}. Act as a Bridge Inspector. 1. Sensor Data, 2. Visual Analysis (look for cracks/spalling), 3. Conclusion."

    try:
        res = model.generate_content([prompt, image])
        return res.text
    except Exception as e:
        return f"Expert Error: {e}"


def process_vision_smart(hrnet, image_pil, plan):
    if plan.get('intent') == 'chat': return None, "", []

    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    mask_e, mask_d = hrnet.get_raw_masks(image_pil)

    res_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    canvas = np.zeros_like(res_img)
    mask_bool = np.zeros((h, w), dtype=bool)

    targets = plan.get('target_layers', [])
    constraints = plan.get('constraint_layers', [])  # 现在支持列表了！

    # 1. 计算约束层 (Constraint Mask)
    # 逻辑：约束层之间取“并集” (Bearing OR Deck)，然后与目标层取“交集”
    roi_mask = None
    if constraints:
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        for c in constraints:
            cid = c.get('id')
            if cid is None: continue
            if c.get('type') == 'defects':
                roi_mask = cv2.bitwise_or(roi_mask, (mask_d == cid).astype(np.uint8))
            else:  # elements
                roi_mask = cv2.bitwise_or(roi_mask, (mask_e == cid).astype(np.uint8))

        # 画出约束层的白色轮廓
        if np.sum(roi_mask) > 0:
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res_img, contours, -1, (255, 255, 255), 1)

    # 2. 计算目标层 (Target Mask)
    found = []
    legend = []

    for item in targets:
        tid = item.get('id')
        if tid is None: continue

        # 自动纠正 Type (Gemini 偶尔也会忘，保险起见)
        ttype = item.get('type')
        if not ttype:
            ttype = 'defects' if item.get('name', '').lower() in ['rust', 'corrosion'] else 'elements'

        if ttype == 'elements':
            curr_mask = (mask_e == tid).astype(np.uint8)
            rgb = hrnet.colors[tid] if tid < len(hrnet.colors) else (200, 200, 200)
        else:
            curr_mask = (mask_d == tid).astype(np.uint8)
            rgb = hrnet.colors_1[tid] if tid < len(hrnet.colors_1) else (0, 0, 255)

        # 🌟 核心：Target AND Constraint
        if roi_mask is not None:
            curr_mask = cv2.bitwise_and(curr_mask, roi_mask)

        if np.sum(curr_mask) > 0:
            found.append(item['name'])
            canvas[curr_mask > 0] = rgb
            mask_bool = np.logical_or(mask_bool, curr_mask > 0)
            if item['name'] not in [l[0] for l in legend]: legend.append((item['name'], rgb))

    # 渲染
    if found:
        mask_u8 = mask_bool.astype(np.uint8) * 255
        blended = cv2.addWeighted(res_img, 0.6, canvas, 0.4, 0)
        res_img[mask_bool] = blended[mask_bool]
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(res_img, contours, -1, (255, 255, 255), 2)

    stats = f"Found: {', '.join(found)}" if found else "No targets found."
    return res_img, stats, legend


def render_legend(legend):
    html = ""
    for name, rgb in legend:
        html += f"<span style='display:inline-block;width:12px;height:12px;background:rgb({rgb[0]},{rgb[1]},{rgb[2]});margin-right:5px;'></span>{name} "
    return html


# ==========================================
# 3. Frontend
# ==========================================
st.title("🌉 Bridge AI (All-Gemini)")

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
            if msg.get("img"): st.image(msg["img"])
            if msg.get("log"):
                with st.expander("🧠 Thought Process"): st.code(msg["log"], language="json")

    if up_file and (query := st.chat_input("Ex: Segment all elements")):
        st.session_state['history'].append({"role": "user", "content": query})
        with chat_box.chat_message("user"):
            st.markdown(query)

        with chat_box.chat_message("assistant"):
            status = st.empty()

            # Step 1: Gemini Plan
            status.markdown("🧠 *Gemini is planning...*")
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

                # Display Image Logic
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