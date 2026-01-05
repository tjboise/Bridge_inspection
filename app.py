import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import json
import os
import io
import time
import google.generativeai as genai
from openai import OpenAI
from AECIF_Net import HRnet_Segmentation

# ==========================================
# 0. SYSTEM CONFIGURATION (部署修改版)
# ==========================================
# 🛑 这里的 Key 必须从云端 Secrets 读取，不能写死在代码里
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    # 如果本地运行没有 .streamlit/secrets.toml，或者云端没配好，会报错提示
    st.error("❌ API Keys not found! Please configure Streamlit Secrets.")
    st.info("💡 On Streamlit Cloud: Go to App Settings -> Secrets")
    st.info("💡 On Local Machine: Create .streamlit/secrets.toml")
    st.stop()

# 配置 Google API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 1. Page Setup
# ==========================================
st.set_page_config(
    page_title="Bridge Inspection Dashboard",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Sidebar
with st.sidebar:
    st.header("⚙️ System Status")
    st.info("🧠 Vision: Gemini 2.0 Flash")
    st.info("⚡ Planning: Llama 3 (Groq)")
    st.info("👁️ Segmentation: AECIF-Net")

    if not GROQ_API_KEY: st.error("⚠️ Groq Key Missing")
    if not GOOGLE_API_KEY: st.error("⚠️ Google Key Missing")


# ==========================================
# 2. Backend Logic
# ==========================================

@st.cache_resource
def load_model():
    weight_path = 'model_data/best_epoch_weights.pth'
    if not os.path.exists(weight_path): return None, "Weight file missing"

    try:
        if torch.cuda.is_available():
            dummy = torch.tensor([1.0]).cuda()
            res = dummy + 1.0
            model = HRnet_Segmentation(model_path=weight_path, cuda=True)
            return model, "GPU"
    except:
        pass
    model = HRnet_Segmentation(model_path=weight_path, cuda=False)
    return model, "CPU"


def ask_llm_plan(query):
    """Step 1: 使用 Groq (Llama3) 进行快速规划"""
    if not GROQ_API_KEY: return None, "Groq API Key is invalid."

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    system_prompt = """
    You are a Bridge Inspection Agent. 

    Intents Definition:
    1. "visualize": The user EXPLICITLY asks to "See", "Show", "Highlight", "Mark", "Display", or "Draw".
    2. "detect_defects": The user asks "What", "How many", "Analyze", "Assess", "Report", "Is there any...".
    3. "scan": User asks to list or find elements.
    4. "chat": Greetings.

    Output JSON ONLY. 
    JSON Schema:
    {"intent": "visualize"|"chat"|"detect_defects"|"scan", "reply": "...", "target_layers": [{"type": "elements"|"defects", "id": int, "name": str}], "constraint_layer": {"type": "elements", "id": int} or null}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
            temperature=0.1
        )
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content), None
    except Exception as e:
        return None, f"LLM Error: {str(e)}"


def generate_hybrid_expert_summary(user_query, visual_stats, image_pil):
    """Step 3: Google Gemini 通用视觉检测"""
    if not GOOGLE_API_KEY:
        return "⚠️ Google API Key missing."

    candidate_models = [
        'gemini-2.0-flash',
        'gemini-2.5-flash',
        'gemini-1.5-flash',
        'gemini-1.5-flash-001'
    ]

    prompt = f"""
    You are a Senior Bridge Inspector.

    【Data Source 1: CNN Sensors】
    {visual_stats}
    (CNN ONLY detects Rust. It is blind to cracks.)

    【Data Source 2: Your Vision】
    Look at the image. Identify **ANY** structural anomalies (Cracks, Spalling, Exposed Rebar, etc.).

    【Task】
    Answer: "{user_query}"
    Structure:
    1. **Sensor Readings**: Quote Rust data.
    2. **Visual Observations**: Describe what YOU see.
    3. **Conclusion**: Assess condition.
    """

    last_error = ""
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, image_pil])
            return response.text
        except Exception as e:
            last_error = str(e)
            continue

    return f"All Google models failed. Last error: {last_error}"


def process_vision_colorful(hrnet, image_pil, plan):
    if plan.get('intent') == 'chat':
        return None, plan.get('reply', ''), [], None

    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    mask_e_idx, mask_d_idx = hrnet.get_raw_masks(image_pil)

    res_img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # 🌟 Single Canvas 渲染 (高清无叠加)
    color_canvas = np.zeros_like(res_img_rgb)
    combined_mask_bool = np.zeros((h, w), dtype=bool)

    targets_to_check = []
    intent = plan.get('intent', 'scan')

    if intent == 'scan':
        all_elements = [(1, "Bearing"), (2, "Bracing"), (3, "Deck"), (4, "Floor Beam"), (5, "Girder"), (6, "Pier")]
        for eid, name in all_elements:
            targets_to_check.append({"type": "elements", "id": eid, "name": name})
    elif intent == 'detect_defects':
        targets_to_check.append({"type": "defects", "id": 1, "name": "Rust"})
    else:
        targets_to_check = plan.get('target_layers', [])

    # 鲁棒性检查 Constraint Layer
    roi_mask = None
    c_layer = plan.get('constraint_layer')

    if c_layer and isinstance(c_layer, dict) and 'id' in c_layer:
        c_id = c_layer.get('id')
        if c_id is not None:
            if c_layer.get('type') == 'elements':
                roi_mask = (mask_e_idx == c_id).astype(np.uint8)
            else:
                roi_mask = (mask_d_idx == c_id).astype(np.uint8)

    found_items = []
    legend_data = []
    stats_info = []

    for item in targets_to_check:
        if 'id' not in item: continue

        if item['type'] == 'elements':
            current_mask = (mask_e_idx == item['id']).astype(np.uint8)
            color_rgb = hrnet.colors[item['id']] if item['id'] < len(hrnet.colors) else (255, 0, 0)
        elif item['type'] == 'defects':
            current_mask = (mask_d_idx == item['id']).astype(np.uint8)
            color_rgb = hrnet.colors_1[item['id']] if item['id'] < len(hrnet.colors_1) else (0, 0, 255)

        if roi_mask is not None:
            current_mask = cv2.bitwise_and(current_mask, roi_mask)

        pixel_count = np.sum(current_mask)
        if pixel_count > 0:
            found_items.append(item['name'])
            # 🌟 直接填色，不叠加
            color_canvas[current_mask > 0] = color_rgb
            combined_mask_bool = np.logical_or(combined_mask_bool, current_mask > 0)

            if item['name'] not in [l[0] for l in legend_data]:
                legend_data.append((item['name'], color_rgb))

            total_pixels = np.sum(roi_mask) if roi_mask is not None else (h * w)
            ratio = (pixel_count / total_pixels) * 100
            stats_info.append(f"{item['name']} (Area: {pixel_count}px, Coverage: {ratio:.2f}%)")

    if not found_items:
        findings_str = f"CNN Detectors scanned for Rust but found NOTHING (0% coverage)."
    else:
        findings_str = f"CNN Detectors found: {'; '.join(stats_info)}."

    # 🌟 最后一次性混合渲染
    if found_items:
        mask_uint8 = combined_mask_bool.astype(np.uint8) * 255
        blended = cv2.addWeighted(res_img_rgb, 0.6, color_canvas, 0.4, 0)
        res_img_rgb[combined_mask_bool] = blended[combined_mask_bool]

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(res_img_rgb, contours, -1, (255, 255, 0), 2)
        if roi_mask is not None:
            contours_roi, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(res_img_rgb, contours_roi, -1, (0, 0, 255), 1)

    return res_img_rgb, findings_str, legend_data


def render_legend_html(legend_data):
    if not legend_data: return ""
    html_str = "#### 🎨 Legend (CNN)<br>"
    for name, rgb in legend_data:
        color_css = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        html_str += f"<span style='display:inline-block; width:12px; height:12px; background-color:{color_css}; margin-right:5px; border-radius:2px;'></span>**{name}** &nbsp;&nbsp; "
    return html_str


# ==========================================
# 3. Frontend Layout
# ==========================================
st.title("🌉 Bridge Inspection Dashboard")
st.markdown("Ask: *'What defects are here?'* (Text) OR *'Show me the rust'* (Visual)")

with st.spinner("Booting up..."):
    hrnet, dev_status = load_model()

if not hrnet: st.stop()

col_img, col_chat = st.columns([1, 2])

with col_img:
    st.subheader("1. Image")
    uploaded_file = st.file_uploader("Upload", type=["jpg", "png"], label_visibility="collapsed")
    if uploaded_file:
        if 'curr_image_name' not in st.session_state or st.session_state['curr_image_name'] != uploaded_file.name:
            st.session_state['curr_image'] = Image.open(uploaded_file)
            st.session_state['curr_image_name'] = uploaded_file.name
            st.session_state['chat_history'] = []
        st.image(st.session_state['curr_image'], use_container_width=True)
    else:
        st.info("Upload image...")

with col_chat:
    st.subheader("2. Analysis")
    if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []

    chat_container = st.container(height=600)
    with chat_container:
        for msg in st.session_state['chat_history']:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("image") is not None: st.image(msg["image"])
                if msg.get("legend"): st.markdown(msg["legend"], unsafe_allow_html=True)

    if uploaded_file and (user_query := st.chat_input("Ask about the bridge...")):
        st.session_state['chat_history'].append({"role": "user", "content": user_query})
        with chat_container:
            with st.chat_message("user"): st.markdown(user_query)

        with chat_container:
            with st.chat_message("assistant"):
                ph = st.empty()

                # 1. 规划 (Groq)
                ph.markdown("🧠 *Step 1: Planning (Llama3)...*")
                plan, err = ask_llm_plan(user_query)

                if err or not plan:
                    ph.error(f"Planning Error: {err}")
                elif plan.get('intent') == 'chat':
                    reply = plan.get('reply', 'Hello')
                    ph.markdown(reply)
                    st.session_state['chat_history'].append({"role": "assistant", "content": reply})
                else:
                    # 2. CNN (后台计算)
                    ph.markdown("👁️ *Step 2: CNN Analyzing (Background)...*")
                    res_img, cnn_stats, legend_data = process_vision_colorful(hrnet, st.session_state['curr_image'],
                                                                              plan)
                    legend_html = render_legend_html(legend_data)

                    # 3. Gemini (Google) - 报告
                    ph.markdown("🕵️ *Step 3: Gemini Generating Report...*")
                    final_report = generate_hybrid_expert_summary(user_query, cnn_stats, st.session_state['curr_image'])

                    ph.markdown(final_report)

                    # 只有 intent 是 visualize 才显示图片
                    should_show_image = (plan.get('intent') == 'visualize')

                    img_to_save = None
                    if should_show_image:
                        st.image(res_img)
                        if legend_html: st.markdown(legend_html, unsafe_allow_html=True)
                        img_to_save = res_img

                    st.session_state['chat_history'].append({
                        "role": "assistant",
                        "content": final_report,
                        "image": img_to_save,
                        "legend": legend_html if should_show_image else None
                    })