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
# 0. SYSTEM CONFIGURATION
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("❌ API Keys not found! Please configure Streamlit Secrets.")
    st.stop()

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

with st.sidebar:
    st.header("⚙️ System Status")
    st.info("🧠 Vision: Gemini 2.0 Flash")
    st.info("⚡ Planning: Llama 3 (Groq)")
    st.info("👁️ Segmentation: AECIF-Net")


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
            model = HRnet_Segmentation(model_path=weight_path, cuda=True)
            return model, "GPU"
    except:
        pass
    model = HRnet_Segmentation(model_path=weight_path, cuda=False)
    return model, "CPU"


def ask_llm_plan(query):
    """
    Step 1: Llama 3 (Groq) - 意图识别与任务规划
    User wants to optimize prompt to be dynamic based on intent.
    """
    if not GROQ_API_KEY: return None, "Groq API Key is invalid."

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    # 🌟 核心升级：System Prompt 增加 response_mode 逻辑
    system_prompt = """
    You are the "Brain" of a Bridge Inspection System. You coordinate three modules:
    1. CNN (Vision): Can segment elements (Bearing, Deck, Pier, etc.) and Rust.
    2. Gemini (Expert): Can analyze images and answer complex questions.
    3. User (Human): Asks questions.

    Your Task: Analyze the user's input and output a JSON plan.

    **Response Modes (`response_mode`):**
    - "concise": User just wants to SEE something (e.g., "Show me the pier", "Segment all elements"). 
      -> Instruction: Do NOT generate a long report. Just show the image and a 1-sentence caption.
    - "detailed": User asks for analysis, condition assessment, or defects (e.g., "Analyze the cracks", "What is the condition?", "Report").
      -> Instruction: Generate a full structural report with observations and conclusions.
    - "chat": Casual greeting.

    **JSON Schema:**
    {
      "intent": "visualize" | "detect_defects" | "chat",
      "response_mode": "concise" | "detailed" | "chat", 
      "reply": "Only for chat intent",
      "target_layers": [{"type": "elements"|"defects", "id": int, "name": str}], 
      "constraint_layer": {"type": "elements", "id": int} or null
    }

    **Element IDs:** 1:Bearing, 2:Bracing, 3:Deck, 4:Floor Beam, 5:Girder, 6:Pier.
    **Defect IDs:** 1:Rust.

    **Example:**
    User: "Show me all elements"
    Output: {"intent": "visualize", "response_mode": "concise", "target_layers": [], "constraint_layer": null}
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


def generate_hybrid_expert_summary(user_query, visual_stats, image_pil, response_mode="detailed"):
    """
    Step 3: Gemini (Google) - 根据 Llama 指定的模式生成回答
    """
    if not GOOGLE_API_KEY: return "⚠️ Google API Key missing."

    # 🌟 动态 Prompt：根据模式切换“性格”
    if response_mode == "concise":
        # 模式 A：简短说明（看图模式）
        prompt = f"""
        You are a Bridge Inspection Assistant.
        The user has requested a visual segmentation of the bridge.

        **Context:**
        - The user is looking at a segmented image where the following parts were detected by CNN: {visual_stats}

        **Task:**
        - Provide a VERY BRIEF (1-2 sentences) caption for the image.
        - Mention what main structural elements are visible.
        - Do NOT analyze defects or write a report unless explicitly asked.
        """
    else:
        # 模式 B：详细报告（分析模式）
        prompt = f"""
        You are a Senior Bridge Inspector writing a formal inspection report.

        **Data Source 1 (CNN Segmentation Results):**
        {visual_stats}
        *(Note: CNN only detects Rust/Elements. It cannot see Cracks - you must find cracks visually.)*

        **Data Source 2 (Visual Inspection):**
        Look closely at the image provided. Identify structural anomalies (Cracks, Spalling, Corrosion, Efflorescence).

        **Task:**
        Answer the user's request: "{user_query}"

        **Report Structure:**
        1. **CNN Findings**: Briefly summarize the rust/element coverage data above.
        2. **Visual Observations**: Describe defects you see (Cracks, Spalling, etc.) in detail.
        3. **Assessment**: Conclusion on the bridge's condition (Good/Fair/Poor).
        """

    model_name = 'gemini-2.0-flash'
    try:
        model = genai.GenerativeModel(model_name)
        # 稍微冷却一下防止 429
        time.sleep(1)
        response = model.generate_content([prompt, image_pil])
        return response.text
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"


def process_vision_colorful(hrnet, image_pil, plan):
    if plan.get('intent') == 'chat':
        return None, plan.get('reply', ''), [], None

    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    # 获取原始 Mask
    mask_e_idx, mask_d_idx = hrnet.get_raw_masks(image_pil)

    res_img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    color_canvas = np.zeros_like(res_img_rgb)
    combined_mask_bool = np.zeros((h, w), dtype=bool)

    targets_to_check = []
    intent = plan.get('intent', 'scan')

    # 🌟 逻辑修正：如果用户没指定特定部件，或者说 "segment all"，则强制加入所有部件
    target_layers = plan.get('target_layers', [])
    if not target_layers or intent == 'scan' or (intent == 'visualize' and len(target_layers) == 0):
        # 强制全量扫描
        all_elements = [
            (1, "Bearing"), (2, "Bracing"), (3, "Deck"),
            (4, "Floor Beam"), (5, "Girder"), (6, "Pier")
        ]
        for eid, name in all_elements:
            targets_to_check.append({"type": "elements", "id": eid, "name": name})
    else:
        # 否则只显示用户点的名
        targets_to_check = target_layers

    # 处理 "Analyze Defects" 时，加上 Rust Mask
    if intent == 'detect_defects':
        # 只有当列表里没有 rust 时才加，避免重复
        if not any(t.get('name') == 'Rust' for t in targets_to_check):
            targets_to_check.append({"type": "defects", "id": 1, "name": "Rust"})

    # 约束层逻辑
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

        # 确定 Mask 和 颜色
        if item['type'] == 'elements':
            current_mask = (mask_e_idx == item['id']).astype(np.uint8)
            # 颜色映射保护
            idx = item['id']
            if idx < len(hrnet.colors):
                color_rgb = hrnet.colors[idx]
            else:
                color_rgb = (128, 128, 128)  # 默认灰
        elif item['type'] == 'defects':
            current_mask = (mask_d_idx == item['id']).astype(np.uint8)
            if item['id'] < len(hrnet.colors_1):
                color_rgb = hrnet.colors_1[item['id']]
            else:
                color_rgb = (0, 0, 255)  # 默认红

        if roi_mask is not None:
            current_mask = cv2.bitwise_and(current_mask, roi_mask)

        pixel_count = np.sum(current_mask)

        # 🌟 只有像素点 > 0 才绘制，避免画空 mask
        if pixel_count > 0:
            found_items.append(item['name'])
            color_canvas[current_mask > 0] = color_rgb
            combined_mask_bool = np.logical_or(combined_mask_bool, current_mask > 0)

            # 添加图例
            if item['name'] not in [l[0] for l in legend_data]:
                legend_data.append((item['name'], color_rgb))

            # 统计信息
            total_pixels = np.sum(roi_mask) if roi_mask is not None else (h * w)
            ratio = (pixel_count / total_pixels) * 100
            stats_info.append(f"{item['name']} ({ratio:.1f}%)")

    # 生成统计文本
    if not found_items:
        findings_str = "CNN: No specific elements/defects detected in this view."
    else:
        findings_str = f"**CNN Detected:** {', '.join(stats_info)}"

    # 混合图像
    if found_items:
        mask_uint8 = combined_mask_bool.astype(np.uint8) * 255
        blended = cv2.addWeighted(res_img_rgb, 0.6, color_canvas, 0.4, 0)
        res_img_rgb[combined_mask_bool] = blended[combined_mask_bool]

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(res_img_rgb, contours, -1, (255, 255, 255), 1)

    return res_img_rgb, findings_str, legend_data


def render_legend_html(legend_data):
    if not legend_data: return ""
    html_str = "#### 🎨 Legend<br>"
    for name, rgb in legend_data:
        color_css = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        html_str += f"<span style='display:inline-block; width:12px; height:12px; background-color:{color_css}; margin-right:5px; border-radius:2px;'></span>**{name}** &nbsp;&nbsp; "
    return html_str


# ==========================================
# 3. Frontend Layout
# ==========================================
st.title("🌉 Bridge Inspection Dashboard")
st.markdown("Ask: *'Show me all elements'* (Visual) OR *'Analyze the cracks'* (Report)")

with st.spinner("Loading AI Models..."):
    hrnet, dev_status = load_model()

if not hrnet: st.stop()

col_img, col_chat = st.columns([1, 2])

with col_img:
    st.subheader("1. Image Input")
    uploaded_file = st.file_uploader("Upload Bridge Image", type=["jpg", "png"], label_visibility="collapsed")
    if uploaded_file:
        if 'curr_image_name' not in st.session_state or st.session_state['curr_image_name'] != uploaded_file.name:
            st.session_state['curr_image'] = Image.open(uploaded_file)
            st.session_state['curr_image_name'] = uploaded_file.name
            st.session_state['chat_history'] = []
        st.image(st.session_state['curr_image'], use_container_width=True)
    else:
        st.info("Please upload an image to start.")

with col_chat:
    st.subheader("2. AI Analysis")
    if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []

    chat_container = st.container(height=600)
    with chat_container:
        for msg in st.session_state['chat_history']:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("image") is not None: st.image(msg["image"])
                if msg.get("legend"): st.markdown(msg["legend"], unsafe_allow_html=True)

    if uploaded_file and (user_query := st.chat_input("E.g., 'Segment the pier', 'Any defects?'")):
        # User Message
        st.session_state['chat_history'].append({"role": "user", "content": user_query})
        with chat_container:
            with st.chat_message("user"): st.markdown(user_query)

        # Assistant Response
        with chat_container:
            with st.chat_message("assistant"):
                ph = st.empty()

                # Step 1: Llama 3 规划
                ph.markdown("🧠 *Llama 3 is thinking...*")
                plan, err = ask_llm_plan(user_query)

                if err or not plan:
                    ph.error(f"Planning Error: {err}")
                elif plan.get('intent') == 'chat':
                    reply = plan.get('reply', 'Hello! I am your bridge assistant.')
                    ph.markdown(reply)
                    st.session_state['chat_history'].append({"role": "assistant", "content": reply})
                else:
                    # Step 2: CNN 分割
                    ph.markdown("👁️ *AECIF-Net is segmenting...*")
                    res_img, cnn_stats, legend_data = process_vision_colorful(hrnet, st.session_state['curr_image'],
                                                                              plan)
                    legend_html = render_legend_html(legend_data)

                    # Step 3: Gemini 总结
                    ph.markdown("🕵️ *Gemini is observing...*")

                    # 🌟 关键：将 Llama 决定的模式传给 Gemini
                    response_mode = plan.get('response_mode', 'detailed')
                    final_report = generate_hybrid_expert_summary(
                        user_query,
                        cnn_stats,
                        st.session_state['curr_image'],
                        response_mode=response_mode
                    )

                    ph.markdown(final_report)

                    # 图片展示逻辑：只有 visualize 模式，或者 detect_defects 且发现了东西时才显示
                    should_show_image = False
                    if plan.get('intent') == 'visualize':
                        should_show_image = True
                    elif plan.get('intent') == 'detect_defects' and len(legend_data) > 0:
                        should_show_image = True  # 查病害如果查到了，也显示一下位置

                    img_to_save = None
                    if should_show_image:
                        st.image(res_img)
                        if legend_html: st.markdown(legend_html, unsafe_allow_html=True)
                        img_to_save = res_img

                    # 保存历史
                    st.session_state['chat_history'].append({
                        "role": "assistant",
                        "content": final_report,
                        "image": img_to_save,
                        "legend": legend_html if should_show_image else None
                    })