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
# 0. SYSTEM CONFIGURATION (云端发布版)
# ==========================================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    st.error("❌ API Keys missing! Please set them in Streamlit App Settings -> Secrets.")
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
    st.info("🧠 Brain 1: Llama 3 (Planning)")
    st.info("🧠 Brain 2: Gemini 2.5 (Analysis)")
    st.info("👁️ Vision: AECIF-Net (CPU)")
    st.divider()
    st.caption("v4.0 - With Thought Process Log")


# ==========================================
# 2. Backend Logic
# ==========================================

@st.cache_resource
def load_model():
    weight_path = 'model_data/best_epoch_weights.pth'
    if not os.path.exists(weight_path): return None, "Weight file missing"

    print("🔒 Forcing CPU mode for stability...")
    model = HRnet_Segmentation(model_path=weight_path, cuda=False)
    return model, "CPU"


def fallback_keyword_plan(query):
    """
    🚑 紧急备用计划：当 Llama 脑子短路时，用关键词硬匹配
    """
    q = query.lower()
    debug_log = "⚠️ Llama failed or hallucinated. Triggered Rule-Based Fallback.\n"

    # 1. 尝试匹配部件
    elements_map = {
        "bearing": 1, "bracing": 2, "deck": 3,
        "floor beam": 4, "beam": 4,
        "girder": 5, "pier": 6
    }

    for name, eid in elements_map.items():
        if name in q:
            debug_log += f"✅ Found keyword '{name}' -> Intent: visualize"
            return {
                "intent": "visualize",
                "target_layers": [{"type": "elements", "id": eid, "name": name.capitalize()}],
                "reply": f"Found '{name}' in your query."
            }, debug_log

    # 2. 尝试匹配病害
    if "rust" in q or "corrosion" in q:
        debug_log += f"✅ Found keyword 'rust/corrosion' -> Intent: visualize (Rust)"
        return {
            "intent": "visualize",
            "target_layers": [{"type": "defects", "id": 1, "name": "Rust"}],
            "reply": "Switching to Rust detection."
        }, debug_log

    # 3. 默认全量检测
    debug_log += "❌ No keywords found. Defaulting to full defect report."
    return {
        "intent": "detect_defects",
        "target_layers": [],
        "constraint_layer": None
    }, debug_log


def ask_llm_plan(query):
    """Step 1: 使用 Groq (Llama3) 规划 + 关键词双保险 + 返回思考日志"""
    if not GROQ_API_KEY: return fallback_keyword_plan(query)[0], "Groq Key Missing"

    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

    system_prompt = """
    You are the intelligent brain of a Bridge Inspection System. 
    Your job is to understand the user's *underlying intent* and map it to system modules.

    **Core Intents Logic:**

    1. **"visualize"** (The "Eye" Module): 
       - USE WHEN: User wants to know the **location, shape, or existence** of specific parts.
       - KEYWORDS (Flexible): "Show", "Segment", "Highlight", "Where is...", "Draw", "Mark", "Identify visual", "Find".
       - EXAMPLES:
         - "Segment the girder" -> {"intent": "visualize", "target_layers": [{"id": 5, "name": "Girder"}]}
         - "Show me the deck" -> {"intent": "visualize", "target_layers": [{"id": 3, "name": "Deck"}]}

    2. **"detect_defects"** (The "Analyst" Module):
       - USE WHEN: User wants an **assessment, judgment, report, or diagnosis**.
       - KEYWORDS (Flexible): "Analyze", "Assess condition", "Check for damage", "Is it safe?", "Report".

    3. **"scan"** (The "Inventory" Module):
       - USE WHEN: User asks what elements are present in the image.

    4. **"chat"**: Casual conversation.

    **Output JSON ONLY.**
    ID Mapping: Elements: 1:Bearing, 2:Bracing, 3:Deck, 4:Floor Beam, 5:Girder, 6:Pier. Defects: 1:Rust.
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
            temperature=0.1
        )
        raw_content = response.choices[0].message.content

        # 📝 记录原始思考过程
        thought_log = f"**Llama 3 Raw Output:**\n{raw_content}\n\n"

        # 清洗 JSON
        content = raw_content.replace("```json", "").replace("```", "").strip()
        s = content.find('{');
        e = content.rfind('}')

        if s != -1 and e != -1:
            content = content[s:e + 1]
            plan = json.loads(content)
            thought_log += f"**Parsed Plan:**\n{json.dumps(plan, indent=2)}"
            return plan, thought_log
        else:
            raise ValueError("No valid JSON found in response")

    except Exception as e:
        # 🌟 触发兜底，并记录日志
        fallback_plan, fallback_log = fallback_keyword_plan(query)
        error_log = f"⚠️ Llama Error: {str(e)}\n\n👉 {fallback_log}"
        return fallback_plan, error_log


def generate_hybrid_expert_summary(user_query, visual_stats, image_pil, plan_intent="detect_defects"):
    """Step 3: Google Gemini 分析 + 返回思考日志"""
    if not GOOGLE_API_KEY: return "⚠️ Google API Key missing.", "No Key"

    # 🌟 智能切换 Prompt
    if plan_intent == 'visualize':
        prompt_type = "Concise (Visual Check)"
        prompt = f"""
        User Request: "{user_query}"
        CNN Findings: {visual_stats}

        Task: The user wants to see specific parts. 
        1. Confirm what is highlighted in the image based on CNN findings.
        2. Keep it SHORT (1-2 sentences). 
        3. Do NOT write a full inspection report.
        """
    else:
        prompt_type = "Detailed (Inspection Report)"
        prompt = f"""
        You are a Senior Bridge Inspector.
        【Data Source 1: CNN Sensors】 {visual_stats}
        【Data Source 2: Your Vision】 Look at the image. Identify structural anomalies.
        【Task】 Answer: "{user_query}"
        Structure: 1. Sensor Readings, 2. Visual Observations, 3. Conclusion.
        """

    candidate_models = [
        'gemini-2.5-flash',  # 优先
        'gemini-1.5-flash-002',  # 备用
        'gemini-1.5-flash'
    ]

    last_error = ""
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, image_pil])

            # 📝 记录 Gemini 的思考背景
            thought_log = f"**Model Used:** {model_name}\n"
            thought_log += f"**Prompt Strategy:** {prompt_type}\n"
            thought_log += f"**Visual Inputs (CNN):** {visual_stats}\n"

            return response.text, thought_log
        except Exception as e:
            last_error = str(e)
            continue

    return f"All Google models failed. Last error: {last_error}", f"Failed. Errors: {last_error}"


def process_vision_colorful(hrnet, image_pil, plan):
    if plan.get('intent') == 'chat': return None, "", []

    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    mask_e_idx, mask_d_idx = hrnet.get_raw_masks(image_pil)

    res_img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    color_canvas = np.zeros_like(res_img_rgb)
    combined_mask_bool = np.zeros((h, w), dtype=bool)

    targets = plan.get('target_layers', [])
    intent = plan.get('intent')

    # 🛠️ 修复 2：防止漏检 Rust
    # 如果意图是“查病害(detect_defects)”，但列表里竟然没有 Rust，强制加进去！
    if intent == 'detect_defects':
        has_rust = any(t.get('name') == 'Rust' for t in targets)
        if not has_rust:
            print("🔧 Auto-injecting Rust detection for defect mode.")
            targets.append({"type": "defects", "id": 1, "name": "Rust"})

    # 🛠️ 修复 3：防止 visualize 模式下空列表导致不显示
    # 如果是 visualize 但列表为空（比如用户只说 Show me），默认显示所有部件
    if intent == 'visualize' and not targets:
        all_elements = [(1, "Bearing"), (2, "Bracing"), (3, "Deck"), (4, "Floor Beam"), (5, "Girder"), (6, "Pier")]
        for eid, name in all_elements:
            targets.append({"type": "elements", "id": eid, "name": name})

    found_items = []
    legend_data = []
    stats_info = []

    for item in targets:
        # 🛡️ 空值保护：如果 ID 没了，跳过
        if 'id' not in item or item['id'] is None: continue

        # 🛠️ 修复 1：解决 KeyError 'type'
        # 如果 Llama 忘了写 type，我们根据名字自动猜！
        item_type = item.get('type')
        if not item_type:
            # 简单的规则：如果是 Rust 就归为 defects，否则默认 elements
            if item.get('name', '').lower() in ['rust', 'corrosion', 'defect']:
                item_type = 'defects'
            else:
                item_type = 'elements'

        # 根据类型取 Mask
        if item_type == 'elements':
            current_mask = (mask_e_idx == item['id']).astype(np.uint8)
            rgb = hrnet.colors[item['id']] if item['id'] < len(hrnet.colors) else (255, 0, 0)
        elif item_type == 'defects':
            current_mask = (mask_d_idx == item['id']).astype(np.uint8)
            rgb = hrnet.colors_1[item['id']] if item['id'] < len(hrnet.colors_1) else (0, 0, 255)
        else:
            continue

        pixel_count = np.sum(current_mask)
        if pixel_count > 0:
            found_items.append(item['name'])
            color_canvas[current_mask > 0] = rgb
            combined_mask_bool = np.logical_or(combined_mask_bool, current_mask > 0)
            if item['name'] not in [l[0] for l in legend_data]:
                legend_data.append((item['name'], rgb))

            total = h * w
            ratio = (pixel_count / total) * 100
            stats_info.append(f"{item['name']} ({ratio:.1f}%)")

    stats = f"CNN Found: {', '.join(stats_info)}" if found_items else "CNN: No specific target pixels found."

    if found_items:
        mask_uint8 = combined_mask_bool.astype(np.uint8) * 255
        blended = cv2.addWeighted(res_img_rgb, 0.6, color_canvas, 0.4, 0)
        res_img_rgb[combined_mask_bool] = blended[combined_mask_bool]
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(res_img_rgb, contours, -1, (255, 255, 255), 2)

    return res_img_rgb, stats, legend_data


def render_legend_html(legend_data):
    if not legend_data: return ""
    html = "#### 🎨 Legend<br>"
    for name, rgb in legend_data:
        html += f"<span style='display:inline-block;width:12px;height:12px;background:rgb({rgb[0]},{rgb[1]},{rgb[2]});margin-right:5px;'></span>{name} "
    return html


# ==========================================
# 3. Frontend
# ==========================================
st.title("🌉 Bridge Inspection Dashboard")

with st.spinner("Booting up..."):
    hrnet, _ = load_model()

if not hrnet: st.stop()

col_img, col_chat = st.columns([1, 2])

with col_img:
    uploaded_file = st.file_uploader("Upload", type=["jpg", "png"], label_visibility="collapsed")
    if uploaded_file:
        if 'curr_image_name' not in st.session_state or st.session_state['curr_image_name'] != uploaded_file.name:
            st.session_state['curr_image'] = Image.open(uploaded_file)
            st.session_state['curr_image_name'] = uploaded_file.name
            st.session_state['chat_history'] = []
        st.image(st.session_state['curr_image'], use_container_width=True)

with col_chat:
    if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []
    chat_container = st.container(height=500)

    for msg in st.session_state['chat_history']:
        with chat_container.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("image") is not None: st.image(msg["image"])
            if msg.get("legend"): st.markdown(msg["legend"], unsafe_allow_html=True)
            # 🌟 回显历史记录里的思考过程 (如果有)
            if msg.get("thought_log"):
                with st.expander("🧠 View AI Thought Process (Debug Log)"):
                    st.markdown(msg["thought_log"])

    if uploaded_file and (query := st.chat_input("Ex: Segment the girder")):
        st.session_state['chat_history'].append({"role": "user", "content": query})
        with chat_container.chat_message("user"):
            st.markdown(query)

        with chat_container.chat_message("assistant"):
            ph = st.empty()
            ph.markdown("🧠 *Planning...*")

            # Step 1: Plan & Log
            plan, plan_log = ask_llm_plan(query)

            if plan['intent'] == 'chat':
                response = plan.get('reply', 'Hello')
                ph.markdown(response)
                full_log = plan_log
                res_img = None
                legend = None
            else:
                ph.markdown("👁️ *Scanning...*")
                res_img, stats, legend = process_vision_colorful(hrnet, st.session_state['curr_image'], plan)

                ph.markdown("🕵️ *Analyzing...*")
                # Step 3: Analysis & Log
                response, analysis_log = generate_hybrid_expert_summary(query, stats, st.session_state['curr_image'],
                                                                        plan.get('intent'))

                ph.markdown(response)

                # 🌟 合并思考日志
                full_log = f"### Step 1: Planning (Llama 3)\n{plan_log}\n\n---\n### Step 2: Vision (CNN)\n{stats}\n\n---\n### Step 3: Analysis (Gemini)\n{analysis_log}"

                # 展示思考过程折叠框
                with st.expander("🧠 View AI Thought Process (Debug Log)"):
                    st.markdown(full_log)

            # 显示图片逻辑
            should_show = (plan.get('intent') in ['visualize', 'detect_defects']) or (len(legend or []) > 0)

            if should_show and res_img is not None:
                st.image(res_img)
                if legend: st.markdown(render_legend_html(legend), unsafe_allow_html=True)

            st.session_state['chat_history'].append({
                "role": "assistant",
                "content": response,
                "image": res_img if should_show else None,
                "legend": render_legend_html(legend) if should_show and legend else None,
                "thought_log": full_log  # 保存日志到历史
            })