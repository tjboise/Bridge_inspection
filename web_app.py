import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os
import google.generativeai as genai
from AECIF_Net import HRnet_Segmentation

# ==========================================
# 1. 系统配置 & 风格
# ==========================================
RUTGERS_LOGO = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Rutgers_Scarlet_Knights_logo.svg/1200px-Rutgers_Scarlet_Knights_logo.svg.png"

st.set_page_config(
    page_title="BridgeGPT",
    page_icon=RUTGERS_LOGO,
    layout="wide"
)

# ==========================================
# 1.5 页面固定布局 (CSS Injection)
# 🎯 作用：锁定全局滚动，强制适配屏幕
# ==========================================
st.markdown("""
    <style>
        /* 隐藏最外层的滚动条 */
        .stApp {
            height: 100vh;
            overflow: hidden;
        }

        /* 移除顶部多余的间距，让内容更紧凑 */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
            height: 100vh;
        }

        /* 针对 Streamlit 的主内容区域进行高度锁定 */
        section.main {
            overflow: hidden;
        }

        /* 保持侧边栏如果内容多的话可以滚动，或者也固定 */
        [data-testid="stSidebarUserContent"] {
            overflow-y: auto;
        }
    </style>
    """, unsafe_allow_html=True)


# API 配置
# GOOGLE_API_KEY = "AIzaSyDMLr1ohvRxzcahRm6-vClKH7fcc1cGqzo"


# 修改后：
import streamlit as st

# 从 Streamlit 的安全设置中读取
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("请在 Secrets 中配置 GOOGLE_API_KEY")



# genai.configure(api_key=GOOGLE_API_KEY)

# 类别映射
ELEMENT_MAP = {1: "bearing", 2: "bracing", 3: "deck", 4: "floor_beam", 5: "girder", 6: "pier"}
DEFECT_MAP = {1: "rust"}


# ==========================================
# 2. 模型加载 (The Loader)
# ==========================================
@st.cache_resource
def load_model():
    weight_path = 'model_data/best_epoch_weights.pth'
    if not os.path.exists(weight_path): return None, "Weight file missing"
    model = HRnet_Segmentation(model_path=weight_path, cuda=False)
    return model, "CPU"


# ==========================================
# 3. 增强型规划者 (The Logical Planner)
# 🎯 引入 Single, Union, Intersection 逻辑
# ==========================================
def ask_gemini_planner(query):
    models = ['models/gemini-2.0-flash']

    base_prompt = f"""
    # Role Assignment
    You are an AI Bridge Inspection Assistant (BridgeGPT) from Rutgers University. 

    # Task
    Analyze the user's query and decide the best intent:
    1. **Visual Task**: Requests to show, segment, highlight, or measure bridge parts (bearing, bracing, deck, floor_beam, girder, pier, rust).
    2. **General Chat**: Greetings, "Who are you?", or general bridge knowledge NOT requiring image marking.
    3. - If user asks about "Condition State", "CS", "Severity", or "Status", set "mode": "technical", "intent": "intersection" (to focus on the part).

    # Output Schema (JSON only)
    - For Visual Tasks:
      {{ "intent": "single"|"union"|"intersection", "target": {{ "type": "element"|"defect", "name": "str" }}, "mode": "summary"|"technical" }}
    - For General Chat:
      {{ "intent": "chat", "reply": "Direct conversational answer" }}

    # Logic Rules:
    - "Corrosion" is "rust" (Defect ID 1).
    - Use "intersection" ONLY for relationship patterns like "rust on pier" or "corrosion in girder".
    - Set "mode": "technical" ONLY if the user asks for areas, percentages, or explicit numbers.
    - For "What is this image?" or "Describe this bridge", use intent "single", target "deck" (as a pivot), mode "summary".

    # Examples
    ## 1. General Chat (No image processing)
    User: "Hello, who are you?"
    {{ "intent": "chat", "reply": "Hello! I am BridgeGPT, an AI assistant developed at Rutgers University specializing in bridge structural health monitoring and inspection." }}

    User: "What is the purpose of a bridge pier?"
    {{ "intent": "chat", "reply": "A bridge pier is a type of structure that transmits the vertical load of the bridge super-structure to the foundation, providing intermediate support between the abutments." }}

    ## 2. Visual Task (Segmentation required)
    User: "Show me the rust and the girders."
    {{ "intent": "union", "targets": [ {{ "type": "defect", "name": "rust" }}, {{ "type": "element", "name": "girder" }} ], "mode": "summary" }}

    User: "How much corrosion is on the pier?"
    {{ "intent": "intersection", "base": {{ "type": "element", "name": "pier" }}, "filter": {{ "type": "defect", "name": "rust" }}, "mode": "technical" }}

    User: "Highlight the deck."
    {{ "intent": "single", "target": {{ "type": "element", "name": "deck" }}, "mode": "summary" }}

    # Operation:
    Always respond in correct JSON without explanations or markdown formatting. 
    Now, for this user query:
    User Query: {query}
    """

    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            res = model.generate_content(base_prompt)

            # 强化 JSON 提取逻辑
            raw_text = res.text
            start_idx = raw_text.find("{")
            end_idx = raw_text.rfind("}")

            if start_idx != -1 and end_idx != -1:
                json_str = raw_text[start_idx:end_idx + 1]
                plan = json.loads(json_str)
                return plan, model_name
        except Exception as e:
            continue

    # 最终兜底：如果完全无法解析，作为普通对话处理
    return {"intent": "chat",
            "reply": "I'm having trouble understanding the logic, but I'm here to help with your bridge inspection!"}, "Fallback"


# ==========================================
# 4. 逻辑视觉执行 (The Logical Vision Executor)
# 🎯 核心逻辑：处理并集与交集运算
# ==========================================
def process_logical_vision(hrnet, image_pil, plan):
    # 获取原始 CNN 输出
    mask_e_pil, mask_d_pil = hrnet.get_raw_masks(image_pil)
    raw_e, raw_d = np.array(mask_e_pil), np.array(mask_d_pil)
    h, w = raw_e.shape

    # 准备画布
    final_mask = np.zeros((h, w), dtype=np.uint8)
    intent = plan.get("intent", "single")
    found_info = []
    legend = []

    # 反向查找 ID
    E_REV = {v: k for k, v in ELEMENT_MAP.items()}

    try:
        # --- 逻辑 1: Single ---
        if intent == "single":
            target = plan['target']
            name = target['name'].lower()
            if "rust" in name or "corrosion" in name:
                final_mask = (raw_d == 1).astype(np.uint8) * 101  # 101 为 Rust 特殊色
            else:
                eid = next((k for k, v in ELEMENT_MAP.items() if v == name), 0)
                final_mask = (raw_e == eid).astype(np.uint8) * eid

        # --- 逻辑 2: Union (并集) ---
        elif intent == "union":
            for t in plan.get('targets', []):
                name = t['name'].lower()
                if "rust" in name:
                    final_mask[raw_d == 1] = 101
                else:
                    eid = next((k for k, v in ELEMENT_MAP.items() if v == name), 0)
                    final_mask[raw_e == eid] = eid

        # --- 逻辑 3: Intersection (交集/约束) ---
        elif intent == "intersection":
            base_name = plan['base']['name'].lower()
            filt_name = plan['filter']['name'].lower()

            eid = next((k for k, v in ELEMENT_MAP.items() if v == base_name), 0)
            base_mask = (raw_e == eid)

            if "rust" in filt_name:
                rust_mask = (raw_d == 1)
                overlap = np.logical_and(base_mask, rust_mask)
                final_mask[overlap] = 101
                # 计算在该构件上的病害占比
                base_area = np.sum(base_mask)
                if base_area > 0:
                    ratio = (np.sum(overlap) / base_area) * 100
                    found_info.append(f"Rust on {base_name}: {ratio:.2f}% of component area")

    except Exception as e:
        st.error(f"Logic Processing Error: {e}")

    # 计算统计信息（通用）
    unique_ids = np.unique(final_mask)
    for uid in unique_ids:
        if uid == 0: continue
        px = np.sum(final_mask == uid)
        name = "Rust" if uid == 101 else ELEMENT_MAP.get(uid, "Unknown")
        color = (128, 0, 0) if uid == 101 else hrnet.colors[uid]
        legend.append((name, color))
        if intent != "intersection":  # Intersection 已在上面单独处理
            ratio = (px / (h * w)) * 100
            found_info.append(f"{name}: {ratio:.2f}% total area")

    # 渲染可视化
    res_img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    overlay = np.zeros_like(res_img)
    for name, clr in legend:
        uid = 101 if name == "Rust" else next(k for k, v in ELEMENT_MAP.items() if v == name.lower())
        overlay[final_mask == uid] = clr[::-1]  # RGB to BGR

    if legend:
        res_img = cv2.addWeighted(res_img, 0.7, overlay, 0.3, 0)
        # 画轮廓线
        mask_binary = (final_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(res_img, contours, -1, (255, 255, 255), 2)

    return cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), ", ".join(found_info), legend

# ==========================================
# 5.5 RAG 知识库加载 (AASHTO PDF)
# ==========================================
@st.cache_resource
def load_aashto_manual():
    # 建议使用相对路径的基准，防止云端路径偏移
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "standard/AASHTO-bridge_element_guide_manual__05092010.pdf")

    if not os.path.exists(pdf_path):
        st.error(f"找不到手册文件: {pdf_path}")
        return None

    try:
        # 💡 注意：genai.upload_file 在云端可能因为并发产生冲突
        # 建议在 st.cache_resource 下运行，确保每个 session 只上传一次
        pdf_file = genai.upload_file(path=pdf_path, display_name="AASHTO Manual")
        return pdf_file
    except Exception as e:
        st.error(f"Google API 上传失败: {e}")
        return None


# ==========================================
# 5. 推理生成 (The Reasoner)
# ==========================================
def generate_reasoning_response(query, stats, image, plan, pdf_file_handle):
    """
    🎯 核心功能：结合 AASHTO 手册、视觉数据和原始图片进行专业判定。
    pdf_file_handle: 通过 genai.upload_file 上传后的文件对象。
    """
    mode = plan.get('mode', 'summary')
    intent = plan.get('intent', 'single')

    # 逻辑判断：用户是否在问等级、状态或严重程度
    is_condition_query = any(k in query.lower() for k in ["condition", "state", "cs", "rank", "severity", "level"])

    if is_condition_query and pdf_file_handle:
        # --- RAG 模式：资深工程师持证上岗 ---
        instruction = (
            "You are a Senior Bridge Inspector at Rutgers University. "
            "A user is asking for the Condition State (CS) of a bridge element. "
            "\n\n[TASK]"
            "1. Reference the attached 'AASHTO Manual' specifically for 'Defect 1000: Corrosion'. "
            "2. Analyze the 'Vision Stats' which provides the percentage of corrosion area. "
            "3. Cross-reference the percentage with the AASHTO criteria (CS1 to CS4). "
            "4. Provide a definitive Condition State (e.g., CS3 - Poor). "
            "5. Cite the specific criteria from the manual as 'Evidence'. "
            "6. Keep the response under 4 sentences, professional and evidence-based."
        )
    elif mode == "technical":
        # --- 技术数据模式：只报数 ---
        instruction = (
            "Role: Bridge Inspector (Data Focus). "
            "Report the quantitative findings from 'Vision Stats' accurately. "
            "Do NOT reference the manual unless asked for Condition State. "
            "Direct and concise (Max 2 sentences)."
        )
    else:
        # --- 概览模式：感性描述 ---
        instruction = (
            "Role: Bridge Assistant. Provide a qualitative summary. "
            "Avoid numbers. Focus on the visual appearance and location. "
            "Brief and natural (Max 2 sentences)."
        )

    # 构造最终 Prompt
    prompt = f"""
    {instruction}

    [CONTEXT DATA]
    - User Query: {query}
    - Vision Stats (AI Detection): {stats}

    [OUTPUT]
    Respond in a neutral, expert tone. No prefaces like 'Sure' or 'Based on'.
    """

    # 调用模型
    # 注意：我们同时传入了 pdf_file_handle (手册), image (图片) 和 prompt (文字)
    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        # 如果你已经有 gemini-2.0-flash 权限，也可以尝试替换
        content_payload = [pdf_file_handle, image, prompt]
        res = model.generate_content(content_payload)
        return res.text
    except Exception as e:
        return f"Logic Error: Could not access AASHTO manual. {str(e)}"

# ==========================================
# 6. UI 前端
# ==========================================
st.markdown(
    f"""<div style="display:flex;align-items:center;"><img src="{RUTGERS_LOGO}" width="40"><h1 style="margin-left:15px;">BridgeGPT</h1></div>""",
    unsafe_allow_html=True)

# hrnet, _ = load_model()
# 初始化模型和 PDF 知识库
hrnet, _ = load_model()
aashto_pdf = load_aashto_manual() # 获取 PDF 句柄

col1, col2 = st.columns([1, 1.8])

with col1:
    up_file = st.file_uploader("Upload Bridge Photo", type=["jpg", "png"])
    if up_file:
        img_pil = Image.open(up_file)
        st.image(img_pil, width="stretch")

with col2:
    if 'history' not in st.session_state: st.session_state['history'] = []
    chat_box = st.container(height=500)

    for msg in st.session_state['history']:
        with chat_box.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("img") is not None:
                # 历史记录也使用并排布局
                t_col, i_col = st.columns([1.5, 1])
                with t_col:
                    st.markdown(msg["content"])
                    if msg.get("legend_html"):
                        st.markdown(msg["legend_html"], unsafe_allow_html=True)
                with i_col:
                    st.image(msg["img"], width="stretch")
            else:
                # 用户消息或纯文本回复
                st.markdown(msg["content"])

    st.write("---")
    st.markdown("##### 💡 Popular Questions")
    q_cols = st.columns(2)
    questions = [
        "Can you simply describe the figure?",  # Summary mode
        "Show me the deck and the girders",  # Union mode
        "Can you segment all elements?",  # Union mode
        "Can you count the corrosion area on pier?"  # Intersection + Technical mode
    ]

    selected_query = None
    for i, q in enumerate(questions):
        if q_cols[i % 2].button(q, use_container_width=True, key=f"q_{i}"):
            selected_query = q

    user_query = st.chat_input("Ask about bridge logic...")
    final_query = selected_query if selected_query else user_query

    if up_file and final_query:
        st.session_state['history'].append({"role": "user", "content": final_query})
        with chat_box.chat_message("user"):
            st.markdown(final_query)

        with chat_box.chat_message("assistant"):
            st_status = st.empty()

            # 1. Planner 阶段
            st_status.info("🧠 Thinking (Single/Union/Intersection)...")
            plan, model_used = ask_gemini_planner(final_query)

            # 2. 视觉逻辑阶段
            st_status.info("👁️ Executing Pixel Logic...")
            res_img, stats, legend = process_logical_vision(hrnet, img_pil, plan)

            # 3. 推理生成阶段
            st_status.info("📝 Generating Report...")
            reply = generate_reasoning_response(final_query, stats, img_pil, plan,aashto_pdf)

            st_status.empty()  # 移除状态提示

            # ✨ 核心修改：使用 columns 实现左文右图
            if plan['intent'] != 'chat' and res_img is not None:
                # 创建两列，比例可以根据喜好调整，这里建议 1.5 : 1
                text_col, img_col = st.columns([1.5, 1])

                with text_col:
                    st.markdown(reply)
                    # 如果有图例，显示在文字下方
                    if legend:
                        lg_html = "".join([
                            f"<span style='background:rgb{tuple(c)};padding:2px 8px;margin-right:5px;border-radius:3px;font-size:0.8rem;color:white;'>{n}</span>"
                            for n, c in legend])
                        st.markdown(lg_html, unsafe_allow_html=True)

                with img_col:
                    # 图像在这里会自动缩小以适应列宽
                    st.image(res_img, width="stretch")
            else:
                # 如果只是普通聊天（没有图像生成），则直接显示文字
                st.markdown(reply)

            # 存入历史记录时，我们也需要标记它是并排显示的
            st.session_state['history'].append({
                "role": "assistant",
                "content": reply,
                "img": res_img if plan['intent'] != 'chat' else None,
                "legend_html": lg_html if (plan['intent'] != 'chat' and legend) else None
            })