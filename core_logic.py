import os
import cv2
import json
import numpy as np
import google.generativeai as genai
import torch
import torch.nn.functional as F
from PIL import Image

# 尝试从环境变量获取 Key
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
except:
    pass

# ==========================================
# 1. 知识库定义
# ==========================================
ELEMENT_MAP = {
    1: "Bearing", 2: "Bracing", 3: "Deck",
    4: "Floor Beam", 5: "Girder", 6: "Pier"
}
DEFECT_MAP = {
    1: "Rust"
}


# ==========================================
# 2. 核心逻辑
# ==========================================
def preprocess_input_safe(image):
    # ⚠️ 关键修正：确保输入是 RGB 格式，且归一化正确
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])  # ImageNet Mean (RGB)
    image /= np.array([0.229, 0.224, 0.225])  # ImageNet Std (RGB)
    return image


def safe_get_raw_masks_patch(self, image):
    # 1. 确保转为 Numpy 数组 (RGB)
    if isinstance(image, Image.Image):
        image = np.array(image)

    image = np.array(image, dtype=np.float32)
    # ❌ [DELETED] 不要转 BGR！PyTorch 模型通常需要 RGB！
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    original_h, original_w = image.shape[:2]

    # 获取模型期望的输入尺寸
    if hasattr(self, 'input_shape'):
        input_shape = self.input_shape
    else:
        # 如果模型没存 input_shape，默认给个常见的 HRNet 尺寸 (防止报错)
        input_shape = [480, 480]

    iw, ih = input_shape[1], input_shape[0]
    w, h = original_w, original_h

    # 保持长宽比缩放 (Letterbox)
    scale = min(iw / w, ih / h)
    nw, nh = int(w * scale), int(h * scale)

    image_data = cv2.resize(image, (nw, nh))

    # 填充灰色背景 (128)
    new_image = np.full((ih, iw, 3), 128, dtype=np.float32)
    new_image[(ih - nh) // 2:(ih - nh) // 2 + nh, (iw - nw) // 2:(iw - nw) // 2 + nw, :] = image_data

    # 归一化
    new_image = preprocess_input_safe(new_image)

    # HWC -> CHW
    image_data = np.transpose(new_image, (2, 0, 1))
    image_data = np.expand_dims(image_data, 0)

    # 推理
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if self.cuda:
            images = images.cuda()
        else:
            images = images.cpu()

        outputs = self.net(images)

        # 处理输出
        if isinstance(outputs, (list, tuple)):
            pr_e = outputs[0][0]  # Element head
            pr_d = outputs[1][0]  # Defect head
        else:
            pr_e = outputs[0]
            pr_d = outputs[0]  # Fallback

        # Softmax
        pr_e = F.softmax(pr_e.permute(1, 2, 0), dim=-1).cpu().numpy()
        pr_d = F.softmax(pr_d.permute(1, 2, 0), dim=-1).cpu().numpy()

        # 裁剪掉 Padding 区域 (反向操作)
        pr_e = pr_e[int((ih - nh) // 2): int((ih - nh) // 2 + nh), int((iw - nw) // 2): int((iw - nw) // 2 + nw)]
        pr_d = pr_d[int((ih - nh) // 2): int((ih - nh) // 2 + nh), int((iw - nw) // 2): int((iw - nw) // 2 + nw)]

        # 缩放回原图尺寸
        pr_e = cv2.resize(pr_e, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        pr_d = cv2.resize(pr_d, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

        # Argmax 获取最终 Mask
        mask_e = pr_e.argmax(axis=-1)
        mask_d = pr_d.argmax(axis=-1)

    return mask_e, mask_d


def load_model_core():
    from AECIF_Net import HRnet_Segmentation
    try:
        use_cuda = torch.cuda.is_available()
        # use_cuda = False # 如需强制 CPU，取消此行注释

        print(f"⚙️ Hardware Device: {'GPU (CUDA)' if use_cuda else 'CPU'}")

        # 加载模型
        hrnet = HRnet_Segmentation(model_path='model_data/best_epoch_weights.pth', cuda=use_cuda)

        # 打印模型输入尺寸，用于诊断
        if hasattr(hrnet, 'input_shape'):
            print(f"📏 Model Expected Input Shape: {hrnet.input_shape}")
        else:
            print("⚠️ Model input_shape not found, defaulting logic to [480, 480]")

        # 应用补丁
        import types
        hrnet.get_raw_masks = types.MethodType(safe_get_raw_masks_patch, hrnet)
        print("✅ Patch applied: 'safe_get_raw_masks_patch' (RGB Color Fix).")
        return hrnet
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_vision_core(hrnet, image_path_or_pil, plan):
    # 1. 读图
    image = None
    if isinstance(image_path_or_pil, str):
        try:
            from PIL import Image
            image = Image.open(image_path_or_pil).convert('RGB')  # 强制 RGB
        except:
            return np.zeros((100, 100), dtype=np.uint8)
    else:
        image = image_path_or_pil

    if hasattr(image, 'shape'):
        h, w = image.shape[:2]
    else:
        w, h = image.size

    if not plan or not isinstance(plan, dict): return np.zeros((h, w), dtype=np.uint8)

    # 2. 推理
    mask_e, mask_d = hrnet.get_raw_masks(image)

    # 尺寸对齐
    if mask_e.shape[:2] != (h, w):
        mask_e = cv2.resize(mask_e, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_d = cv2.resize(mask_d, (w, h), interpolation=cv2.INTER_NEAREST)

    final_mask = np.zeros((h, w), dtype=np.uint8)
    targets = plan.get("target_layers", [])
    constraints = plan.get("constraint_layers", [])

    # 3. 组合 Mask
    roi_mask = None
    if constraints:
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        for c in constraints:
            cid = c.get("id")
            if cid is None and c.get("name"):
                for eid, ename in ELEMENT_MAP.items():
                    if ename.lower() in c["name"].lower():
                        cid = eid;
                        break
            if cid is not None:
                roi_mask = cv2.bitwise_or(roi_mask, (mask_e == cid).astype(np.uint8))

    for t in targets:
        tid = t.get("id")
        tname = t.get("name", "").lower()
        ttype = t.get("type", "elements")

        # 强制修正 Rust ID
        if any(x in tname for x in ["rust", "oxidation", "corrosion", "damage", "defect"]):
            tid = 1
            ttype = "defects"

        # 补全 ID
        if tid is None and tname:
            for eid, ename in ELEMENT_MAP.items():
                if ename.lower() in tname:
                    tid = eid;
                    ttype = "elements";
                    break

        if tid is not None:
            if ttype == "defects":
                curr_mask = (mask_d == tid).astype(np.uint8)
            else:
                curr_mask = (mask_e == tid).astype(np.uint8)

            if roi_mask is not None:
                curr_mask = cv2.bitwise_and(curr_mask, roi_mask)

            final_mask = cv2.bitwise_or(final_mask, curr_mask)

    return final_mask


def ask_gemini_planner_core(user_prompt):
    model = genai.GenerativeModel('models/gemini-2.0-flash')

    schema_str = f"""
    AVAILABLE ELEMENTS (ID map): {json.dumps(ELEMENT_MAP)}
    AVAILABLE DEFECTS (ID map): {json.dumps(DEFECT_MAP)}
    """

    system_instruction = f"""
    You are a Bridge Inspection Planner.
    Translate queries into a JSON execution plan.
    {schema_str}
    REQUIRED OUTPUT FORMAT (JSON ONLY):
    {{
      "target_layers": [{{"id": <int>, "type": "elements"|"defects", "name": "<str>"}}],
      "constraint_layers": [{{"id": <int>, "type": "elements", "name": "<str>"}}]
    }}
    RULES:
    1. Map "Oxidation", "Corrosion" -> Rust (ID: 1).
    2. Map "Beam" -> Girder (ID: 5).
    """

    try:
        response = model.generate_content(f"{system_instruction}\n\nUser: {user_prompt}\nAI:")
        text = response.text.replace("```json", "").replace("```", "").strip()
        if "{" in text: text = text[text.find("{"):text.rfind("}") + 1]
        return json.loads(text)
    except Exception as e:
        print(f"\n❌ Gemini Planner Error: {e}")
        return {}