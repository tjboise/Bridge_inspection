import time
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import copy

# 导入模型
from AECIF_Net import HRnet_Segmentation


# ==========================================
# 补丁：手动实现双头 detect_image 功能
# ==========================================
def detect_image_patch_two_heads(model_wrapper, image):
    # 1. 图像预处理
    image = image.convert('RGB')
    old_img = copy.deepcopy(image)
    orininal_h = np.array(image).shape[0]
    orininal_w = np.array(image).shape[1]

    # 标准化参数 (ImageNet)
    image_data = np.array(image, dtype=np.float32)
    image_data /= 255.0
    image_data -= np.array([0.485, 0.456, 0.406])
    image_data /= np.array([0.229, 0.224, 0.225])

    # 调整尺寸 (Letterbox)
    input_shape = [480, 480]  # 默认尺寸
    ih, iw = input_shape
    scale = min(iw / orininal_w, ih / orininal_h)
    nw, nh = int(orininal_w * scale), int(orininal_h * scale)
    image_data = cv2.resize(image_data, (nw, nh))

    new_image = np.full((ih, iw, 3), 128, dtype=np.float32)
    new_image[(ih - nh) // 2:(ih - nh) // 2 + nh, (iw - nw) // 2:(iw - nw) // 2 + nw, :] = image_data

    # 转 Tensor
    images = np.transpose(new_image, (2, 0, 1))
    images = np.expand_dims(images, 0)
    images = torch.from_numpy(images).float()

    if model_wrapper.cuda:
        images = images.cuda()

    # 2. 推理 (获取两个头的输出)
    print("⏳ Running inference (Dual Head)...")
    with torch.no_grad():
        outputs = model_wrapper.net(images)

        # ⚠️ 关键点：outputs 是一个 list，包含两个 tensor
        # outputs[0] -> Elements (构件)
        # outputs[1] -> Defects (病害)
        output_e = outputs[0]
        output_d = outputs[1]

    # ---------------------------------------------------
    # 3. 处理第一个头：构件 (Elements)
    # ---------------------------------------------------
    pr_e = F.softmax(output_e[0].permute(1, 2, 0), dim=-1).cpu().numpy()
    pr_e = pr_e[int((ih - nh) // 2): int((ih - nh) // 2 + nh), int((iw - nw) // 2): int((iw - nw) // 2 + nw)]
    pr_e = cv2.resize(pr_e, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
    seg_e = pr_e.argmax(axis=-1)

    # 构件上色 (7色)
    colors_e = [
        [0, 0, 0],  # 0: BG
        [0, 0, 128],  # 1: Bearing (蓝)
        [0, 128, 0],  # 2: Bracing (绿)
        [128, 0, 0],  # 3: Deck (深红)
        [128, 128, 0],  # 4: Floor Beam (青)
        [0, 0, 128],  # 5: Girder (深蓝) - 注意：OpenCV是BGR，所以可能是红
        [128, 0, 128],  # 6: Pier (紫)
        [0, 128, 128]
    ]
    seg_img_e = np.zeros((orininal_h, orininal_w, 3), dtype=np.uint8)
    for c in np.unique(seg_e):
        if c > 0 and c < len(colors_e):
            seg_img_e[seg_e == c] = colors_e[c]

    image_res_e = Image.fromarray(np.uint8(seg_img_e))
    # 混合原图
    final_e = Image.blend(old_img, image_res_e, 0.7)

    # ---------------------------------------------------
    # 4. 处理第二个头：病害 (Defects)
    # ---------------------------------------------------
    pr_d = F.softmax(output_d[0].permute(1, 2, 0), dim=-1).cpu().numpy()
    pr_d = pr_d[int((ih - nh) // 2): int((ih - nh) // 2 + nh), int((iw - nw) // 2): int((iw - nw) // 2 + nw)]
    pr_d = cv2.resize(pr_d, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
    seg_d = pr_d.argmax(axis=-1)

    # 病害上色 (通常 Rust 是红色)
    colors_d = [
        [0, 0, 0],  # 0: BG
        [255, 0, 0]  # 1: Rust (红) - 这里是 RGB 顺序因为用 PIL 处理
    ]
    seg_img_d = np.zeros((orininal_h, orininal_w, 3), dtype=np.uint8)
    for c in np.unique(seg_d):
        if c > 0 and c < len(colors_d):
            seg_img_d[seg_d == c] = colors_d[c]

    image_res_d = Image.fromarray(np.uint8(seg_img_d))
    # 混合原图
    final_d = Image.blend(old_img, image_res_d, 0.7)

    print(f"✅ Elements found: {np.unique(seg_e)}")
    print(f"✅ Defects found:  {np.unique(seg_d)}")

    # 返回两个图的列表
    return [final_e, final_d]


if __name__ == "__main__":
    # 1. 强制 CPU
    hrnet = HRnet_Segmentation(cuda=False)

    # 2. 预测模式
    mode = "predict"

    if mode == "predict":
        while True:
            img = input('Input image filename (e.g., data/JPEGImages/t_293.jpg): ')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                # 3. 调用双头补丁
                r_image = detect_image_patch_two_heads(hrnet, image)

                # 4. 分别显示两张图
                print("🖼️ Displaying Element Mask (Image 1)...")
                r_image[0].show(title="Elements")  # 显示构件图

                print("🖼️ Displaying Defect Mask (Image 2)...")
                r_image[1].show(title="Defects")  # 显示病害图