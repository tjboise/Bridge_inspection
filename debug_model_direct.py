import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from AECIF_Net import HRnet_Segmentation

# ================= 配置 =================
# 1. 这里填刚才那张全黑的图
TEST_IMAGE_PATH = "data/JPEGImages/t_293.jpg"
# 2. 权重路径
MODEL_PATH = "model_data/best_epoch_weights.pth"


# ========================================

def run_debug():
    print("🚀 启动底层诊断模式...")

    # 1. 强制 CPU 加载 (避免 CUDA 报错)
    try:
        hrnet = HRnet_Segmentation(cuda=False)
        print("✅ 模型包装类初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return

    # 2. 获取核心网络 (绕过所有包装函数)
    net = hrnet.net
    net.eval()  # 开启评估模式
    print("✅ 获取到底层 PyTorch 网络 (hrnet.net)")

    # 3. 读取并预处理图片 (手动复现标准预处理)
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ 找不到图片: {TEST_IMAGE_PATH}")
        return

    image = Image.open(TEST_IMAGE_PATH).convert('RGB')
    w, h = image.size
    print(f"🖼️ 读入图片: {w}x{h}")

    # --- 关键：手动制作输入数据 ---
    # 模拟标准的 resize 和 归一化
    input_shape = [480, 480]  # 假设模型输入是 480x480
    image_data = np.array(image, dtype=np.float32)
    image_data /= 255.0
    image_data -= np.array([0.485, 0.456, 0.406])
    image_data /= np.array([0.229, 0.224, 0.225])

    # Resize + Padding (Letterbox)
    ih, iw = input_shape
    scale = min(iw / w, ih / h)
    nw, nh = int(w * scale), int(h * scale)
    image_resized = cv2.resize(image_data, (nw, nh))

    new_image = np.full((ih, iw, 3), 128, dtype=np.float32)  # 灰色填充
    new_image[(ih - nh) // 2:(ih - nh) // 2 + nh, (iw - nw) // 2:(iw - nw) // 2 + nw, :] = image_resized

    # 转为 Tensor: (Batch, Channel, Height, Width)
    input_tensor = np.transpose(new_image, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, 0)
    input_tensor = torch.from_numpy(input_tensor).float()
    print("✅ 预处理完成，数据形状:", input_tensor.shape)

    # 4. 推理
    print("⏳ 开始推理...")
    with torch.no_grad():
        outputs = net(input_tensor)

        # 兼容性处理：有些模型输出是 list，有些是 tensor
        if isinstance(outputs, (list, tuple)):
            output = outputs[0]  # 取第一个输出 (Element)
        else:
            output = outputs

        # 解析输出
        # output shape: [1, num_classes, h, w]
        pr = F.softmax(output[0].permute(1, 2, 0), dim=-1).cpu().numpy()

        # 裁剪掉 padding 部分
        pr = pr[int((ih - nh) // 2): int((ih - nh) // 2 + nh), int((iw - nw) // 2): int((iw - nw) // 2 + nw)]
        # 放大回原图
        pr = cv2.resize(pr, (w, h), interpolation=cv2.INTER_LINEAR)
        # 取最大概率的类别 ID
        mask = pr.argmax(axis=-1)

    # 5. 结果统计
    unique = np.unique(mask)
    print(f"\n📊 [诊断结果] 图片中检测到的类别 ID: {unique}")

    if len(unique) <= 1 and unique[0] == 0:
        print("😱 结果全黑 (只有背景 ID 0)！")
        print("可能的结论：")
        print("1. 权重文件失效 (best_epoch_weights.pth)")
        print("2. 图片确实是全背景")
        print("3. 模型训练时没收敛")
    else:
        print("🎉 成功检测到构件！模型是好的！")

        # 保存可视化
        colors = [
            [0, 0, 0], [0, 0, 128], [0, 128, 0], [128, 0, 0],
            [128, 128, 0], [0, 0, 128], [128, 0, 128]
        ]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for c in unique:
            if c > 0 and c < len(colors):
                vis[mask == c] = colors[c]  # BGR format

        cv2.imwrite("debug_direct_result.png", vis)
        print("💾 可视化结果已保存为 'debug_direct_result.png'，快去看看！")


if __name__ == "__main__":
    run_debug()