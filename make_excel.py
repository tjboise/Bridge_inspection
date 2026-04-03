import os
import numpy as np
import pandas as pd
from PIL import Image
import sys

# --- 1. 路径配置 ---
BASE_PATH = "VOCdevkit/VOC2007"
IMAGE_FOLDER = os.path.join(BASE_PATH, "JPEGImages")
MASK_FOLDER = os.path.join(BASE_PATH, "SegmentationClass", "element")
OUTPUT_EXCEL = "Bridge_Elements_List.xlsx"

# --- 2. 基于 AECIF-Net 代码定义的颜色映射 (RGB) ---
COLOR_MAP = {
    (0, 0, 0): "BackGround",
    (128, 0, 0): "Bearing",
    (0, 128, 0): "Bracing",
    (128, 128, 0): "Deck",
    (0, 0, 128): "Floor Beam",
    (128, 0, 128): "Girder",
    (0, 128, 128): "Pier"
}


def get_elements_from_mask(mask_path):
    """提取 Mask 图片中所有出现的构件名称"""
    if not os.path.exists(mask_path):
        return None  # 返回 None 表示找不到文件

    mask = Image.open(mask_path).convert('RGB')
    mask_np = np.array(mask)

    # 获取图中出现的所有颜色
    unique_colors = np.unique(mask_np.reshape(-1, 3), axis=0)

    found_elements = []
    for color in unique_colors:
        color_tuple = tuple(color)
        if color_tuple in COLOR_MAP:
            name = COLOR_MAP[color_tuple]
            if name != "BackGround":
                found_elements.append(name)

    return found_elements


def main():
    data_list = []

    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ 错误：找不到图片文件夹: {IMAGE_FOLDER}")
        return

    # 获取所有图片列表
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_count = len(image_files)
    print(f"🚀 开始处理，共计 {total_count} 张图片\n" + "-" * 50)

    for index, img_name in enumerate(image_files):
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(MASK_FOLDER, mask_name)

        elements = get_elements_from_mask(mask_path)

        # 打印当前进度
        progress = (index + 1) / total_count * 100

        if elements is None:
            status = "⚠️ 缺失标注文件"
        elif len(elements) == 0:
            status = "⚪ 仅有背景"
            data_list.append({"Image Name": img_name, "Elements": "None (Background Only)"})
        else:
            status = f"✅ 发现构件: {', '.join(elements)}"
            for element in elements:
                data_list.append({"Image Name": img_name, "Elements": element})

        # 实时打印每一步结果
        print(f"[{index + 1}/{total_count}] {progress:.1f}% | 正在处理: {img_name} -> {status}")

    # --- 保存结果 ---
    print("-" * 50 + "\n📊 扫描完成，正在生成 Excel...")

    if not data_list:
        print("❌ 未提取到任何有效数据，不生成 Excel。")
        return

    df = pd.DataFrame(data_list)
    df = df.sort_values(by="Image Name")

    try:
        df.to_excel(OUTPUT_EXCEL, index=False)
        print(f"✨ 成功！结果已保存至: {os.path.abspath(OUTPUT_EXCEL)}")
    except Exception as e:
        print(f"❌ 导出失败: {e}")


if __name__ == "__main__":
    main()