import colorsys
import copy
import time
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.AECIF_Net import AECIF_Net
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


class HRnet_Segmentation(object):
    _defaults = {
        "model_path": 'model_data/best_epoch_weights.pth',
        "num_classes": [7, 2],
        "backbone": "hrnetv2_w48",
        "input_shape": [520, 520],
        "mix_type": 0,
        "cuda": True,  # 5060 显卡请务必开启 True
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # 颜色表初始化
        if self.num_classes[0] <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128),
                           (128, 64, 12)]
            self.colors_1 = [(128, 128, 128), (128, 64, 0), (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
                             (0, 0, 128), (128, 0, 128), (0, 128, 128),
                             (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                             (64, 128, 128), (192, 128, 128), (0, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                             (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

    def generate(self, onnx=False):
        self.net = AECIF_Net(num_classes=self.num_classes, backbone=self.backbone, pretrained=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🔄 正在加载权重: {self.model_path}")

        # 1. 智能加载 (兼容新老版本 PyTorch)
        try:
            checkpoint = torch.load(self.model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location=device)

        # 2. 自动剥壳 (如果包含 state_dict)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # 3. 修正键名 (移除 module. 前缀)
        weights_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v

        # 4. 加载参数 (严格模式，确保完全匹配)
        self.net.load_state_dict(weights_dict)
        print('✅ 模型权重加载成功！')

        self.net = self.net.eval()
        if not onnx and self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # ==========================================
    # 核心接口：只返回 Mask 数据，不画图
    # ==========================================
    def get_raw_masks(self, image):
        # 1. 预处理
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # Resize & Normalize
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # 2. 推理
            outputs = self.net(images)
            pr_e = outputs[0][0]
            pr_d = outputs[1][0]

            # 3. 后处理
            pr_e = F.softmax(pr_e.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr_d = F.softmax(pr_d.permute(1, 2, 0), dim=-1).cpu().numpy()

            # 去除灰条 (Padding)
            pr_e = pr_e[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr_d = pr_d[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

            # 缩放回原图
            pr_e = cv2.resize(pr_e, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr_d = cv2.resize(pr_d, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

            # 4. 获取 Mask ID
            mask_e = pr_e.argmax(axis=-1)
            mask_d = pr_d.argmax(axis=-1)

        return mask_e, mask_d