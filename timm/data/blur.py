from PIL import Image, ImageFilter
import random
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms


# 高斯模糊
class RandomGaussianBlur:
    def __init__(self, radius_range=(1, 3)):
        self.radius_range = radius_range  # 模糊半径范围

    def __call__(self, img):
        # 随机选择模糊的半径值
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius))


# 运动模糊
class RandomMotionBlur:
    def __init__(self, kernel_size=5, angle_range=(-45, 45)):
        self.kernel_size = kernel_size
        self.angle_range = angle_range

    def __call__(self, img):
        # 随机选择角度
        angle = random.uniform(*self.angle_range)
        # 生成运动模糊卷积核
        kernel = self._motion_blur_kernel(self.kernel_size, angle)

        # 转换为tensor并进行卷积
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)  # 添加batch维度
        # 使用conv2d进行卷积
        img_blurred = F.conv2d(img_tensor, kernel, padding=self.kernel_size // 2, stride=1)
        # 去掉batch维度并确保值在[0, 1]范围内
        img_blurred = img_blurred.squeeze(0).clamp(0, 1)
        # 转换回PIL图像
        return transforms.ToPILImage()(img_blurred)

    def _motion_blur_kernel(self, size, angle):
        # 生成运动模糊的卷积核
        kernel = np.zeros((3, size, size), dtype=np.float32)  # 为每个通道生成卷积核
        center = size // 2
        angle_rad = np.deg2rad(angle)

        # 构造运动模糊效果的卷积核
        for i in range(size):
            x = int(center + i * np.cos(angle_rad))
            y = int(center + i * np.sin(angle_rad))
            if 0 <= x < size and 0 <= y < size:
                kernel[:, x, y] = 1  # 为所有通道添加模糊效应

        kernel /= kernel.sum()  # 归一化，保持图像亮度不变
        # 转换为torch tensor并调整为[1, 3, size, size]，适用于卷积
        kernel = torch.tensor(kernel).unsqueeze(0)  # [1, 3, size, size]
        return kernel.repeat(3, 1, 1, 1)
