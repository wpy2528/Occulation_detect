import cv2
import numpy as np

def is_low_light(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算灰度图像的噪音标准差
    std_dev = np.std(gray)
    
    # 根据噪音标准差判断是否为暗光
    if std_dev < 30:  # 可根据具体情况调整阈值
        return True
    else:
        return False
