import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from timm.models import resnet


def main():
    model = resnet.resnet18()
    model_state_dict = model.state_dict()
    state_dict = torch.load('output/train/20241128-104606-resnet18-224/model_best.pth.tar')
    filtered_state_dict = {k: v for k, v in state_dict.items() if
                           k in model_state_dict and model_state_dict[k].size() == v.size()}
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    model.eval()

    img_path = '/data/DATA/Mower_labeled_data/Occlusion_data/occulation_datasets/wpy/occulation_data/20241128-104606-resnet18-224/True'
    cv2.namedWindow('Grad-CAM', cv2.WINDOW_NORMAL)  # 创建一个窗口
    cv2.moveWindow('Grad-CAM', 100, 100)  # 设置窗口位置
    index = 0
    image_files = os.listdir(img_path)
    while True:
        img_origin = Image.open(os.path.join(img_path, image_files[index]))
        img_data_origin = np.array(img_origin)
        img_data = img_data_origin[50:, :, :]
        img = Image.fromarray(img_data)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(img).unsqueeze(0)
        input_tensor.requires_grad_(True)

        final_conv_layer = model.layer4[1].conv2  # 根据你的模型结构选择

        # 存储梯度和特征图
        gradients = []
        activations = []

        def save_gradient(module, grad_input, grad_output):
            gradients.append(grad_output[0])  # 取第一个梯度

        def get_activation(model, input, output):
            activations.append(output)

        final_conv_layer.register_forward_hook(get_activation)
        # final_conv_layer.register_backward_hook(save_gradient)

        output = model(input_tensor)
        predicted_class = output.argmax(dim=1)

        # 反向传播
        model.zero_grad()
        output[0, predicted_class].backward()

        fc_weights = model.fc.weight.detach().cpu().numpy()
        activation = activations[0].squeeze().detach().cpu().numpy()

        # 获取特征图和梯度
        # activation = activations[0].squeeze().cpu().detach().numpy()
        # gradient = gradients[0].squeeze().cpu().detach().numpy()

        # 计算权重
        # weights = np.mean(gradient, axis=(1, 2))  # 按空间维度求平均

        # 计算Grad-CAM热图
        cam = np.zeros(activation.shape[1:], dtype=np.float32)
        for i in range(activation.shape[0]):
            cam += fc_weights[predicted_class, i] * activation[i, :, :]

        # 使用ReLU激活
        cam = np.maximum(cam, 0)

        # 归一化到[0, 1]
        cam = cam - np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (img.size[0], img.size[1]))

        img_np = np.array(img)
        # 将图像从 RGB 转换为 BGR
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 归一化热图到 [0, 255]
        cam = np.uint8(255 * cam)  # 将热图转换为 0-255
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # 使用 Jet 色图

        # 叠加热图和原始图像
        superimposed_img = cv2.addWeighted(img_np, 0.7, cam, 0.4, 0)
        output_img = np.hstack((superimposed_img, img_data))

        inference_data = img_data_origin[:50, :, :]
        array = np.full((50, inference_data.shape[1], 3), 255, dtype=np.uint8)
        inference_data = np.hstack((inference_data, array))
        out_put = np.vstack((inference_data, output_img))
        # out_put = out_put.astype(np.uint8)
        new_size = (int(out_put.shape[1] * 3.3), int(out_put.shape[0] * 3.3))
        out_put = cv2.resize(out_put, new_size, interpolation=cv2.INTER_LANCZOS4)
        cv2.resize(out_put, (out_put.shape[0] * 2, out_put.shape[1] * 2))
        # 显示结果
        cv2.imshow('Grad-CAM', out_put)

        key = cv2.waitKey(0) & 0xFF  # 等待按键
        if key == ord('q'):  # 按'q'退出
            break
        elif key == ord('d'):  # 按'n'切换到下一张图片
            index = (index + 1) % len(image_files)  # 循环切换
        elif key == ord('a'):  # 按'p'切换到上一张图片
            index = (index - 1) % len(image_files)  # 循环切换
    cv2.destroyAllWindows()


main()
