import cv2
import numpy as np
import onnxruntime as ort
import os
import shutil

# import string


# onnx_path = "/home/zzf/pytorch-image-model-20231023/pytorch-image-models-main/20240205-135542-res18.onnx"

onnx_path = "output/onnx/20240718-204415-occu-mv2.onnx"
# img_dir = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val"
img_dir = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/bright_dark_1130/dark_dataset/train_data/val"
pre_label_txt_onnx = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/20240205-154811-dark-mv2_onnx.txt"

precoss_qualify_image = False
qualify_image_dir = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/occu_qulify_image"
qualify_class_0_img_count = 0
qualify_class_1_img_count = 0

if precoss_qualify_image == True:
    os.makedirs(qualify_image_dir, exist_ok=True)

import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# 示例
vector = np.array([1.0, 2.0, 3.0])
print("向量的Softmax:", softmax(vector))


def get_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if ".txt" in file:
                print(file)
                continue
            file_list.append(os.path.join(root, file))
    return file_list


def precess_qualify_image(image_file, output):
    class_num = np.argmax(output[0])
    score = np.max(softmax(output[0]))
    global qualify_class_0_img_count
    global qualify_class_1_img_count
    print("class num:", class_num, " score:", score)
    if score > 0.95:
        dest_file = os.path.join(qualify_image_dir, os.path.basename(image_file))
        if class_num == 0:
            qualify_class_0_img_count += 1
            if qualify_class_0_img_count > 100:
                return
            shutil.copy(image_file, dest_file)
        elif class_num == 1:
            qualify_class_1_img_count += 1
            if qualify_class_1_img_count > 100:
                return
            shutil.copy(image_file, dest_file)


def run(image):
    # 加载ONNX模型
    session = ort.InferenceSession(onnx_path)
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    means = np.array(IMAGENET_DEFAULT_MEAN) * 255.0
    std = np.array(IMAGENET_DEFAULT_STD) * 255.0
    # 指定输入名称和形状
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    # 加载图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, tuple(input_shape[2:4]))
    # 归一化图像
    # image = image.astype(np.float32) / 255.0
    image = (image.astype(np.float32) - means) / std
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    # 进行推理
    output = session.run(None, {input_name: image})

    # 解析输出结果
    output = [np.squeeze(out) for out in output]

    # shape (3, 320, 320)
    # seg_out = output[3]
    return output


img_files = get_files(img_dir)
output_lines = []
for image_file in img_files:
    image = cv2.imread(image_file)
    output = run(image)
    if precoss_qualify_image == True:
        precess_qualify_image(image_file, output)
    pres = list(output[0])
    pres = [str(s) for s in pres]
    line = os.path.basename(image_file) + " " + " ".join(pres) + "\n"
    output_lines.append(line)
    # print(output)

f = open(pre_label_txt_onnx, "w")
f.writelines(output_lines)
f.close()
