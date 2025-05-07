import os
from tqdm import tqdm
import cv2
import numpy as np
import math
import random


def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def process_img_gamma(img_path, output_path, gamma_factor):
    img = cv2.imread(img_path)

    if img is None:
        print("Error: cannot read img {}".format(img_path))
        return
    img_float = img.astype(np.float32)
    img_gray = cv2.imread(img_path, 0)
    mean = np.mean(img_gray)
    gamma_val = math.log10(gamma_factor) / math.log10(mean / 255)
    img_gamma_correct = gamma_trans(img, gamma_val)

    cv2.imwrite(output_path, img_gamma_correct)
    print("img save in {}".format(output_path))


def random_find_imgs(img_path, ratio=0.3):
    for root, dirs, files in os.walk(img_path):
        if files[0].endswith(".jpg"):
            img_list = [os.path.join(root, i) for i in files]
            img_list.sort()
            choice_list = random.sample(img_list, int(len(img_list) * ratio))
            return choice_list
        else:
            continue


def enhance_img_if_dark(imgs, brightness_threshold=90):
    need_adjust_bright_imgs = []
    for i in tqdm(imgs):
        img = cv2.imread(i)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_img)

        if avg_brightness < brightness_threshold:
            need_adjust_bright_imgs.append(i)
    return need_adjust_bright_imgs


def flip_img(imgs, save_path):
    for i in tqdm(imgs):
        if "_bright" in i:
            img = cv2.imread(i)
            img_flip = cv2.flip(img, 1)
            # cv2.imwrite(os.path.join(save_path, i.split('/')[-1]), img_flip)
            cv2.imwrite(i, img_flip)


def main(img_path, save_path):
    img_list = random_find_imgs(img_path)
    # need_adjust_bright = enhance_img_if_dark(img_list)
    flip_img(img_list, save_path)
    # for i in need_adjust_bright:
    #     i_save = i.split(".")[0] + "_bright.jpg"
    #     save_file = os.path.join(save_path, i_save.split('/')[-1])
    #     process_img_gamma(i, save_file, gamma_factor=0.55)




main(img_path='/home/wangpengyuan/OcculsionProject/occlusion_data/occulation_datasets/wpy/occulation_data/train_data/train/no_occu', save_path='/home/wangpengyuan/OcculsionProject/occlusion_data/occulation_datasets/wpy/occulation_data/train_data/test')