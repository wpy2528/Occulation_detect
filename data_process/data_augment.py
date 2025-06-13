import os
from tqdm import tqdm
import cv2
import numpy as np
import math
import random


def random_find_imgs(img_path, ratio=0.3):
    for root, dirs, files in os.walk(img_path):
        if files[0].endswith(".jpg"):
            img_list = [os.path.join(root, i) for i in files]
            img_list.sort()
            choice_list = random.sample(img_list, int(len(img_list) * ratio))
            return choice_list
        else:
            continue


def img_flip(img_file):
    img = cv2.imread(img_file)
    cv2.flip(img, 1)  # 1 水平翻转;0 垂直翻转;-1 水平垂直翻转
    return img


def save_img(img, save_path, file_name):
    if img is None:
        print("Error: cannot read img {}".format(file_name))
        return
    cv2.imwrite(os.path.join(save_path, file_name), img)
    print("img save in {}".format(os.path.join(save_path, file_name)))


def main(img_path, save_path):
    img_list = random_find_imgs(img_path)
    for i in img_list:
        i = os.path.basename(i)
        i_save = i.split(".")[0] + "_flip.jpg"
        img = img_flip(os.path.join(img_path, i))
        save_img(img, save_path, i_save)


main(
    img_path='/home/wangpengyuan/OcculsionProject/occlusion_data/occulation_datasets/wpy/occulation_data/train_data/train/transfer_occu',
    save_path='/home/wangpengyuan/OcculsionProject/occlusion_data/occulation_datasets/wpy/occulation_data/train_data/test')
