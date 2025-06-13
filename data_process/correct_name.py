import os
import cv2
import shutil
# image_dir = '/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/charge_datasets/charge_2023_11_24/image'
image_dir = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/no_occu"
for img_name in os.listdir(image_dir):
    new_name = ''
    if ".JPG" in img_name:
        ori_path = os.path.join(image_dir, img_name)
        new_name = img_name.replace(".JPG",".jpg")
        new_path = os.path.join(image_dir, new_name)
        os.rename(ori_path, new_path)
    if " " in img_name:
        ori_path = os.path.join(image_dir, img_name)
        new_name = img_name.replace(" ","")
        new_path = os.path.join(image_dir, new_name)
        os.rename(ori_path, new_path)
        # print(ori_path)
        # print(new_path)
