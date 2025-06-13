import os
import shutil
import random

# train_data_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/train/occu/"
# val_data_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/occu"

# train_data_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/occu_dataset_3_27/train_data/train/transfer_occu"
# val_data_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/occu_dataset_3_27/train_data/val/transfer_occu"
train_data_path = "/data/DATA/Mower_labeled_data/Occlusion_data/occulation_datasets/wpy/test_data/ZLUE-12916/train"
val_data_path = train_data_path.replace("train", "val")
replace_list = []
val_percent = 0.30
os.makedirs(val_data_path, exist_ok=True)
cls_list = ['no_occu', 'occu', 'transfer_occu']
for i in cls_list:
    try:
        train_imgs = os.listdir(os.path.join(train_data_path, i))

        tr = int(len(train_imgs) * val_percent)
        random.seed(10)
        val = random.sample(range(len(train_imgs)), tr)
        for j, img_file in enumerate(train_imgs):
            img_ori_path = os.path.join(train_data_path, i, img_file)
            if j in val:
                os.makedirs(os.path.join(val_data_path, i), exist_ok=True)
                val_img_path = os.path.join(val_data_path, i, img_file)
                shutil.move(img_ori_path, val_img_path)
    except:
        continue
# test_path =[]
# replace_list.append(os.listdir(test_path))
#
# train_imgs = os.listdir(train_data_path)
#
# tr = int(len(train_imgs) * val_percent)
# random.seed(10)
# val = random.sample(range(len(train_imgs)), tr)
# for j, img_file in enumerate(train_imgs):
#     img_ori_path = os.path.join(train_data_path, img_file)
#     if j in val:
#         os.makedirs(val_data_path, exist_ok=True)
#         val_img_path = os.path.join(val_data_path, img_file)
#         shutil.move(img_ori_path, val_img_path)