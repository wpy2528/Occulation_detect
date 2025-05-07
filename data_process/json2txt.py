import os
import json
import cv2
import shutil
import tqdm

img_path = '/data/DATA/Mower_labeled_data/Occlusion_data/occulation_datasets/wpy/occulation_inference/transfer_occu'
img_save = '/data/DATA/Mower_labeled_data/Occlusion_data/occulation_datasets/wpy/occulation_inference'
json_path = ('/home/wangpengyuan/下载/test.json')
text_save = json_path.replace("json", "txt")
other_path = json_path.replace("json", "other")
# os.makedirs(save_path, exist_ok=True)
# os.makedirs(other_path, exist_ok=True)
cls_list = ['LightOcc', 'PartialOcc', 'FullOcc', 'Full_Occu', 'Light_Occu']
cls_dict = {}


def write_txt():
    img_dict = {}
    with open(json_path, 'r', encoding="utf-8") as f1:
        dates = json.load(f1)
        for data in dates:
            try:
                annotations = data["annotations"]
                results = annotations[0]["result"]
                img_path = str(data["data"]["image"])
                cls_label = str(results[0]["value"]["choices"][0])
                img_dict[img_path] = cls_label
                cls_dict[cls_label] = cls_dict.get(cls_label, 0) + 1
            except:
                print("Do not find the label for img: {}\n".format(str(data["data"]["image"])))
                continue
    print(cls_dict)
    f1.close()
    with open(text_save, "w", encoding="utf-8") as f2:
        for img_name, cls_name in img_dict.items():
            img_name = os.path.basename(img_name)
            data = img_name + " " + cls_name + "\n"
            f2.write(data)
    f2.close()


def split_imgs():
    with open(text_save, 'r', encoding='utf-8') as f:
        dates = f.readlines()
        for data in tqdm.tqdm(dates):
            img_name = data.split(" ")[0]
            cls_name = data.split(" ")[-1].replace('\n', "")
            if cls_name in cls_list:
                os.makedirs(os.path.join(img_save, cls_name), exist_ok=True)
                shutil.move(os.path.join(img_path, img_name), os.path.join(img_save, cls_name, img_name))
    f.close()
        # for i in os.listdir(img_path):




write_txt()
print(cls_dict)
# split_imgs()


# for i in os.listdir(img_path):
#     name = i.replace("jpg", "txt")
#     if name not in os.listdir(save_path):
#         shutil.move(os.path.join(img_path, i), os.path.join(other_path, i))
# for j in os.listdir(json_path):
#     name = j.replace("json", "txt")
#     if name not in os.listdir(save_path):
#         shutil.move(os.path.join(json_path, j), os.path.join(other_path, j))
