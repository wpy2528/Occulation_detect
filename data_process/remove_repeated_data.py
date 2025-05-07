import os
import shutil

img_path_1 = '/home/wangpengyuan/OcculsionProject/occlusion_data/occulation_datasets/wpy/occulation_1011'
img_path_2 = '/home/wangpengyuan/OcculsionProject/occlusion_data/occulation_datasets/wpy/occulation_inference'

dict_1 = {}
for root, dirs, files in os.walk(img_path_1):
    for file in files:
        if file.endswith('jpg'):
            dict_1[file] = os.path.join(root, file)

list_2 = []
for root, dirs, files in os.walk(img_path_2):
    for file in files:
        if file.endswith('jpg'):
            list_2.append(file)

for name, path in dict_1.items():
    if name in list_2:
        print(path)
        os.remove(path)