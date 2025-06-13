import cv2
import os

img_path = '/home/wangpengyuan/OcculsionProject/occlusion_data/inference'

for root, dirs, files in os.walk(img_path):
    for file in files:
        if file.endswith('.jpg'):
            img_file = os.path.join(root, file)
            img = cv2.imread(img_file)
            img_data = img[100:, :].copy()
            cv2.imwrite(img_file, img_data)