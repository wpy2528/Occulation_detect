import cv2
import os
import numpy as np

img_path = r'/data/DATA/割草机草地采集数据/record_20250412_135752'
save_path = r'/data/DATA/Mower_labeled_data/Occlusion_data/occulation_datasets/wpy/occulation_data_0417'
size_list = ['left', 'right', 'side_left', 'side_right']
# size = size_list[0]
os.makedirs(save_path, exist_ok=True)

for size in size_list:
    for i in os.listdir(img_path):
        if i.endswith("png"):
            img_file = os.path.join(img_path, i)
            # img = cv2.imread(img_file)
            img_data = np.fromfile(img_file, dtype=np.int8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img.shape[:2] == (816, 960):
                if size == 'left':
                    img2 = img[:272, :320]
                if size == 'right':
                    img2 = img[:272, 320:640]
                if size == 'side_left':
                    img2 = img[272:544, :320]
                if size == 'side_right':
                    img2 = img[272:544, 320:640]
            # else:
            #     img2 = img[:272, :320]
            i = size + '_' + i
            save_file = os.path.join(save_path, i)
            result = cv2.imwrite(save_file, img2)
            if result:
                print(save_file)