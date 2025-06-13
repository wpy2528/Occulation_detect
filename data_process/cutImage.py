import cv2
import os
import numpy as np

img_path = r'/data/DATA/Mower_labeled_data/Occlusion_data/occulation_datasets/wpy/test_data/ZLUE-12916/output'
save_path = img_path.replace("output", "")
size_list = ['left', 'right', 'side_left', 'side_right']
# size = size_list[0]
os.makedirs(save_path, exist_ok=True)

for size in size_list:
    for i in os.listdir(img_path):
        print(i)
        if i.endswith("png") or i.endswith("jpg"):
            img_file = os.path.join(img_path, i)
            # img = cv2.imread(img_file)
            img_data = np.fromfile(img_file, dtype=np.int8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            w, h = img.shape[1], img.shape[0]
            # if size == 'left':
            #     img2 = img[:int(h/3), :int(w/3)]
            # if size == 'right':
            #     img2 = img[:int(h/3), int(w/3):int(w*2/3)]
            # if size == 'side_left':
            #     img2 = img[int(h/3):int(h*2/3), :int(w/3)]
            # if size == 'side_right':
            #     img2 = img[int(h/3):int(h*2/3), int(w/3):int(w*2/3)]
            # else:
            #     img2 = img[:272, :320]
            # img2 = img[50:, :]
            if w == 2880:
                if size == 'left':
                    img2 = img[:int(h), :int(w/9)]
                if size == 'right':
                    img2 = img[:int(h), int(w/9):int(w*2/9)]
                if size == 'side_left':
                    img2 = img[:int(h), int(w/3):int(w*4/9)]
                if size == 'side_right':
                    img2 = img[:int(h), int(w*4/9):int(w*5/9)]
            elif w == 3200:
                if size == 'left':
                    img2 = img[:int(h), :int(w/10)]
                if size == 'right':
                    img2 = img[:int(h), int(w/10):int(w*1/5)]
                if size == 'side_left':
                    img2 = img[:int(h), int(w*3/10):int(w*2/5)]
                if size == 'side_right':
                    img2 = img[:int(h), int(w*2/5):int(w*1/2)]

            i = size + '_' + i
            save_file = os.path.join(save_path, size)
            os.makedirs(save_file, exist_ok=True)
            img_save = os.path.join(save_file, i)
            # # cv2.imshow(save_file, img2)
            # # cv2.waitKey()
            # # cv2.destroyAllWindows()
            result = cv2.imwrite(img_save, img2)
            if result:
                    print(img_save)