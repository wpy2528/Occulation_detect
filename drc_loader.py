import os
from io import BytesIO
from PIL import Image
import numpy
import cv2

save_path = r'/data/DATA/Mower_labeled_data/Occlusion_data/occulation_bug_data/output'
os.makedirs(save_path, exist_ok=True)
drc_path = r"/data/DATA/Mower_labeled_data/Occlusion_data/occulation_bug_data/drc"
data_class = 'occulation'
for file in os.listdir(drc_path):
    if file.endswith('drc'):
        data_file_path = os.path.join(drc_path, file)
        with open(data_file_path, 'rb') as f:
            while True:
                # data = f.read()
                # print(data)

                protocol_head = bytes(f.read(1 + 1 + 2 + 4))
                small_head = bytes(f.read(1 + 1 + 1))
                log_type = int.from_bytes(bytes(f.read(1)), 'little')
                real_len = int.from_bytes(bytes(f.read(4)), 'little')
                robot_sn = bytes(f.read(16))
                log_data = bytes(f.read(real_len))

                if real_len == 0:
                    break
                if log_type != 7:
                    continue
                sys_count = int.from_bytes(log_data[:1], 'little')
                data_type = int.from_bytes(log_data[1:2], 'little')
                data_len = int.from_bytes(log_data[2:6], 'little')
                time_stamp = int.from_bytes(log_data[6:6 + 8], 'little')

                img_data_1 = log_data[14:14 + data_len]
                img_data_2 = log_data[13:13 + data_len]
                img_data_3 = log_data[12:12 + data_len]
                img_data_4 = log_data[11:11 + data_len]

                if data_type > 30:
                    continue

                try:
                    img = Image.open(BytesIO(img_data_1))
                    # img.show()
                    img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
                    if img.shape[1] == 1120:
                        front_left_img = img[:, :160, :]
                        front_right_img = img[:, 160:320, :]
                    elif img.shape[1] == 2240:
                        front_left_img = img[:, :320, :]
                        front_right_img = img[:, 320:640, :]
                    left_img_file_name = os.path.join(save_path, f"{data_class}_left_{time_stamp}.jpg")
                    right_img_file_name = os.path.join(save_path, f"{data_class}_right_{time_stamp}.jpg")
                    # print("stop")
                    cv2.imwrite(left_img_file_name, front_left_img)
                    cv2.imwrite(right_img_file_name, front_right_img)

                    # channel_id = int.from_bytes(img_data[:1], 'little')
                    # pixel_size = int.from_bytes(img_data[1:2], 'little')
                    # width = int.from_bytes(img_data[2:4], 'little')
                    # height = int.from_bytes(img_data[4:6], 'little')
                    # stream_len = int.from_bytes(img_data[6:10], 'little')
                    # stream = img_data[10:10+stream_len]
                except:
                    print("cannot read this drc")
                    continue