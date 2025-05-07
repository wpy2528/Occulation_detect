import os
from io import BytesIO
from PIL import Image
import numpy
import cv2

save_path = r'/data/DATA/Mower_labeled_data/Occlusion_data/occulation_datasets/wpy/occulation_data_0417'
os.makedirs(save_path, exist_ok=True)
data_file_path = r"/data/DATA/割草机草地采集数据/record_20250412_135752"
for data_file in os.listdir(data_file_path):
    if data_file.endswith('.drc'):
        with open(os.path.join(data_file_path, data_file), 'rb') as f:
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

                img_data = log_data[14:14 + data_len]

                if data_type > 30:
                    continue
                try:
                    img = Image.open(BytesIO(img_data))
                    # img.show()
                    img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
                    front_left_img = img[:, :320, :]
                    img_file_name = os.path.join(save_path, str(time_stamp) + ".jpg")
                    print(img_file_name)
                    cv2.imwrite(img_file_name, front_left_img)
                except:
                    continue