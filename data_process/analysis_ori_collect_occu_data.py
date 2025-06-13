'''
数据整理编码格式
0. 日期 [2023XXXX]
1. 天气 [0：晴天，1：大太阳，2：阴天，3：雨天]
2. 时间 [0：9.00～11.00，1：14.00～17.00，2：18.00～22.00]
3. 光线 [0：强光，1：正常亮度光，2：暗光]
4. 目标物体 [0：墙，1：树，2：灌木，3：高草，4：人，5：各种颜色胶纸，6：各种膜（透明、半透明、不透明），7：雨水，8：灰尘/泥土，9：植物叶子，10：手]
5. 采集距离 [0：大于1m，1：1m-20cm，2：20cm-10cm，3：0-10cm]
6. 遮挡占比 [0：无遮挡，1：0-30%，2：30-50%，3：50-80%，4：80-100%]
*_*_*_*_3_[2,3,4]为遮挡采集；
其余命名为非遮挡；
其余命名为非遮挡；
举例：
11月14日，晴天，9.00～11.00正常亮度光，距离植物叶子0-10cm，遮挡30-50%的照片命名为：
20231114_0_0_1_9_3_2_n.jpg
其中“n”为该类别的第n张
'''
import os
import shutil
# ori_collection_occu_data_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/src"
# dest_occu_data_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/occu"
# dest_no_occu_data_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/no_occu"
# os.makedirs(dest_occu_data_path, exist_ok=True)
# os.makedirs(dest_no_occu_data_path, exist_ok=True)
# OCCU_DISTANCE_MARK_LIST = ["3"]
# OOCCU_ASPATIO_MARK_LIST = ["1","2","3","4"]

ori_collection_occu_data_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/occu_224"
dest_occu_data_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/no_occu_224_doubt"
os.makedirs(dest_occu_data_path, exist_ok=True)
OCCU_DISTANCE_MARK_LIST = ["2","3"]
OOCCU_ASPATIO_MARK_LIST = ["0","1","2"]

for ori_img_name in os.listdir(ori_collection_occu_data_path):
    print(ori_img_name)
    clt_distance, clt_aspatio = ori_img_name.split("_")[-3:-1]
    ori_img_path = os.path.join(ori_collection_occu_data_path, ori_img_name)
    if clt_distance in OCCU_DISTANCE_MARK_LIST and clt_aspatio in OOCCU_ASPATIO_MARK_LIST:
        dest_occu_img_path = os.path.join(dest_occu_data_path, ori_img_name)
        shutil.move(ori_img_path, dest_occu_img_path)
    # else:
    #     dest_no_occu_img_path = os.path.join(dest_no_occu_data_path, ori_img_name)
    #     shutil.move(ori_img_path, dest_no_occu_img_path)
    
    
    
     


