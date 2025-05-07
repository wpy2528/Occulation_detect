import os
class_0_img_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/bright_dark_1130/new_dark_train/train_data/val/bright"
class_1_img_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/bright_dark_1130/new_dark_train/train_data/val/dark"
# class_0_img_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/no_occu"
# class_1_img_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/occu"
class_0_img_names = os.listdir(class_0_img_path)
class_1_img_names = os.listdir(class_1_img_path)
class_name_dic = {0:class_0_img_names, 1:class_1_img_names}

# pred_txt_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/class_result_occu.txt"
# pred_txt_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/class_result_occu_onnx.txt"
# pred_txt_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/20240205-135542-res18_onnx.txt"
pred_txt_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/20240205-154811-dark-mv2_onnx.txt"
# pred_txt_dest_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/class_result_occu_dest.txt"
# pred_txt_dest_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/20240205-135542-res18_onnx_dest.txt"
pred_txt_dest_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/20240205-154811-dark-mv2_onnx_dest.txt"

# ori_txt_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/ori_image_file.txt"
ori_txt_path = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/bright_dark_1130/new_dark_train/train_data/val/ori_image_file.txt"

def set_ori_file_label():
    lines = []
    for label,file_names in class_name_dic.items():
        for file_name in file_names:
            line = file_name + " " + str(label) + "\n"
            lines.append(line)
    if len(lines) > 0:
        f = open(ori_txt_path, "w")
        f.writelines(lines)
        f.close()

def set_pre_file_label():
    f = open(pred_txt_path, "r")
    preds = f.readlines()
    f.close()
    new_pred_lines = []
    for pred_line in preds:
        pred_line = pred_line.strip()
        img_name = pred_line.split(" ")[0]
        label = -1000
        for class_num, name_list in class_name_dic.items():
            if img_name in name_list:
                label = class_num
        if label == -1000:
            print("image not find error:", img_name)
            continue
        pred_line = pred_line + " " + str(label) + "\n"
        new_pred_lines.append(pred_line)

    if len(new_pred_lines) > 0:
        f = open(pred_txt_dest_path, "w")
        f.writelines(new_pred_lines)
        f.close()
        
set_ori_file_label()
# set_pre_file_label()





