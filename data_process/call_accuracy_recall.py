import torch
import torch.nn.functional as F

def accuracy_acc_recall(output, target):
    # 计算整体的准确率
    _, predicted = torch.max(output, 1)
    total = target.size(0)
    correct = (predicted == target).sum().item()
    overall_accuracy = correct / total

    # 计算整体的召回率
    true_positive = ((predicted == 1) & (target == 1)).sum().item()
    false_negative = ((predicted == 0) & (target == 1)).sum().item()
    overall_recall = true_positive / (true_positive + false_negative)

    # 计算每个类别的准确率和召回率
    class_accuracy = torch.zeros(2)
    class_recall = torch.zeros(2)
    for i in range(2):
        class_correct = ((predicted == i) & (target == i)).sum().item()
        class_total = (target == i).sum().item()
        class_accuracy[i] = class_correct / class_total
        class_recall[i] = class_correct / (class_correct + false_negative)

    return overall_accuracy, overall_recall, class_accuracy, class_recall

# pre_label_txt = ""
# pre_label_txt = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/class_result_occu_dest.txt"
# pre_label_txt = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/class_result_occu_onnx_dest.txt"
# pre_label_txt = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/20240205-135542-res18_onnx_dest.txt"
# pre_label_txt = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/20240205-154811-dark-mv2_onnx_dest.txt"
# pre_label_txt = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/20240205-110228-mv2-dest.txt"
# pre_label_txt = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/occulotion_datasets/2023_12_1/train_data/val/20240205-135542-res18-dest.txt"
pre_label_txt = "/media/wpy/558e3a8b-7a8f-4603-a6fb-20c87dbd584b/zhangzhaofeng/bright_dark_1130/new_dark_train/train_data/val/ori_image_file-dest.txt"

f = open(pre_label_txt, "r")
pre_label_lines = f.readlines()
f.close()

pre_array = []
label_array = []
for line in pre_label_lines:
    line = line.strip()
    line_array = line.split(" ")
    temp_pre_arr = []
    temp_pre_arr.append(float(line_array[1]))
    temp_pre_arr.append(float(line_array[2]))
    pre_array.append(temp_pre_arr)
    label_array.append(float(line_array[3]))
    
pre_array_t = torch.Tensor(pre_array)
pre_array_l = torch.Tensor(label_array)
overall_accuracy, overall_recall, class_accuracy, class_recall = accuracy_acc_recall(pre_array_t, pre_array_l)
print("overall_accuracy: ", overall_accuracy)
print("overall_recall: ", overall_recall)
print("class_accuracy: ", class_accuracy)
print("class_recall: ", class_recall)