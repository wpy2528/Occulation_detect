import os
import cv2
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

# path = r'/home/wangpengyuan/OcculsionProject/occlusion_data/inference'
# img_path = r'E:\DATA\Mower_labeled_data\Occlusion_data\transfer\test\no_occu'

img_path = r'/data/DATA/Mower_labeled_data/Occlusion_data/occulation_datasets/wpy/occulation_inference'
t1 = os.path.join(img_path, 'class_result.txt')
t2 = os.path.join(img_path, 'inference.txt')

move_img = False
conf_score = 0.0
cls_list = ["no_occu", "occu", "transfer_occu"]
cls_dict = {"no_occu": 0, "occu": 1, "transfer_occu": 2}
flipped_dict = {value: key for key, value in cls_dict.items()}

cls_tp_dict = {"no_occu": 0, "occu": 0, "transfer_occu": 0}
cls_result_sum_dict = {"no_occu": 0, "occu": 0, "transfer_occu": 0}
cls_truth_sum_dict = {"no_occu": 0, "occu": 0, "transfer_occu": 0}

label_list = []
predict_list = []
score_list = []
end_list = ['jpg', 'png', 'jpeg']

def write_txt():
    cls_write = []
    with open(t1, 'w') as f:
        for root, dirs, files in os.walk(img_path):
            imges = [file for file in files if file.split('.')[-1] in end_list]
            for img in tqdm(imges, desc="reading imgs"):
                for key, index in cls_dict.items():
                    if key == os.path.basename(root).split('-')[0]:
                        if img in cls_write:
                            print(img)
                            break
                        else:
                            cls = img + " " + str(index) + "\n"
                            cls_write.append(cls)
        f.writelines(cls_write)
    f.close()


def img_resize():
    for i in os.listdir(img_path):
        img = cv2.imread(os.path.join(img_path, i))
        if img.shape[:2] != (224, 224):
            img = cv2.resize(img, (224, 224))
            cv2.imwrite(os.path.join(img_path, i), img)


def compare_txt(truth, inference, move_false):
    with open(truth, 'r', encoding='utf-8') as f1:
        f1_data = f1.readlines()
    f1.close()

    tp, fp = 0, 0
    false_NoOccu = 0

    with open(inference, 'r', encoding='utf-8') as f2:
        f2_data = f2.readlines()
    f2.close()
    f1_data.sort()
    f2_data.sort()

    for i in range(len(f1_data)):
        if i < len(f2_data):
            tr = f1_data[i]
            label = tr.split(" ")[-1].replace('\n', "")
            label_list.append(label)
            name_turth = tr.split(" ")[0]
            truth_label = flipped_dict[int(label)]
            infer = f2_data[i]
            name_infer = infer.split(" ")[0]
            result = infer.split(" ")[1]
            predict_list.append(result)
            conf = infer.split(" ")[-1].replace('\n', "")
            score_list.append(conf)
            result_label = flipped_dict[int(result)]
            if float(infer.split(" ")[-1].replace('\n', "")) < conf_score and name_turth == name_infer:
                continue
            elif label == result:
                tp += 1
                cls_tp_dict[result_label] += 1
            else:
                fp += 1
                if move_false:
                    save_path = r'/data/DATA/Mower_labeled_data/Occlusion_data/0826/all'
                    os.makedirs(save_path, exist_ok=True)
                    img_file = os.path.join(img_path, infer.split(" ")[0])
                    save_file = os.path.join(save_path, infer.split(" ")[0])
                    move_false_img(img_file, save_file, save_path)
            if truth_label != 'no_occu' and result_label == 'no_occu':
                false_NoOccu += 1
            cls_truth_sum_dict[truth_label] += 1
            cls_result_sum_dict[result_label] += 1
            print("img {}:  label:{}    pred:{}".format(tr.split(" ")[0], truth_label, result_label))

    acc = tp / (tp + fp) * 100
    false_dete = false_NoOccu / (tp + fp) * 100
    output = 'tp:', tp, 'fp:', fp, 'acc:', str(acc) + '%', 'false:', str(false_dete)
    print(output)
    return output


def caculate():
    data = []
    for i in range(len(cls_result_sum_dict)):
        label = flipped_dict[i]
        tp = cls_tp_dict[label]
        result_sum = cls_result_sum_dict[label]
        truth_sum = cls_truth_sum_dict[label]
        if result_sum != 0 and truth_sum != 0 and tp != 0:
            pre = tp / result_sum * 100
            recall = tp / truth_sum * 100
        else:
            pre = 0
            recall = 0
        pre_result = "{} pre:{}".format(flipped_dict[i], str(pre) + '%')
        print(pre_result)
        recall_result = "{} recall:{}".format(flipped_dict[i], str(recall) + '%')
        print(recall_result)
        data.append(pre_result + '\n')
        data.append(recall_result + '\n')
    return data


def plot_confusion_matrix(cm, labels_name, title, colorbar=False, cmap=None):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    if colorbar:
        plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)
    plt.yticks(num_local, labels_name)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(img_path, "confusion_matrix.png"), format='png')
    plt.show()


def plot_pre_recall_curve(label_bin, pre_list):
    plt.figure(figsize=(8, 6))
    for i in range(label_bin.shape[1]):
        precision, recall, _ = precision_recall_curve(label_bin[:, i], [x[i] for x in pre_list])
        auc_pr = average_precision_score(label_bin[:, i], [x[i] for x in pre_list])
        cls_name = cls_list[i]
        plt.plot(recall, precision, label=f'Class {cls_name} (AUC = {auc_pr:.2f})')

    plt.xlim(0.5, 1)
    plt.ylim(0.5, 1)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Multiclass)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(img_path, "PR_Curve.png"), format='png')
    plt.show()


def move_false_img(img, save_file, save_path):
    print("detection is false,moving img{} to {}".format(img, save_file))
    shutil.copy(img, save_path)


write_txt()
# output = compare_txt(t1, t2, move_false=move_img)
# result = caculate()
# # img_resize()
#
# cm = confusion_matrix(label_list, predict_list)
# plot_confusion_matrix(cm, cls_list, "Confusion Matrix")
#
# label_bin = label_binarize(label_list, classes=[0, 1, 2])
# plot_pre_recall_curve(label_bin, score_list)
