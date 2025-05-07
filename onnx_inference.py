import argparse
import os.path

import torch
import onnx
import onnxruntime
import cv2
import numpy as np
import timm
from timm.utils.model import reparameterize_model
from timm.utils.onnx import onnx_export
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

EndSwitch = ["jpg", "png", "jpeg"]

cls_dict = {0: 'no_occu', 1: 'occu', 2: 'transfer_occu'}
cls_list = ["no_occu", "occu", "transfer_occu"]
# cls_dict = {2: 'no_occu', 3: 'occu', 1: 'light_occu', 4: 'partial_occu', 0: 'full_occu'}
# cls_list = ["no_occu", "occu", "light_occu", "partial_occu", "full_occu"]
flipped_dict = {value: key for key, value in cls_dict.items()}

cls_tp_dict = {'no_occu': 0, 'occu': 0, 'transfer_occu': 0}
cls_result_sum_dict = {'no_occu': 0, 'occu': 0, 'transfer_occu': 0}
cls_truth_sum_dict = {'no_occu': 0, 'occu': 0, 'transfer_occu': 0}
# cls_tp_dict = {'no_occu': 0, 'occu': 0, 'light_occu': 0, 'partial_occu': 0, 'full_occu': 0}
# cls_result_sum_dict = {'no_occu': 0, 'occu': 0, 'light_occu': 0, 'partial_occu': 0, 'full_occu': 0}
# cls_truth_sum_dict = {'no_occu': 0, 'occu': 0, 'light_occu': 0, 'partial_occu': 0, 'full_occu': 0}
label_list, pre_list = [], []

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
# parser.add_argument('output', metavar='ONNX_FILE',default="black_dark_regney_004.onnx",
#                     help='output model filename')
# parser.add_argument('output', metavar='ONNX_FILE',
#                     help='output model filename', default='black_dark_regney_004.onnx')
# mobilenetv2_100
# resnet18
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet18',
                    help='model architecture (default: mobilenetv3_large_100)')
parser.add_argument('--checkpoint',
                    default="output/onnx/20250417-222324-resnet18-224.onnx",
                    help='onnx model filename')
parser.add_argument('--input_file',
                    default="/data/DATA/Mower_labeled_data/Occlusion_data/occulation_datasets/wpy/occulation_inference",
                    help='onnx model filename')
parser.add_argument('--output_dir', metavar='DIR',
                    default=None,
                    help='path to output files')
parser.add_argument('--opset', type=int, default=None,
                    help='ONNX opset to use (default: 10)')
parser.add_argument('--keep-init', action='store_true', default=False,
                    help='Keep initializers as input. Needed for Caffe2 compatible export in newer PyTorch/ONNX.')
parser.add_argument('--aten-fallback', action='store_true', default=False,
                    help='Fallback to ATEN ops. Helps fix AdaptiveAvgPool issue with Caffe2 in newer PyTorch/ONNX.')
parser.add_argument('--dynamic-size', action='store_true', default=False,
                    help='Export model width dynamic width/height. Not recommended for "tf" models with SAME padding.')
parser.add_argument('--check-forward', action='store_true', default="",
                    help='Do a full check of torch vs onnx forward after export.')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--img-size', default=224, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--num-classes', type=int, default=3,
                    help='Number classes in dataset')
parser.add_argument('--reparam', default=False, action='store_true',
                    help='Reparameterize model')
parser.add_argument('--training', default=False, action='store_true',
                    help='Export in training mode (default is eval)')
parser.add_argument('--verbose', default=False, action='store_true',
                    help='Extra stdout output')


def get_input_name(onnx_session):
    input_name = []
    for node in onnx_session.get_inputs():
        input_name.append(node.name)
    return input_name


def get_input_feed(img_ndarray, onnx_session):
    input_feed = {}
    get_input_list = get_input_name(onnx_session)
    for name in get_input_list:
        input_feed[name] = img_ndarray
    return input_feed


def load_imgs(img_path):
    if os.path.isdir(img_path):
        imgs_list = []
        for root, dirs, files in os.walk(img_path):
            for file in files:
                img_file = os.path.join(root, file)
                imgs_list.append([root, img_file])
        return imgs_list
    else:
        return img_path


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y


def caculate_accuracy(tp, fp):
    return tp / (tp + fp) * 100


def img_dataset(img_path, onnx_session, size=224, source=0, show=False):
    imgs = load_imgs(img_path)
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    means = np.array(IMAGENET_DEFAULT_MEAN) * 255.0
    std = np.array(IMAGENET_DEFAULT_STD) * 255.0
    if isinstance(imgs, list):
        tp, fp = 0, 0
        for root, img_file in imgs:
            if img_file.split(".")[-1] in EndSwitch:
                img = cv2.imread(img_file)
            else:
                continue
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # # or_img = cv2.resize(img, size)
            # image = (img.astype(dtype=np.float32) - means) / std
            # image = image.astype(np.float32)
            # image = np.transpose(image, (2, 0, 1))
            # image = np.expand_dims(image, axis=0)
            # input_feed = get_input_feed(image, onnx_session)
            # label = os.path.basename(root)
            # # pred = sigmoid(onnx_session.run(None, input_feed)[0])
            # output = onnx_session.run(None, input_feed)[0]

            input_name = onnx_session.get_inputs()[0].name
            input_shape = onnx_session.get_inputs()[0].shape
            # 加载图像
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (size, size))
            # 归一化图像
            # image = image.astype(np.float32) / 255.0
            image = (image.astype(np.float32) - means) / std
            image = image.astype(np.float32)
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, axis=0)

            # 进行推理
            output = onnx_session.run(None, {input_name: image})

            # 解析输出结果
            output = [np.squeeze(out) for out in output]

            pred = softmax(output)
            label = os.path.basename(root)
            pred_index = np.argmax(pred)
            result = cls_dict[pred_index]
            label_index = flipped_dict[label]
            print((
                    f"img:{os.path.basename(img_file)}" + '\t' + f"label:{label}" + '\t' + f"pred:{result}" + '\t' + f"source:{np.max(pred) * 100:.2f}%").expandtabs(
                40))
            if label_index == pred_index and np.max(pred) > source:
                tp += 1
                cls_tp_dict[label] += 1
                cls_truth_sum_dict[label] += 1
                cls_result_sum_dict[result] += 1
            elif np.max(pred) < source:
                continue
            else:
                fp += 1
                cls_truth_sum_dict[label] += 1
                cls_result_sum_dict[result] += 1
                if show:
                    cv2.imshow("pre:{}  truth:{}".format(result, label), img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            label_list.append(label_index)
            pre_list.append(pred_index)
        acc = caculate_accuracy(tp, fp)
        print(f"acc:{acc:.4f}%, tp:{tp}, fp:{fp}")
    else:
        img = cv2.imread(imgs)
        or_img = cv2.resize(img, size)
        img = or_img[:, :, ::-1].transpose(2, 0, 1)
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        input_feed = get_input_feed(img, onnx_session)
        pred = onnx_session.run(None, input_feed)[0]
        label = os.path.basename(imgs)
        print((f"img:{img}" + '\t' + f"label:{label}" + '\t' + f"pred:{pred}").expandtabs(40))


def plot_confusion_matrix(save_path, cm, labels_name, title, colorbar=False, cmap=None):
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
    if save_path:
        plt.savefig(os.path.join(save_path, "confusion_matrix.png"), format='png')
    plt.show()


def caculate():
    for cls, num in cls_tp_dict.items():
        if num != 0:
            cls_pre = num / cls_result_sum_dict[cls] * 100
            cls_recall = num / cls_truth_sum_dict[cls] * 100
        else:
            cls_pre = 0
            cls_recall = 0
        print("{} precision:".format(cls), str(cls_pre) + '%')
        print("{} recall:".format(cls), str(cls_recall) + '%')


def main():
    args = parser.parse_args()
    onnx_session = onnxruntime.InferenceSession(args.checkpoint)
    img_dataset(args.input_file, onnx_session, size=args.img_size, show=False)
    caculate()
    cm = confusion_matrix(label_list, pre_list)
    plot_confusion_matrix(args.output_dir, cm, cls_list, "Confusion Matrix")


if __name__ == "__main__":
    main()
