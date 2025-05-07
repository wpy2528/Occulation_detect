#!/usr/bin/env python3
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import json
import logging
import os
import shutil
import time
from contextlib import suppress
from functools import partial

import numpy as np
import pandas as pd
import torch

from timm.data import create_dataset, create_loader, resolve_data_config, ImageNetInfo, infer_imagenet_subset
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import AverageMeter, setup_default_logging, set_jit_fuser, ParseKwargs
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

try:
    from apex import amp

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

_FMT_EXT = {
    'json': '.json',
    'json-record': '.json',
    'json-split': '.json',
    'parquet': '.parquet',
    'csv': '.csv',
}

cls_list = ["no_occu", "occu", "transfer_occu"]
cls_dict = {0: 'no_occu', 1: 'occu', 2: 'transfer_occu'}
flippp_dict = {value: key for key, value in cls_dict.items()}
cls_tp_dict = {'no_occu': 0, 'occu': 0, 'transfer_occu': 0}
cls_result_sum_dict = {'no_occu': 0, 'occu': 0, 'transfer_occu': 0}
cls_truth_sum_dict = {'no_occu': 0, 'occu': 0, 'transfer_occu': 0}

flipped_dict = {value: key for key, value in cls_dict.items()}

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    # default='/data/DATA/Mower_labeled_data/Occlusion_data/occulation_datasets/wpy/occulation_inference',
                    default='/data/DATA/Mower_labeled_data/Occlusion_data/occulation_datasets/wpy/occulation_data/train_data/val',
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('-s', '--score', default=0.0, type=float,
                    metavar='N', help='conf source')
parser.add_argument('--save_result2img', default=False,
                    help='save result to img')
parser.add_argument('--output_dir', metavar='DIR',
                    default='output/train/best/inference/20250423-192338-resnet18-224',
                    help='path to output files')
parser.add_argument('--checkpoint',
                    default='output/train/20250423-192338-resnet18-224/model_best.pth.tar',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# net-structure improvement
parser.add_argument('--DWconv', default=False, type=bool, help='Using depthwise conv for network')
parser.add_argument('--DSconv', default=False, type=bool, help='Using distributeshift conv for network')
parser.add_argument('--GSconv', default=0, type=int,
                    help='Using groupwise conv for network if using DWconv.-1:groups=c1/2, 1:Not use, >1:=groups')
parser.add_argument('-PPM', '--PyrmaidPooling', default=False, type=bool,
                    help='Add PyrmaidPooling Module in the end of network')
parser.add_argument('-BEM', '--BoundaryEnhancementModule', default=False, type=bool,
                    help='Add BoundaryEnhancementModule in the end of network')
parser.add_argument('-SAFM', '--SpatialAttentionFusionModule', default=False, type=bool,
                    help='Add SpatialAttentionFusionModule in the end of network')
parser.add_argument('--SE', default='se', type=str,
                    help='Using SE Module for network:(se,ese,eca,ecam,ceca,ge,gc,gca,cbam,lcbam')
parser.add_argument('--CBAM', default=False, type=bool, help='Using CBAM Module for network')

parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet18',
                    help='model architecture (default: resnet50)')
# parser.add_argument('--model', '-m', metavar='MODEL', default='resnet18',
#                     help='model architecture (default: resnet50)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=128, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=3,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda:0', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--amp', action='store_true', default=False,
                    help='use Native AMP for mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)

scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', default=False, action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                             help="Enable AOT Autograd support.")

parser.add_argument('--results-dir', type=str, default=None,
                    help='folder for output results')
parser.add_argument('--results-file', type=str, default=None,
                    help='results filename (relative to results-dir)')
parser.add_argument('--results-format', type=str, nargs='+', default=['csv'],
                    help='results format (one of "csv", "json", "json-split", "parquet")')
parser.add_argument('--results-separate-col', action='store_true', default=False,
                    help='separate output columns per result index.')
parser.add_argument('--topk', default=0, type=int,
                    metavar='N', help='Top-k to output to CSV')
parser.add_argument('--fullname', action='store_true', default=False,
                    help='use full sample name in output (not just basename).')
parser.add_argument('--filename-col', type=str, default='filename',
                    help='name for filename / sample name column')
parser.add_argument('--index-col', type=str, default='index',
                    help='name for output indices column(s)')
parser.add_argument('--label-col', type=str, default='label',
                    help='name for output indices column(s)')
parser.add_argument('--output-col', type=str, default=None,
                    help='name for logit/probs output column(s)')
parser.add_argument('--output-type', type=str, default='prob',
                    help='output type colum ("prob" for probabilities, "logit" for raw logits)')
parser.add_argument('--label-type', type=str, default='description',
                    help='type of label to output, one of  "none", "name", "description", "detailed"')
parser.add_argument('--include-index', action='store_true', default=True,
                    help='include the class index in results')
parser.add_argument('--exclude-output', action='store_true', default=False,
                    help='exclude logits/probs from results, just indices. topk must be set !=0.')


def main():
    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # resolve AMP arguments based on PyTorch / Apex availability
    amp_autocast = suppress
    if args.amp:
        assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
        assert args.amp_dtype in ('float16', 'bfloat16')
        amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        _logger.info('Running inference in mixed precision with native PyTorch AMP.')
    else:
        _logger.info('Running inference in float32. AMP not enabled.')

    if args.fuser:
        set_jit_fuser(args.fuser)

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=in_chans,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        use_PPM=args.PyrmaidPooling,
        use_SAFM=args.SpatialAttentionFusionModule,
        attn_layer=args.SE,
        use_DWconv=args.DWconv,
        use_GSconv=args.GSconv,
        use_DSconv=args.DSconv,
        use_CBAM=args.CBAM,
        use_BEM=args.BoundaryEnhancementModule,
        **args.model_kwargs,
    )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.save_result2img:
        clean_save_path(path=args.output_dir)

    _logger.info(
        f'Model {args.model} created, param count: {sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    model = model.to(device)
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        model = torch.jit.script(model)
    elif args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        torch._dynamo.reset()
        model = torch.compile(model, backend=args.torchcompile)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    root_dir = args.data or args.data_dir
    dataset = create_dataset(
        root=root_dir,
        name=args.dataset,
        split=args.split,
        class_map=args.class_map,
    )

    if test_time_pool:
        data_config['crop_pct'] = 1.0

    workers = 1 if 'tfds' in args.dataset or 'wds' in args.dataset else args.workers
    loader = create_loader(
        dataset,
        batch_size=args.batch_size,
        use_prefetcher=True,
        num_workers=workers,
        device=device,
        **data_config,
    )

    to_label = None
    if args.label_type in ('name', 'description', 'detail'):
        imagenet_subset = infer_imagenet_subset(model)
        if imagenet_subset is not None:
            dataset_info = ImageNetInfo(imagenet_subset)
            if args.label_type == 'name':
                to_label = lambda x: dataset_info.index_to_label_name(x)
            elif args.label_type == 'detail':
                to_label = lambda x: dataset_info.index_to_description(x, detailed=True)
            else:
                to_label = lambda x: dataset_info.index_to_description(x)
            to_label = np.vectorize(to_label)
        else:
            _logger.error("Cannot deduce ImageNet subset from model, no labelling will be performed.")

    top_k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    all_indices = []
    all_labels = []
    all_outputs = []
    use_probs = args.output_type == 'prob'
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):

            with amp_autocast():
                output = model(input)

            if use_probs:
                output = output.softmax(-1)

            if top_k:
                output, indices = output.topk(top_k)
                np_indices = indices.cpu().numpy()
                if args.include_index:
                    all_indices.append(np_indices)
                if to_label is not None:
                    np_labels = to_label(np_indices)
                    all_labels.append(np_labels)

            all_outputs.append(output.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))

    all_indices = np.concatenate(all_indices, axis=0) if all_indices else None
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    all_outputs = np.concatenate(all_outputs, axis=0).astype(np.float32)
    filenames = loader.dataset.filenames(basename=not args.fullname)

    output_col = args.output_col or ('prob' if use_probs else 'logit')
    data_dict = {args.filename_col: filenames}
    if args.results_separate_col and all_outputs.shape[-1] > 1:
        if all_indices is not None:
            for i in range(all_indices.shape[-1]):
                data_dict[f'{args.index_col}_{i}'] = all_indices[:, i]
        if all_labels is not None:
            for i in range(all_labels.shape[-1]):
                data_dict[f'{args.label_col}_{i}'] = all_labels[:, i]
        for i in range(all_outputs.shape[-1]):
            data_dict[f'{output_col}_{i}'] = all_outputs[:, i]
    else:
        if all_indices is not None:
            if all_indices.shape[-1] == 1:
                all_indices = all_indices.squeeze(-1)
            data_dict[args.index_col] = list(all_indices)
        if all_labels is not None:
            if all_labels.shape[-1] == 1:
                all_labels = all_labels.squeeze(-1)
            data_dict[args.label_col] = list(all_labels)
        if all_outputs.shape[-1] == 1:
            all_outputs = all_outputs.squeeze(-1)
        data_dict[output_col] = list(all_outputs)

    df = pd.DataFrame(data=data_dict)

    results_filename = args.results_file
    if results_filename:
        filename_no_ext, ext = os.path.splitext(results_filename)
        if ext and ext in _FMT_EXT.values():
            # if filename provided with one of expected ext,
            # remove it as it will be added back
            results_filename = filename_no_ext
    else:
        # base default filename on model name + img-size
        img_size = data_config["input_size"][1]
        results_filename = f'{args.model}-{img_size}'

    if args.results_dir:
        results_filename = os.path.join(args.results_dir, results_filename)

    for fmt in args.results_format:
        # save_results(df, results_filename, fmt)
        write_results2img(df, args, image_path=args.data, save_path=args.output_dir)

    print(f'--result')
    # print(df.set_index(args.filename_col).to_json(orient='index', indent=4))


def caculate_accuracy(tp, fp):
    return tp / (tp + fp) * 100


def clean_save_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if len(os.listdir(path)) > 0:
        for i in os.listdir(path):
            try:
                shutil.rmtree(os.path.join(path, i))
            except:
                os.remove(os.path.join(path, i))


def save_image(img, cls_index, real_label, conf, img_name, save_path=None):
    width, height = img.size
    n_img = Image.new('RGB', (width, height + 50), color='white')
    n_img.paste(img, (0, 50))
    draw = ImageDraw.Draw(n_img)

    font = ImageFont.load_default()

    pre_cls = cls_dict[cls_index]
    text_cls = f'class:{pre_cls}'
    text_conf = f'confidence:{conf:.6f}'
    text_real_label = f'real truth:{real_label}'

    # text_width_cls, text_height_cls = draw.textsize(text_cls, font=font)
    # text_width_conf, text_height_conf = draw.textsize(text_conf, font=font)
    # text_width_realtruth, text_height_realtruth = draw.textsize(text_real_label, font=font)

    draw.text((5, 5), text_cls, fill='black', font=font)
    draw.text((5, 15), text_conf, fill='black', font=font)
    draw.text((5, 25), text_real_label, fill='black', font=font)

    output_path = os.path.join(save_path, img_name)
    n_img.save(output_path)


def plot_confusion_matrix(args, cm, labels_name, title, colorbar=False, cmap=None):
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
    if args.save_result2img:
        plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), format='png')
    plt.show()


def plot_pre_recall_curve(args, label_bin, pre_list):
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
    plt.title('{}_Precision-Recall Curve (Multiclass)'.format(args.output_dir.split('/')[-1]))
    plt.legend(loc='best')
    plt.grid(True)
    if args.save_result2img:
        plt.savefig(os.path.join(args.output_dir, "PR_Curve.png"), format='png')
    plt.show()


def write_results2img(df, args, image_path, save_path):
    pre_list, label_list, score_list = [], [], []
    tp, fp = 0, 0
    if args.save_result2img:
        true_path = os.path.join(save_path, "True")
        os.makedirs(true_path, exist_ok=True)
        false_path = os.path.join(save_path, "False")
        os.makedirs(false_path, exist_ok=True)
    for index, row in df.iterrows():
        img_name = row['filename']
        try:
            cls_index = row['index']
            conf = row['prob']
        except:
            conf = max(row['prob'])
            cls_index = np.argmax(row['prob'])

        for root, dirs, files in os.walk(image_path):
            if img_name in files:
                img_path = root
                try:
                    # real_label = os.path.basename(img_path)
                    real_label = os.path.basename(img_path)
                    img_file = os.path.join(img_path, img_name)
                    img = Image.open(img_file)
                except:
                    print(f"cannot find picture{img_file}")
                    continue
                try:
                    label_index = flipped_dict[real_label]
                    result = cls_dict[cls_index]

                    if label_index == cls_index and conf > args.score:
                        tp += 1
                        cls_tp_dict[real_label] += 1
                        if args.save_result2img and os.path.exists(save_path):
                            save_image(img, cls_index, real_label, conf, img_name, save_path=true_path)
                        cls_result_sum_dict[result] += 1
                        cls_truth_sum_dict[real_label] += 1
                    elif conf < args.score:
                        # save_image(img, cls_index, real_label, conf, img_name, save_path=false_path)
                        continue
                    else:
                        fp += 1
                        if args.save_result2img and os.path.exists(save_path):
                            save_image(img, cls_index, real_label, conf, img_name, save_path=false_path)
                        cls_result_sum_dict[result] += 1
                        cls_truth_sum_dict[real_label] += 1

                    print((
                            f"img:{os.path.basename(img_file)}" + '\t' + f"label:{real_label}" + '\t' + f"pred:{result}" + '\t' + f"source:{conf * 100:.2f}%").expandtabs(
                        35))

                    pre_list.append(cls_index)
                    label_list.append(label_index)
                    score_list.append(row['prob'])
                    if args.save_result2img and os.path.exists(save_path):
                        save_file = os.path.join(save_path, real_label)
                        os.makedirs(save_file, exist_ok=True)
                        save_image(img, cls_index, real_label, conf, img_name, save_path=save_file)
                    elif args.save_result2img and not os.path.exists(save_path):
                        os.makedirs(save_path, exist_ok=True)
                        save_file = os.path.join(save_path, real_label)
                        os.makedirs(save_file, exist_ok=True)
                        save_image(img, cls_index, real_label, conf, img_name, save_path=save_file)
                    else:
                        print("Do not exist save_path, please check your save_result2img!!")
                except:
                    print(f"Do not has {real_label} in our claas list:{cls_list}")

    acc = caculate_accuracy(tp, fp)
    print(f"acc:{acc:.4f}%, tp:{tp}, fp:{fp}")

    for cls, num in cls_tp_dict.items():
        if num != 0:
            cls_pre = num / cls_result_sum_dict[cls] * 100
            cls_recall = num / cls_truth_sum_dict[cls] * 100
        else:
            cls_pre = 0
            cls_recall = 0
        print("{} precision:".format(cls), str(cls_pre) + '%')
        print("{} recall:".format(cls), str(cls_recall) + '%')

    cm = confusion_matrix(label_list, pre_list)

    label_bin = label_binarize(label_list, classes=[0, 1, 2])

    plot_confusion_matrix(args, cm, cls_list, "{}_Confusion Matrix".format(args.output_dir.split('/')[-1]))
    plot_pre_recall_curve(args, label_bin, score_list)


def save_results(df, results_filename, results_format='csv', filename_col='filename'):
    results_filename += _FMT_EXT[results_format]
    if results_format == 'parquet':
        df.set_index(filename_col).to_parquet(results_filename)
    elif results_format == 'json':
        df.set_index(filename_col).to_json(results_filename, indent=4, orient='index')
    elif results_format == 'json-records':
        df.to_json(results_filename, lines=True, orient='records')
    elif results_format == 'json-split':
        df.to_json(results_filename, indent=4, orient='split', index=False)
    else:
        df.to_csv(results_filename, index=False)


if __name__ == '__main__':
    main()
