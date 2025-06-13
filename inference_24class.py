#!/usr/bin/env python
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
import numpy as np
import torch
import shutil

from timm.models import create_model, apply_test_time_pool
from timm.data import Dataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='./',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=5, type=int,
                    metavar='N', help='Top-k to output to CSV')


def main():
    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint)

    _logger.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = apply_test_time_pool(model, config, args)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    loader = create_loader(
        Dataset(args.data),
        input_size=config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else config['crop_pct'])

    model.eval()

    k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    topk_ids = []
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            input = input.cuda()
            labels = model(input)
            topk = labels.topk(k)[1]
            topk_ids.append(topk.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    batch_idx, len(loader), batch_time=batch_time))

    topk_ids = np.concatenate(topk_ids, axis=0).squeeze()

    filenames = loader.dataset.filenames()
    for filename, label in zip(filenames, topk_ids):
        filename = os.path.basename(filename)
        if label[0] == 0:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'bed', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 1:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'blanket', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 2:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'cabinet', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 3:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'cat', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 4:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'chair_base', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 5:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'cupboard', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 6:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'dining_table', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 7:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'dirt', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 8:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'dog', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 9:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'door_anno', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 10:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'invalid', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 11:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'person', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 12:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'pet_feces', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 13:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'refrigerator', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 14:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'shoe', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 15:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'socks', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 16:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'sofa', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 17:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'sweeping_machine', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 18:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'tile_floor', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 19:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'toilet', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 20:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'unknown', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 21:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'washing_machine', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 22:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'wire', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)
        if label[0] == 23:
            oldpath = os.path.join(args.data, filename)
            newpath = os.path.join(args.output_dir, 'wood_floor', filename)
            #shutil.copy(oldpath, newpath)
            shutil.move(oldpath, newpath)

    '''
    with open(os.path.join(args.output_dir, './topk_ids.txt'), 'w') as out_file:
        filenames = loader.dataset.filenames()
        for filename, label in zip(filenames, topk_ids):
            filename = os.path.basename(filename)
            out_file.write('{0},{1},{2},{3},{4}\n'.format(
                filename, label[0], label[1], label[2], label[3]))
    '''

if __name__ == '__main__':
    main()
