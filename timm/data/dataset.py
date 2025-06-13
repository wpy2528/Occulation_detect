""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import io
import logging
import os.path
import random
from typing import Optional

import torch
import torch.utils.data as data
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from .blur import RandomGaussianBlur, RandomMotionBlur
from .readers import create_reader

_logger = logging.getLogger(__name__)

_ERROR_RETRY = 50


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            img_mode='RGB',
            transform=None,
            target_transform=None,
            gamma_prob=0.3,
            gamma_range=(0.8, 1.2),
            noise_prob=0.5,
            noise_std=0.5,
            blur_prob=0.3,
            motion_kernel_size=3,
            motion_angle_range=(0, 360),
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        if not transform:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-90, 90)),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

        self.random_blur = blur_prob
        self.gaussian_range = (0.5, 1)
        self.motion_kernel_size = motion_kernel_size   # 模糊卷积核大小
        self.motion_angle_range = motion_angle_range

        self.gamma_prob = gamma_prob  # 应用伽马校正的概率
        self.gamma_range = gamma_range  # 伽马值的范围
        self.noise_prob = noise_prob  # 噪声注入的概率
        self.noise_std = noise_std  # 噪声的标准差

    def __getitem__(self, index):
        img, target = self.reader[index]

        try:
            img = img.read() if self.load_bytes else Image.open(img)
            # img = Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0
        if self.img_mode and not self.load_bytes:
<<<<<<< HEAD
            try:
                img = img.convert(self.img_mode)
            except:
                print(self.reader.filename(index))
=======
            img = img.convert(self.img_mode)
>>>>>>> b46b557b2b78636d97c62ecb11541e61ef349921
        if random.random() < self.gamma_prob:
            gamma_value = random.uniform(*self.gamma_range)
            if gamma_value > 1:
                self.noise_std = 0.05
            else:
                self.noise_std = 5
            img = F.adjust_gamma(img, gamma=gamma_value)

        if random.random() < self.random_blur:
            if random.random() <= self.random_blur / 2:
                motion = RandomMotionBlur(kernel_size=self.motion_kernel_size, angle_range=self.motion_angle_range)
                img = motion(img)
            else:
                gaussion = RandomGaussianBlur(radius_range=self.gaussian_range)
                img = gaussion(img)
            # img.show()
        if self.transform is not None:
            img = self.transform(img)

        # 随机决定是否应用噪声
        if random.random() < self.noise_prob:
            # 生成与图像相同形状的高斯噪声
            noise = np.random.normal(0, self.noise_std, img.shape).astype(np.float32)
            # 添加噪声并截断到 [0, 1] 范围
            img = np.clip(img + noise, 0.0, 255.0)
            img = np.round(img).astype(np.uint8)
            # 转换回 tensor
            # img = torch.from_numpy(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            is_training=False,
            batch_size=None,
            seed=42,
            repeats=0,
            download=False,
            transform=None,
            target_transform=None,
    ):
        assert reader is not None
        if isinstance(reader, str):
            self.reader = create_reader(
                reader,
                root=root,
                split=split,
                class_map=class_map,
                is_training=is_training,
                batch_size=batch_size,
                seed=seed,
                repeats=repeats,
                download=download,
            )
        else:
            self.reader = reader
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.reader:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.reader, '__len__'):
            return len(self.reader)
        else:
            return 0

    def set_epoch(self, count):
        # TFDS and WDS need external epoch count for deterministic cross process shuffle
        if hasattr(self.reader, 'set_epoch'):
            self.reader.set_epoch(count)

    def set_loader_cfg(
            self,
            num_workers: Optional[int] = None,
    ):
        # TFDS and WDS readers need # workers for correct # samples estimate before loader processes created
        if hasattr(self.reader, 'set_loader_cfg'):
            self.reader.set_loader_cfg(num_workers=num_workers)

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)
