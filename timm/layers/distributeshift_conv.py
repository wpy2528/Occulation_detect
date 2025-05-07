import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from ultralytics.nn.modules.conv import Conv
import math


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class DSConv(_ConvNd):
    def __init__(self, c1, c2, k, stride=1, padding=None, dilation=1, groups=1, padding_mode='zeros',
                 bias=False, block_size=32, KDSBias=False, CDS=False):
        padding = _pair(autopad(k, padding, dilation))
        kernel_size = _pair(k)
        stride = _pair(stride)
        dilation = _pair(dilation)

        blck_numb = math.ceil(((c1) / (block_size * groups)))
        super(DSConv, self).__init__(
            c1, c2, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.intweight = torch.Tensor(c2, c1, *kernel_size)
        self.alpha = torch.Tensor(c2, blck_numb, *kernel_size)

        self.KDSBias = KDSBias
        self.CDS = CDS

        if KDSBias:
            self.KDSb = torch.Tensor(c2, blck_numb, *kernel_size)
        if CDS:
            self.CDSw = torch.Tensor(c2)
            self.CDSb = torch.Tensor(c2)

        self.reset_parameters()

    def get_weight_res(self):
        alpha_res = torch.zeros(self.weight.shape).to(self.alpha.device)

        if self.KDSBias:
            KDSBias_res = torch.zeros(self.weight.shape).to(self.alpha.device)

        nmb_blocks = self.alpha.shape[1]
        total_depth = self.weight.shape[1]
        bs = total_depth // nmb_blocks

        llb = total_depth - (nmb_blocks - 1) * bs

        for i in range(nmb_blocks):
            length_blk = llb if i == nmb_blocks - 1 else bs

            shp = self.alpha.shape
            to_repaet = self.alpha[:, i, ...].view(shp[0], 1, shp[2], shp[3]).clone()
            repeated = to_repaet.expand(shp[0], length_blk, shp[2], shp[3]).clone()
            alpha_res[:, i * bs:(i * bs + length_blk), ...] = repeated.clone()

            if self.KDSBias:
                to_repaet = self.KDSb[:, i, ...].view(shp[0], 1, shp[2], shp[3]).clone()
                repeated = to_repaet.expand(shp[0], length_blk, shp[2], shp[3]).clone()
                KDSBias_res[:, i * bs:(i * bs + length_blk), ...] = repeated.clone()

        if self.CDS:
            to_repaet = self.CDSw.view(-1, 1, 1, 1)
            repeated = to_repaet.expand_as(self.weight)
            print(repeated.shape)

        weight_res = torch.mul(alpha_res, self.weight)
        if self.KDSBias:
            weight_res = torch.add(weight_res, KDSBias_res)
        return KDSBias_res

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias,
                        self.stride, self.padding, self.dilation,
                        self.groups)


class DSConv2D(Conv):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(inc, ouc, k, s, p, g, d, act)
        self.conv = DSConv(inc, ouc, k, s, p, g, d)


class Bottleneck_DSConv(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = DSConv2D(c1, c_, 1, 1)
        self.cv2 = DSConv2D(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f_DSConv(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = DSConv2D(c1, 2 * self.c, 1, 1)
        self.cv2 = DSConv2D((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck_DSConv(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
