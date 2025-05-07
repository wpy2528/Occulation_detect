import torch
import torch.nn as nn
import torch.nn.functional as F
from .depthwise_conv import DepthWiseSeparableConv


def adaptive_avg_pool_onnx(input, output_size):
    """
    手动实现适用于 ONNX 导出的自适应平均池化
    """
    # 获取输入空间维度 (H, W)
    input_size = input.shape[2:]

    # 计算步长并确保最小为1
    stride = [max(1, input_size[i] // output_size[i]) for i in range(2)]  # 保证步长≥1
    # stride = [torch.clamp(input_size[i] // output_size[i], min=1) for i in range(2)]

    # 计算核大小并确保最小为1
    kernel_size = [max(1, input_size[i] - (output_size[i] - 1) * stride[i]) for i in range(2)]
    # kernel_size = [input_size[i] - (output_size[i] - 1) * stride[i] for i in range(2)]

    # 执行池化操作
    return F.avg_pool2d(input, kernel_size=kernel_size, stride=stride)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], use_DWconv=True):
        super(PyramidPoolingModule, self).__init__()
        inter_channels = in_channels // 4
        self.pool_sizes = pool_sizes

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True)
            ) for _ in pool_sizes
        ])
        self.out_conv = nn.Conv2d(in_channels + len(pool_sizes) * inter_channels, out_channels, kernel_size=1,
                                  bias=False)
        self.out_bn = nn.BatchNorm2d(out_channels)
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.size()[2:]  # (H, W)
        features = [x]

        for pool_size, conv in zip(self.pool_sizes, self.convs):
            pooled = adaptive_avg_pool_onnx(x, (pool_size, pool_size))
            upsampled = F.interpolate(pooled, size=size, mode='bilinear', align_corners=True)
            features.append(conv(upsampled))

        x = torch.cat(features, dim=1)
        x = self.out_conv(x)
        x = self.out_bn(x)
        return self.out_relu(x)
