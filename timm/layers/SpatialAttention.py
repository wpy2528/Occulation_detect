import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3/7'
        padding = 3 if kernel_size == 7 else 1
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=padding,
                                bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
