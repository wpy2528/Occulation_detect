import torch.nn as nn


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, c1, c2, k_size, s=1, p=0):
        super(DepthWiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(c1, c1, kernel_size=k_size, stride=s, padding=p, groups=c1, bias=False)
        self.pointwise = nn.Conv2d(c1, c2, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class GroupWiseSeparableConv(nn.Module):
    def __init__(self, c1, c2, k_size, s=1, p=0, groups=1):
        super(GroupWiseSeparableConv, self).__init__()

        self.groups = groups
        if self.groups > 0:
            self.depthwise = nn.Conv2d(c1, c1, kernel_size=k_size, stride=s, padding=p, groups=self.groups)
        elif self.groups == -1:
            if c1 >= 256:
                groups = int(c1 / 2)
            elif 64 < c1 < 256:
                groups = 8
            elif c1 <= 64:
                groups = 2
            self.depthwise = nn.Conv2d(c1, c1, kernel_size=k_size, stride=s, padding=p, groups=groups)
        self.pointwise = nn.Conv2d(c1, c2, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x