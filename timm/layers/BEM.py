import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeDetection(nn.Module):
    def __init__(self, c1, device):
        super(SobelEdgeDetection, self).__init__()
        # Sobel算子用于X方向和Y方向的边缘检测
        self.sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0).repeat(c1, 1, 1, 1).to(device)
        self.sobel_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0).repeat(c1, 1, 1, 1).to(device)

        # self.register_buffer('sobel_kernel_x',
        #                      torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        #                                   dtype=torch.float32).view(1, 1, 3, 3)
        #                      )
        # self.register_buffer('sobel_kernel_y',
        #                      torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        #                                   dtype=torch.float32).view(1, 1, 3, 3)
        #                      )

    def forward(self, x):
        # Edge detection using convolution with Sobel kernels
        grad_x = F.conv2d(x, self.sobel_kernel_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, self.sobel_kernel_y, padding=1, groups=x.size(1))
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return edges


class BoundaryEnhancementLayer(nn.Module):
    def __init__(self, c1):
        super(BoundaryEnhancementLayer, self).__init__()
        self.conv1 = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enhanced_edges = self.conv1(x)
        enhanced_edges = self.relu(enhanced_edges)
        enhanced_edges = self.conv2(enhanced_edges)
        enhanced_edges = self.sigmoid(enhanced_edges)
        return enhanced_edges


class BoundaryEnhancementModule(nn.Module):
    def __init__(self, c1):
        super(BoundaryEnhancementModule, self).__init__()
        self.sobel = None
        self.enhancer = BoundaryEnhancementLayer(c1)
        self.conv = nn.Conv2d(c1 * 2, c1, kernel_size=1)

    def forward(self, x):
        # 提取边界信息
        device = x.device
        if self.sobel is None:  # 如果 sobel 还没有初始化，进行初始化
            self.sobel = SobelEdgeDetection(x.size(1), device)
        edges = self.sobel(x)
        # 增强边界特征
        enhanced_edges = self.enhancer(edges)
        # 将增强后的边界特征与原始图像特征拼接
        combined = torch.cat([x, enhanced_edges], dim=1)  # 在通道维度上拼接
        output = self.conv(combined)
        return output
