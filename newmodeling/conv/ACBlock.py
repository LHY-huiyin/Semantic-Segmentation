import torch
from torch import nn

class ACBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ACBlock, self).__init__()
        self.squre = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=1)
        self.cross_ver = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.cross_hor = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.bn = nn.BatchNorm2d(out_planes)
        self.ReLU = nn.ReLU(True)

    def forward(self, x):
        x1 = self.squre(x)
        x2 = self.cross_ver(x)
        x3 = self.cross_hor(x)
        return self.ReLU(self.bn(x1 + x2 + x3))