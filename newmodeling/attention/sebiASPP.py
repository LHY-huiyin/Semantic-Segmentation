import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,  # k=2*5-3=7 out=[in-7(k)+6(padding))/1(stride)]+1=in    out = in
                                            stride=1, padding=padding, dilation=dilation, bias=False)  # 空洞卷积  -> [batch, 256,  h, w]
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)  # -> [batch, 256,  h, w]
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class SeBiFPNASPP(nn.Module):
    def __init__(self, inplanes):
        super(SeBiFPNASPP, self).__init__()
        # if output_stride == 16:  # output_stride = 16
        #     dilations = [1, 6, 12, 18]
        # elif output_stride == 8:
        #     dilations = [1, 12, 24, 36]
        # else:
        #     raise NotImplementedError
        "四个卷积层：1,3,5,跳跃连接；特征图通道数变少"
        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=1)  # padding=0 dilation=1   -> [batch, 256,  h, w]
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=3, dilation=3)  # padding=6 dilation=6  -> [batch, 256,  h, w]
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=5, dilation=5)  # padding=12 dilation=12 out=in -> [batch, 256,  h, w]

        self.conv1 = nn.Conv2d(inplanes + 256 * 3, 256, 1, bias=False)  # -> [batch, 256, h, w]
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout:丢掉   表示每个神经元有0.5的可能性不被激活   p为元素被置0的概率，即被‘丢’掉的概率
        self._init_weight()

    def forward(self, x):  # x=[1, 2048, 32, 32]
        x1 = self.aspp1(x)  # ->[1, 256, 32, 32]
        x2 = self.aspp2(x)  # ->[1, 256, 32, 32]
        x3 = self.aspp3(x)  # ->[1, 256, 32, 32]
        x = torch.cat((x1, x2, x3, x), dim=1)  # 将两个张量（tensor）拼接在一起,按维数（1）列对齐 torch.Size([1, 1280, 32, 32])

        x = self.conv1(x)  # 改变通道数 -> [1,256,32,32]
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)  # 防止过拟合：在训练的过程中，随机选择去除一些神经元，在测试的时候用全部的神经元，这样可以使得模型的泛化能力更强，因为它不会依赖某些局部的特征

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return SeBiFPNASPP(backbone, output_stride, BatchNorm)