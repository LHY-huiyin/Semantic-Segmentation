import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,  # k=2*5-3=7 out=[in-7(k)+6(padding))/1(stride)]+1=in    out = in
                                            stride=1, padding=padding, dilation=dilation, bias=False)  # 空洞卷积  -> [batch, 256,  h, w]
        self.bn = BatchNorm(planes)
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

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        elif backbone == 'ghostnet':
            inplanes = 160
        else:
            inplanes = 2048  # backbone = 'resnet'

        if output_stride == 16:  # output_stride = 16
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)  # padding=0 dilation=1   -> [batch, 256,  h, w]
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)  # padding=6 dilation=6  -> [batch, 256,  h, w]
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)  # padding=12 dilation=12 out=in -> [batch, 256,  h, w]
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)  # padding=18 dilation=18 out=in -> [batch, 256,  h, w]

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化：括号内是输出尺寸1*1：相当于全剧平均池化;对每个特征图，累加所有像素值并求平均;减少参数数量，减少计算量，减少过拟合
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),  # -> [batch, 256, h, w]
                                             BatchNorm(256),
                                             nn.ReLU())  # -> [batch, 256, h, w]
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)  # -> [batch, 256, h, w]
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout:丢掉   表示每个神经元有0.5的可能性不被激活   p为元素被置0的概率，即被‘丢’掉的概率
        self._init_weight()

    def forward(self, x):  # x=[1, 2048, 32, 32]
        x1 = self.aspp1(x)  # ->[1, 256, 32, 32]
        x2 = self.aspp2(x)  # ->[1, 256, 32, 32]
        x3 = self.aspp3(x)  # ->[1, 256, 32, 32]
        x4 = self.aspp4(x)  # ->[1, 256, 32, 32]
        x5 = self.global_avg_pool(x)  # ->[1,256,1,1] 全局平均池化 256个1*1的矩阵
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # ->[1,256,32,32]  把256个1*1矩阵数值copy成32*32
            # align_corners：设置为True，则输入和输出张量由其角像素的中心点对齐，从而保留角像素处的值。
            # bilinear：双线性插值  x4.size()[2:]:按照x4的第三维第四维的尺寸大小（32，32）
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # 将两个张量（tensor）拼接在一起,按维数（1）列对齐 torch.Size([1, 1280, 32, 32])

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
    return ASPP(backbone, output_stride, BatchNorm)