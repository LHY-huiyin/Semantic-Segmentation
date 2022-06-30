import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class MASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(MASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048  # backbone = 'resnet'

        if output_stride == 16:  # output_stride = 16
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        # MSPP
        # 膨胀后卷积核尺寸 = 膨胀系数 * (原始卷积核尺寸 - 1) + 1  = 2x(3-1)+1 =5
        # k=2*5-3=7 out=[in-k+2*padding]/stride+1=[(in-7+6)/1]+1=in    out = in
        self.maspp1 = nn.Sequential(
            nn.Conv2d(inplanes, 256, 1, stride=1, padding=0, dilation=dilations[0], bias=False),  # conv1*1
            BatchNorm(256),
            nn.ReLU())  # con1*1   out=[in-1+2*0]/1+1=in  ->[b,256,h,w]
        self.maspp2 = nn.Sequential(
            nn.Conv2d(inplanes, 256, 3, stride=1, padding=dilations[0], dilation=dilations[0], bias=False),
            # conv3*3 rate=1
            BatchNorm(256),
            nn.ReLU())  # con3*3 rate=1  k=3 out=[in-3+2]/1+1=in     ->[b,256,h,w]
        self.maspp3 = nn.Sequential(
            nn.Conv2d(inplanes, 256, 3, stride=1, padding=dilations[0], dilation=dilations[0], bias=False),
            # conv3*3 rate=1
            BatchNorm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=dilations[1], dilation=dilations[1], bias=False),
            # conv3*3 rate=3
            BatchNorm(256),
            nn.ReLU())  # con3*3 rate=1 -> conv3*3 rate=3    rate=3:k=3*(3-1)+1=5 out=[in-5+6]/1+1=in  ->[b,256,h,w]
        self.maspp4 = nn.Sequential(
            nn.Conv2d(inplanes, 256, 3, stride=1, padding=dilations[0], dilation=dilations[0], bias=False),
            # conv3*3 rate=1
            BatchNorm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=dilations[1], dilation=dilations[1], bias=False),
            # conv3*3 rate=3
            BatchNorm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=dilations[2], dilation=dilations[2], bias=False),
            # conv3*3 rate=5
            BatchNorm(256),
            nn.ReLU(), )  # con3*3 rate=1 -> conv3*3 rate=3 -> conv3*3 rate=5     rate=5:k=5*(3-1)+1=11 out=[in-11+5*2]/1+1=in  ->[b,256,h,w]
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
        x1 = self.maspp1(x)  # ->[1, 256, 32, 32]
        x2 = self.maspp2(x)  # ->[1, 256, 32, 32]
        x3 = self.maspp3(x)  # ->[1, 256, 32, 32]
        x4 = self.maspp4(x)  # ->[1, 256, 32, 32]
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


def build_maspp(backbone, output_stride, BatchNorm):
    return MASPP(backbone, output_stride, BatchNorm)