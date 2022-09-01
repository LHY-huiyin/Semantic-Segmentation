import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from newmodeling.conv.ConvBNReLU import *

class HADCLayer(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, dilation=1, mode='parallel', *args, **kwargs):
        super(HADCLayer, self).__init__()
        self.mode = mode
        self.ks = ks
        if ks > 3:
            padding = int(dilation * ((ks - 1) // 2))
            if mode == 'cascade':
                self.hadc_layer = nn.Sequential(ConvBNReLU(in_chan, out_chan,
                                                           ks=[3, ks], dilation=[1, dilation],
                                                           padding=[1, padding]),
                                                ConvBNReLU(out_chan, out_chan,
                                                           ks=[ks, 3], dilation=[dilation, 1],
                                                           padding=[padding, 1]))
            elif mode == 'parallel':
                self.hadc_layer1 = ConvBNReLU(in_chan, out_chan,
                                              ks=[3, ks], dilation=[1, dilation],
                                              padding=[1, padding])  # Conv2d(2048, 256, kernel_size=(3, 7), stride=(1, 1), padding=(1, 3), dilation=(1, 1))
                                                                     # Conv2d(256, 256, kernel_size=(3, 7), stride=(1, 1), padding=(1, 6), dilation=(1, 2))
                                                                     # Conv2d(256, 256, kernel_size=(3, 7), stride=(1, 1), padding=(1, 9), dilation=(1, 3))
                self.hadc_layer2 = ConvBNReLU(in_chan, out_chan,
                                              ks=[ks, 3], dilation=[dilation, 1],
                                              padding=[padding, 1])  # Conv2d(2048, 256, kernel_size=(7, 3), stride=(1, 1), padding=(3, 1), dilation=(1, 1))
                                                                     # Conv2d(256, 256, kernel_size=(7, 3), stride=(1, 1), padding=(6, 1), dilation=(2, 1))
                                                                     # Conv2d(256, 256, kernel_size=(7, 3), stride=(1, 1), padding=(9, 1), dilation=(3, 1))
            else:
                raise Exception('No %s mode, please choose from cascade and parallel' % mode)

        elif ks == 3:
            self.hadc_layer = ConvBNReLU(in_chan, out_chan, ks=ks, dilation=dilation, padding=dilation)

        else:
            self.hadc_layer = ConvBNReLU(in_chan, out_chan, ks=ks, dilation=1, padding=0)

        self.init_weight()

    def forward(self, x):
        if self.mode == 'cascade' or self.ks <= 3:
            return self.hadc_layer(x)
        elif self.mode == 'parallel' and self.ks > 3:  # 前两个ks=7,ks=5
            x1 = self.hadc_layer1(x)  # 并行的卷积，卷积核大小为(k1,k2)  torch.Size([4, 256, 24, 24])
            x2 = self.hadc_layer2(x)  # 并行的卷积，卷积核大小为(k2,k1)  torch.Size([4, 256, 24, 24])
            return x1 + x2

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class LKPBlock(nn.Module):
    def __init__(self, in_chan, out_chan, ks, dilation=[1, 2, 3], mode='parallel', *args, **kwargs):
        super(LKPBlock, self).__init__()
        if ks >= 3:
            self.lkpblock = nn.Sequential(HADCLayer(in_chan, out_chan,
                                                    ks=ks, dilation=dilation[0], mode=mode),
                                          HADCLayer(out_chan, out_chan,
                                                    ks=ks, dilation=dilation[1], mode=mode),
                                          HADCLayer(out_chan, out_chan,
                                                    ks=ks, dilation=dilation[2], mode=mode))
        else:
            self.lkpblock = HADCLayer(in_chan, out_chan, ks=ks)

        self.init_weight()

    def forward(self, x):
        return self.lkpblock(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


# 设计的多尺度特征融合模块
class AsppLuo(nn.Module):
    def __init__(self, in_chan, out_chan, ks_list=[1, 7, 5, 3], dilation=[1, 3, 3], mode='parallel',
                 with_gp=True,
                 *args,
                 **kwargs):
        super(AsppLuo, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(2, stride=2)
        self.with_gp = with_gp
        self.conv1 = LKPBlock(in_chan, out_chan, ks=ks_list[0], dilation=[1, 2, 3], mode=mode)
        self.conv2 = LKPBlock(in_chan, out_chan, ks=ks_list[1], dilation=[1, 2, 3], mode=mode)
        self.conv3 = LKPBlock(in_chan, out_chan, ks=ks_list[2], dilation=[1, 2, 3], mode=mode)
        self.conv4 = LKPBlock(in_chan, out_chan, ks=ks_list[3], mode=mode)
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, out_chan,
                                      ks=1)  # Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
            "此处更改了：是四层的合并融合"
            self.conv_out = ConvBNReLU(out_chan * 4, out_chan,
                                       ks=1)  # Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
        else:
            self.conv_out = ConvBNReLU(out_chan * 4, out_chan, ks=1)

        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        "通过池化层来减少计算量，步长为2的池化"
        x1 = self.pool(x)
        "conv1"
        feat1 = self.cov1(x)
        "conv5*3 3*5"
        feat2 = self.conv2(x)
        "conv7*3 3*7"
        feat3 = self.conv3(x)
        if self.with_gp:
            avg = self.avg(x)  # [4,2048,1,1]
            feat4 = self.conv1x1(avg)  # [4,256,1,1]
            feat4 = F.interpolate(feat4, (H, W), mode='bilinear', align_corners=True)  # [4,256,24,24]
            feat = torch.cat([feat1, feat2, feat3, feat4], 1)  # [4,1280,24,24]
        else:
            feat = torch.cat([feat1, feat2, feat3], 1)

        feat = self.conv_out(feat)

        return feat


    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params = []
        non_wd_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'bias' in name:
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        return wd_params, non_wd_params