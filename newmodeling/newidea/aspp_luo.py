import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from newmodeling.conv.ConvBNReLU import *

class HADCLayer_luo(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, dilation=1, mode='parallel', *args, **kwargs):
        super(HADCLayer_luo, self).__init__()
        self.mode = mode
        self.ks = ks
        if ks > 3:
            padding = int(dilation * ((ks - 1) // 2))
            self.hadc_layer1 = ConvBNReLU(in_chan, out_chan, ks=[3, ks], dilation=[1, dilation], padding=[1, padding])
            # Conv2d(2048, 256, kernel_size=(3, 7), stride=(1, 1), padding=(1, 3), dilation=(1, 1))
            # Conv2d(256, 256, kernel_size=(3, 7), stride=(1, 1), padding=(1, 6), dilation=(1, 2))
            # Conv2d(256, 256, kernel_size=(3, 7), stride=(1, 1), padding=(1, 9), dilation=(1, 3))
            self.hadc_layer2 = ConvBNReLU(in_chan, out_chan, ks=[ks, 3], dilation=[dilation, 1], padding=[padding, 1])
            # Conv2d(2048, 256, kernel_size=(7, 3), stride=(1, 1), padding=(3, 1), dilation=(1, 1))
            # Conv2d(256, 256, kernel_size=(7, 3), stride=(1, 1), padding=(6, 1), dilation=(2, 1))
            # Conv2d(256, 256, kernel_size=(7, 3), stride=(1, 1), padding=(9, 1), dilation=(3, 1))
        elif ks == 3:
            self.hadc_layer = ConvBNReLU(in_chan, out_chan, ks=ks, dilation=dilation, padding=dilation)

        else:
            self.hadc_layer = ConvBNReLU(in_chan, out_chan, ks=ks, dilation=1, padding=0)

        self.init_weight()

    def forward(self, x):
        if self.ks <= 3:
            return self.hadc_layer(x)
        elif self.ks > 3:  # 前两个ks=7,ks=5
            x1 = self.hadc_layer1(x)  # 并行的卷积，卷积核大小为(k1,k2)  torch.Size([4, 256, 24, 24])
            x2 = self.hadc_layer2(x)  # 并行的卷积，卷积核大小为(k2,k1)  torch.Size([4, 256, 24, 24])
            return x1 + x2

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
        self.with_gp = with_gp
        # 1*1 卷积
        self.conv1 = ConvBNReLU(in_chan, out_chan, ks=ks_list[0], dilation=1, padding=0)
        # 1*3 3*1 卷积 1*3 3*1 卷积  rate=1 rate=3
        self.conv2 = nn.Sequential(HADCLayer_luo(in_chan, out_chan,
                                                 ks=ks_list[1], dilation=dilation[0], mode=mode),
                                   HADCLayer_luo(out_chan, out_chan,
                                                 ks=ks_list[1], dilation=dilation[1], mode=mode))
        # 1*3 3*1 卷积 1*3 3*1 卷积  1*3 3*1 卷积 rate=1 rate=3 rate=3
        self.conv3 = nn.Sequential(HADCLayer_luo(in_chan, out_chan,
                                                 ks=ks_list[2], dilation=dilation[0], mode=mode),
                                   HADCLayer_luo(out_chan, out_chan,
                                                 ks=ks_list[2], dilation=dilation[1], mode=mode),
                                   HADCLayer_luo(out_chan, out_chan,
                                                 ks=ks_list[2], dilation=dilation[2], mode=mode)
                                   )
        # 3*3 卷积 rate=1
        self.conv4 = nn.Sequential(HADCLayer_luo(in_chan, out_chan,
                                                 ks=ks_list[3], dilation=dilation[0], mode=mode))

        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, out_chan,
                                      ks=1)  # Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
            self.conv_out = ConvBNReLU(out_chan * 5, out_chan,
                                       ks=1, dilation=1, padding=0)  # Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
        else:
            self.conv_out = ConvBNReLU(out_chan * 4, out_chan, ks=1)

        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        # 1*3 3*1 卷积 1*3 3*1 卷积  rate=1 rate=3
        feat1 = self.conv1(x)  # [4,256,24,24]
        # 1*3 3*1 卷积 1*3 3*1 卷积 1*3 3*1 卷积 rate=1 rate=2 rate=3
        feat2 = self.conv2(x)  # [4,256,24,24]
        # 3*3 卷积
        feat3 = self.conv3(x)  # [4,256,24,24]
        # 1*1 卷积
        feat4 = self.conv4(x)  # [4,256,24,24]
        if self.with_gp:
            avg = self.avg(x)  # [4,2048,1,1]
            feat5 = self.conv1x1(avg)  # [4,256,1,1]
            feat5 = F.interpolate(feat5, (H, W), mode='bilinear', align_corners=True)  # [4,256,24,24]
            feat = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)  # [4,1280,24,24]

        else:
            feat = torch.cat([feat1, feat2, feat3, feat4], 1)

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