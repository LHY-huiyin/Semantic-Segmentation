import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.PPM import *


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


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
                                              padding=[1,
                                                       padding])  # Conv2d(2048, 256, kernel_size=(3, 7), stride=(1, 1), padding=(1, 3), dilation=(1, 1))
                # Conv2d(256, 256, kernel_size=(3, 7), stride=(1, 1), padding=(1, 6), dilation=(1, 2))
                # Conv2d(256, 256, kernel_size=(3, 7), stride=(1, 1), padding=(1, 9), dilation=(1, 3))
                self.hadc_layer2 = ConvBNReLU(in_chan, out_chan, ks=[ks, 3], dilation=[dilation, 1], padding=[padding,
                                                                                                              1])  # Conv2d(2048, 256, kernel_size=(7, 3), stride=(1, 1), padding=(3, 1), dilation=(1, 1))
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


class LKPP(nn.Module):
    def __init__(self, in_chan=2048, out_chan=256, ks_list=[7, 5, 3, 1], mode='parallel', with_gp=True, *args,
                 **kwargs):
        super(LKPP, self).__init__()
        self.with_gp = with_gp
        self.conv1 = LKPBlock(in_chan, out_chan, ks=ks_list[0], dilation=[1, 2, 3], mode=mode)
        self.conv2 = LKPBlock(in_chan, out_chan, ks=ks_list[1], dilation=[1, 2, 3], mode=mode)
        self.conv3 = LKPBlock(in_chan, out_chan, ks=ks_list[2], dilation=[1, 2, 3], mode=mode)
        self.conv4 = LKPBlock(in_chan, out_chan, ks=ks_list[3], mode=mode)
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, out_chan,
                                      ks=1)  # Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
            self.conv_out = ConvBNReLU(out_chan * 5, out_chan,
                                       ks=1)  # Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
        else:
            self.conv_out = ConvBNReLU(out_chan * 4, out_chan, ks=1)

        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat1 = self.conv1(x)  # [4,256,24,24]
        feat2 = self.conv2(x)  # [4,256,24,24]
        feat3 = self.conv3(x)  # [4,256,24,24]
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

class BifpnConvs(nn.Module):
    def __init__(self, in_chan, out_chan, ks=1, stride=1, padding=1, dilation=1, *args, **kwargs):
        super(BifpnConvs, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=False)
        self.ppm = PSPModule(in_chan, norm_layer=nn.BatchNorm2d, out_features=256)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.ppm(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

"解码器结构"
class Decoder_BiFPN_EaNet(nn.Module):
    def __init__(self, n_classes, low_chan=[1024, 512, 256, 64], num_classes=8, levels=4, init=0.5, eps=0.0001, *args, **kwargs):
        super(Decoder_BiFPN_EaNet, self).__init__()
        self.eps = eps
        self.levels = levels
        # weighted
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.relu1 = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 1).fill_(init))
        self.relu2 = nn.ReLU()
        self.conv_16 = ConvBNReLU(low_chan[0], 256, ks=3, padding=1)  # 1*1卷积，如果padding=1，特征图尺寸会改变
        self.conv_8 = ConvBNReLU(low_chan[1], 256, ks=3, padding=1)
        self.conv_4 = ConvBNReLU(low_chan[2], 256, ks=3, padding=1)
        self.conv_2 = ConvBNReLU(low_chan[3], 256, ks=3, padding=1)

        # self.Poollayer = torch.nn.AvgPool2d(kernel_size=2, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
        # self.ppm = PSPModule(low_chan, norm_layer=nn.BatchNorm2d, out_features=256)
        # self.conv_out = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        self.conv_loss = ConvBNReLU(256, num_classes, kernel_size=1, bias=False)  # supervisor的输出
        self.bifpn_convs = BifpnConvs(256, 256, kernel_size=1, padding=0)  #

        self.init_weight()

    def forward(self, feat2, feat4, feat8, feat16, feat_lkpp):  # feat_lkpp:[4, 256, 26, 26] feat16:[4,1024,24,24] feat8:[4,512,48,48] feat4:[4,256,96,96] feat2:[4,64,192,192]
        H, W = feat16.size()[2:]
        feat_lkpp_up = F.interpolate(feat_lkpp, (H, W), mode='bilinear', align_corners=True)
        "对每一个编码器输出的特征图进行1*1卷积"
        feat16_1 = self.conv_16(feat16)  # [4,1024,24,24] -> [4, 256, 24, 24]  3*3卷积
        feat8_1 = self.conv_8(feat8)  # [4,512,48,48] -> [4, 256, 48, 48]
        feat4_1 = self.conv_4(feat4)  # [4,256,96,96] -> [4, 256, 96, 96]
        feat2_1 = self.conv_2(feat2)  # [4,64,192,192] -> [4,256,192,192]
        "bifpn"
        w1 = self.relu1(self.w1)  # [2,4]
        w1 = w1 / torch.sum(w1, dim=0) + self.eps
        w2 = self.relu2(self.w2)  # [3,3]
        w2 = w2 / torch.sum(w2, dim=0) + self.eps

        feat4_2 = (w1[0, 0] * F.max_pool2d(feat2_1, kernel_size=2) + w1[1, 0] * feat4_1) / (
                w1[0, 0] + w1[1, 0] + self.eps)  # [4, 256, 96, 96]

        # feat8_2 = (w1[0, 0] * self.Poollayer(feat4_1) + w1[1, 0] * feat8_1) / (w1[0, 0] + w1[1, 0] + self.eps)
        feat8_2 = (w1[0, 1] * F.max_pool2d(feat4_1, kernel_size=2) + w1[1, 1] * feat8_1) / (
                w1[0, 1] + w1[1, 1] + self.eps)  # [4, 256, 48, 48]
        feat8_2 = self.bifpn_convs(feat8_2)  # [4, 256, 48, 48]

        # feat16_2 = (w1[0, 1] * self.Poollayer(feat8_2) + w1[0, 1] * feat16_1) / (w1[0, 1] + w1[1, 1] + self.eps)
        feat16_2 = (w1[0, 2] * F.max_pool2d(feat8_2, kernel_size=2) + w1[1, 2] * feat16_1) / (
                w1[0, 2] + w1[1, 2] + self.eps)  # [4, 256, 24, 24]
        feat16_2 = self.bifpn_convs(feat16_2)  # [4, 256, 24, 24]

        # featlkpp_2 = (w1[0, 2] * self.Poollayer(feat16_2) + w1[0, 2] * feat_lkpp) / (w1[0, 2] + w1[1, 2] + self.eps)

        featlkpp_2 = (w1[0, 3] * feat16_2 + w1[1, 3] * feat_lkpp_up) / (
                w1[0, 3] + w1[1, 3] + self.eps)
        featlkpp_2 = self.bifpn_convs(featlkpp_2)  # [4, 256, 24, 24]
        # 做损失函数的时候，没有收敛，还缺少一个步骤
        featlkpp_loss = self.conv_out(featlkpp_2)

        feat16_3 = (w2[0, 0] * feat16_1 + w2[1, 0] * feat16_2 + w2[2, 0] * featlkpp_2) / (
                           w2[0, 0] + w2[1, 0] + w2[2, 0])
        feat16_3 = self.bifpn_convs(feat16_3)  # [4, 256, 24, 24]
        feat16_loss = self.conv_loss(feat16_3)

        feat8_3 = (w2[0, 1] * feat8_1 + w2[1, 1] * feat8_2 + w2[2, 1] *
                   F.interpolate(feat16_3, scale_factor=2, mode='bilinear')) / (
                          w2[0, 1] + w2[1, 1] + w2[2, 1])  # bilinear
        feat8_3 = self.bifpn_convs(feat8_3)  # [4, 256, 48, 48]
        feat8_loss = self.conv_loss(feat8_3)

        feat4_3 = (w2[0, 2] * feat4_1 + w2[1, 2] * feat4_2 + w2[2, 2] *
                   F.interpolate(feat8_3, scale_factor=2, mode='bilinear')) / (
                          w2[0, 2] + w2[1, 2] + w2[2, 2])  # bilinear
        feat4_3 = self.bifpn_convs(feat4_3)  # [4, 256, 96, 96]

        return feat4_3, feat4_1, [featlkpp_loss, feat16_loss, feat8_loss]

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


def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder_BiFPN_EaNet(num_classes, backbone, BatchNorm)
