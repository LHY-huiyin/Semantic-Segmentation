import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from newmodeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from newmodeling.attention.sebiASPP import *


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
        # self.ppm = PSPModule(in_chan, norm_layer=nn.BatchNorm2d, out_features=256)
        self.sebifpnASPP = SeBiFPNASPP(in_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # x = self.ppm(x)
        x = self.sebifpnASPP(x)

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

    def forward(self, feat2, feat4, feat8, feat16, feat32):  # feat_lkpp:[4, 256, 26, 26] feat16:[4,1024,24,24] feat8:[4,512,48,48] feat4:[4,256,96,96] feat2:[4,64,192,192]
        H, W = feat16.size()[2:]
        feat32 = F.interpolate(feat32, (H, W), mode='bilinear', align_corners=True)
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

        featlkpp_2 = (w1[0, 3] * feat16_2 + w1[1, 3] * feat32) / (
                w1[0, 3] + w1[1, 3] + self.eps)
        featlkpp_2 = self.bifpn_convs(featlkpp_2)  # [4, 256, 24, 24]
        # 做损失函数的时候，没有收敛，还缺少一个步骤
        featlkpp_loss = self.conv_loss(featlkpp_2)

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
