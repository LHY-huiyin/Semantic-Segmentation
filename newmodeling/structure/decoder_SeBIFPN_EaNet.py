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
    def __init__(self, in_chan=2048, out_chan=256, ks_list=[1, 7, 5, 3], dilation=[1, 3, 3], mode='parallel',
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
        "1*1卷积"
        feat1 = self.conv1(x)  # [4,256,24,24]
        # 1*3 3*1 卷积 1*3 3*1 卷积 1*3 3*1 卷积 rate=1 rate=2 rate=3
        "双向不对称卷积，7*3 3*7 rate=1,2"
        feat2 = self.conv2(x)  # [4,256,24,24]
        # 3*3 卷积
        "双向不对称卷积，5*3 3*5 rate=1,3,3"
        feat3 = self.conv3(x)  # [4,256,24,24]
        # 1*1 卷积
        "3*3卷积"
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
        # self.ppm = PSPModule(in_chan, norm_layer=nn.BatchNorm2d, out_features=256)
        self.aspp = AsppLuo(in_chan=256, out_chan=256, mode='parallel', with_gp=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # x = self.ppm(x)
        x = self.aspp(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc11 = nn.Conv2d(out_planes, out_planes // ratio, 1, bias=False)
        self.fc12 = nn.Conv2d(out_planes // ratio, out_planes, 1, bias=False)

        self.fc21 = nn.Conv2d(out_planes, out_planes // ratio, 1, bias=False)
        self.fc22 = nn.Conv2d(out_planes // ratio, out_planes, 1, bias=False)
        self.relu1 = nn.ReLU(True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        avg_out = self.fc12(self.relu1(self.fc11(self.avg_pool(x))))
        max_out = self.fc22(self.relu1(self.fc21(self.max_pool(x))))
        out = avg_out + max_out
        del avg_out, max_out
        return self.sigmoid(out)


"解码器结构"
class Decoder_SEBiFPN_EaNet(nn.Module):
    def __init__(self, n_classes, low_chan=[1024, 512, 256, 64], num_classes=8, levels=4, init=0.5, eps=0.0001, *args, **kwargs):
        super(Decoder_SEBiFPN_EaNet, self).__init__()
        self.eps = eps
        self.levels = levels
        # weighted
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.relu1 = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 1).fill_(init))
        self.relu2 = nn.ReLU()
        self.conv_16_BR = ConvBNReLU(low_chan[0], 256, ks=3, padding=1)  # 1*1卷积，如果padding=1，特征图尺寸会改变
        self.conv_8_BR = ConvBNReLU(low_chan[1], 256, ks=3, padding=1)
        self.conv_4_BR = ConvBNReLU(low_chan[2], 256, ks=3, padding=1)
        self.conv_2_BR = ConvBNReLU(low_chan[3], 256, ks=3, padding=1)

        self.conv_fuse1 = ConvBNReLU(256, 256, ks=3, padding=1)
        self.conv_fuse2 = ConvBNReLU(256, 256, ks=3, padding=1)
        self.conv_fuse3 = ConvBNReLU(256, 64, ks=3, padding=1)

        self.conv1_2 = nn.Conv2d(low_chan[3], 256, kernel_size=1, bias=False)
        self.conv1_4 = nn.Conv2d(low_chan[2], 256, kernel_size=1, bias=False)
        self.conv1_8 = nn.Conv2d(low_chan[1], 256, kernel_size=1, bias=False)
        self.conv1_16 = nn.Conv2d(low_chan[0], 256, kernel_size=1, bias=False)

        # self.Poollayer = torch.nn.AvgPool2d(kernel_size=2, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
        # self.ppm = PSPModule(low_chan, norm_layer=nn.BatchNorm2d, out_features=256)


        self.conv_cat_group = ChannelAttention(512, 256)
        # self.conv_cat1 = nn.Conv2d(256 + 256, 256, kernel_size=1, bias=False)
        # self.conv_cat2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)


        self.sigmoid = nn.Sigmoid()
        self.conv_loss = ConvBNReLU(256, num_classes, kernel_size=1, bias=False)  # supervisor的输出
        self.bifpn_convs = BifpnConvs(256, 256, kernel_size=1, padding=0)  # 进行融合
        self.gate_conv = nn.Conv2d(256, 1, kernel_size=1)

        self.fuse = ConvBNReLU(64, 64, ks=3, padding=1)
        self.conv_out = nn.Conv2d(64, n_classes, kernel_size=1, bias=False)

        self.init_weight()

    def forward(self, feat2, feat4, feat8, feat16, feat_lkpp):  # feat_lkpp:[4, 256, 26, 26] feat16:[4,1024,24,24] feat8:[4,512,48,48] feat4:[4,256,96,96] feat2:[4,64,192,192]
        H, W = feat16.size()[2:]
        feat_lkpp_up = F.interpolate(feat_lkpp, (H, W), mode='bilinear', align_corners=True)  # [4, 256, 24, 24]
        # featlkpp_loss = self.conv_loss(feat_lkpp)
        "对每一个编码器输出的特征图进行1*1卷积"
        feat2_1 = self.conv1_2(feat2)
        feat4_1 = self.conv1_4(feat4)
        feat8_1 = self.conv1_8(feat8)
        feat16_1 = self.conv1_16(feat16)
        "对每一个编码器输出的特征图进行3*3卷积"
        feat16_3 = self.conv_16_BR(feat16)  # [4,1024,24,24] -> [4, 256, 24, 24]  3*3卷积
        feat8_3 = self.conv_8_BR(feat8)  # [4,512,48,48] -> [4, 256, 48, 48]
        feat4_3 = self.conv_4_BR(feat4)  # [4,256,96,96] -> [4, 256, 96, 96]
        feat2_3 = self.conv_2_BR(feat2)  # [4,64,192,192] -> [4,256,192,192]

        "sebifpn"
        # w1 = self.relu1(self.w1)  # [2,4]
        # w1 = w1 / torch.sum(w1, dim=0) + self.eps
        # w2 = self.relu2(self.w2)  # [3,3]
        # w2 = w2 / torch.sum(w2, dim=0) + self.eps
        "采用门控的权重分配形式"
        "调整为自上而下的融合方式,以获得更多的深层特征"
        "5融合模式：将层6的特征图双线性插值得到feat_lkpp,然后与feat16_1 concat"
        # 将feat_lkpp与feat16_1进行concat,然后进行卷积(1*1,3*3),sigmoid,全局平均池化,得到权重
        # feat16_w = torch.nn.functional.adaptive_avg_pool2d(
        #     self.sigmoid(self.conv_cat2(self.conv_cat1(torch.cat((feat_lkpp_up, feat16_1), dim=1)))), (1, 1))
        feat16_w = torch.nn.functional.adaptive_avg_pool2d(self.conv_cat_group(torch.cat((feat_lkpp_up, feat16_1), dim=1)), (1, 1))

        # 将权重应用于特征融合
        feat16_fuse1 = feat16_w * feat_lkpp_up + feat16_1 * (1 - feat16_w)
        # 经过1*1卷积,以及多尺度特征融合模块,最后经过1*1卷积,便是sebifpn的完整过程
        feat16_fuse1 = self.bifpn_convs(feat16_fuse1)

        "4融合模式：将层5的特征图双线性插值得到feat16_fuse1_up,然后与feat8_1 concat"
        H, W = feat8_1.size()[2:]
        feat16_fuse1_up = F.interpolate(feat16_fuse1, (H, W), mode='bilinear')
        # feat8_w = torch.nn.functional.adaptive_avg_pool2d(
        #     self.sigmoid(self.conv_cat2(self.conv_cat1(
        #         torch.cat((feat16_fuse1_up, feat8_1), dim=1)))), (1, 1))
        feat8_w = torch.nn.functional.adaptive_avg_pool2d(self.conv_cat_group(torch.cat((feat16_fuse1_up, feat8_1), dim=1)), (1,1))
        feat8_fuse1 = feat8_w * feat16_fuse1_up + feat8_1 * (1 - feat8_w)
        feat8_fuse1 = self.bifpn_convs(feat8_fuse1)

        "3融合模式：将层4的特征图双线性插值得到feat8_fuse1_up,然后与feat4_1 concat"
        H, W = feat4_1.size()[2:]
        feat8_fuse1_up = F.interpolate(feat8_fuse1, (H, W), mode='bilinear')
        # feat4_w = torch.nn.functional.adaptive_avg_pool2d(
        #     self.sigmoid(self.conv_cat2(self.conv_cat1(
        #         torch.cat((feat8_fuse1_up, feat4_1), dim=1)))), (1, 1))
        feat4_w = torch.nn.functional.adaptive_avg_pool2d(self.conv_cat_group(torch.cat((feat8_fuse1_up, feat4_1), dim=1)), (1, 1))
        feat4_fuse1 = feat4_w * feat8_fuse1_up + feat4_1 * (1 - feat4_w)
        feat4_fuse1 = self.bifpn_convs(feat4_fuse1)

        "2融合模式：将层4的特征图双线性插值得到feat4_fuse1_up,然后与feat2_1 concat"
        H, W = feat2_1.size()[2:]
        feat4_fuse1_up = F.interpolate(feat4_fuse1, (H, W), mode='bilinear')
        # feat2_w = torch.nn.functional.adaptive_avg_pool2d(
        #     self.sigmoid(self.conv_cat2(self.conv_cat1(
        #         torch.cat((feat4_fuse1_up, feat2_1), dim=1)))), (1, 1))
        feat2_w = torch.nn.functional.adaptive_avg_pool2d(self.conv_cat_group(torch.cat((feat4_fuse1_up, feat2_1), dim=1)), (1, 1))
        feat2_fuse1 = feat2_w * feat4_fuse1_up + feat2_1 * (1 - feat2_w)
        feat2_fuse1 = self.bifpn_convs(feat2_fuse1)

        # ****************************************************自下而上#
        # 自底向上融合,三个特征图的融合
        # 门控的实现：经过nn.Conv2d(256, 1, kernel_size=1)，以及sigmoid函数，作为一个权重
        "3融合模式：将层4的特征图进行门控权重的融合，其中，令feat4_fuse1为主，将feat2_fuse1进行最大池化后与feat4_1进行两个特征图的门控方式的融合"
        "之后，仍旧按照门控融合的方法进行两个特征图的融合权证配比（与上同）"
        gate_feat4 = self.sigmoid(self.gate_conv(feat4_fuse1))
        gate_feat4_other = self.sigmoid(self.gate_conv(feat4_1)) * feat4_1 + self.sigmoid(
            self.gate_conv(F.max_pool2d(feat2_fuse1, kernel_size=2))) * F.max_pool2d(feat2_fuse1, kernel_size=2)
        feat4_fuse2 = gate_feat4 * feat4_fuse1 + (1 - gate_feat4) * gate_feat4_other  # 此处搞错了，门控融合：应该是（1-G）（1+G）
        feat4_fuse2 = self.bifpn_convs(feat4_fuse2)
        # feat4_loss = self.conv_loss(feat4_fuse2)

        "4融合模式：将层5的特征图进行门控权重的融合，其中，令feat8_fuse1为主，将feat4_fuse1进行最近邻后与feat8_1进行两个特征图的门控方式的融合"
        "之后，仍旧按照门控融合的方法进行两个特征图的融合权证配比（与上同）"
        H, W = feat8_1.size()[2:]
        feat4_fuse2_up = F.interpolate(feat4_fuse2, (H, W), mode='nearest')
        gate_feat8 = self.sigmoid(self.gate_conv(feat8_fuse1))
        gate_feat8_other = self.sigmoid(self.gate_conv(feat8_1)) * feat8_1 + self.sigmoid(
            self.gate_conv(feat4_fuse2_up)) * feat4_fuse2_up
        feat8_fuse2 = gate_feat8 * feat8_fuse1 + (1 - gate_feat8) * gate_feat8_other
        feat8_fuse2 = self.bifpn_convs(feat8_fuse2)
        # feat8_loss = self.conv_loss(feat8_fuse2)

        "5融合模式：将层6的特征图进行门控权重的融合，其中，令feat8_fuse1为主，将feat8_fuse1进行最近邻后与feat16_1进行两个特征图的门控方式的融合"
        "之后，仍旧按照门控融合的方法进行两个特征图的融合权证配比（与上同）"
        H, W = feat16_1.size()[2:]
        feat8_fuse2_up = F.interpolate(feat8_fuse2, (H, W), mode='nearest')
        gate_feat16 = self.sigmoid(self.gate_conv(feat16_fuse1))
        gate_feat16_other = self.sigmoid(self.gate_conv(feat16_1)) * feat16_1 + self.sigmoid(
            self.gate_conv(feat8_fuse2_up)) * feat8_fuse2_up
        feat16_fuse2 = gate_feat16 * feat16_fuse1 + (1 - gate_feat16) * gate_feat16_other
        feat16_fuse2 = self.bifpn_convs(feat16_fuse2)
        # feat16_loss = self.conv_loss(feat16_fuse2)

        # ****************************************************自上而下#
        "上采样"
        # 将三个特征图进行简单粗暴地相加
        "融合方法：3*3卷积，直接相加，得到decoder4"
        feat_out = self.conv_fuse1(feat16_3 + feat16_fuse2 + feat_lkpp_up)

        "融合方式：将decoder4进行双线性插值，然后再与feat8_3、feat8_fuse2相加，得到decoder3"
        H, W = feat8_fuse2.size()[2:]
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear')
        feat_out = self.conv_fuse2(feat8_3 + feat8_fuse2 + feat_out)

        "融合方式：将decoder3进行双线性插值，然后再与feat4_3、feat4_fuse2相加，得到decoder2"
        H, W = feat4_fuse2.size()[2:]
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out = self.conv_fuse3(feat4_3 + feat4_fuse2 + feat_out)

        "经过3*3卷积和1*1卷积"
        logits = self.conv_out(self.fuse(feat_out))

        # return logits, [featlkpp_loss, feat16_loss, feat8_loss]
        return logits

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
    return Decoder_SEBiFPN_EaNet(num_classes, backbone, BatchNorm)
