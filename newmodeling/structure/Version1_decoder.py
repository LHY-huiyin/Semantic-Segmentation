import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from newmodeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from newmodeling.attention.PPM import *
from newmodeling.conv.ConvBNReLU import *
from newmodeling.newidea.aspp_luo import *
from newmodeling.attention.PAM_CAM import *
from newmodeling.conv.decoderblock import *
from newmodeling.conv.DoubleConv import *

class BifpnConvs(nn.Module):
    def __init__(self, in_chan, out_chan, ks=1, stride=1, padding=0, dilation=1, *args, **kwargs):
        super(BifpnConvs, self).__init__()
        self.conv1 = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              dilation=dilation,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        # self.ppm = PSPModule(in_chan, norm_layer=nn.BatchNorm2d, out_features=256)
        self.aspp = AsppLuo(out_chan, out_chan, mode='parallel', with_gp=True)
        self.conv2 = nn.Conv2d(out_chan,
                               out_chan,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               dilation=dilation,
                               bias=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)  #[b,1024, 24,24]
        x = self.bn(x)
        x = self.relu(x)

        # x = self.ppm(x)
        x = self.aspp(x)   #[b,1024, 26,26]

        x = self.conv2(x)  #[b,1024,26,26]
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
        self.relu1 = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        avg_out = self.fc12(self.relu1(self.fc11(self.avg_pool(x))))
        max_out = self.fc22(self.relu1(self.fc21(self.max_pool(x))))
        out = avg_out + max_out
        del avg_out, max_out
        return x * self.sigmoid(out)


"解码器结构"
class Decoder_SEBiFPN_EaNet(nn.Module):
    def __init__(self, n_classes, low_chan=[1024, 512, 256, 64], num_classes=8, levels=4, init=0.5, eps=0.0001, *args, **kwargs):
        super(Decoder_SEBiFPN_EaNet, self).__init__()
        self.eps = eps
        self.levels = levels
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.relu1 = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 1).fill_(init))
        self.relu2 = nn.ReLU()

        filters = [256, 512, 1024, 2048]
        self.attention4 = PAM_CAM_Layer(filters[3])
        self.attention3 = PAM_CAM_Layer(filters[2])
        self.attention2 = PAM_CAM_Layer(filters[1])
        self.attention1 = PAM_CAM_Layer(filters[0])

        self.decoder4 = DoubleConv(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])

        self.conv_fuse3 = ConvBNReLU(filters[2], 256, ks=3, padding=1)
        self.conv_fuse2 = ConvBNReLU(256, 64, ks=3, padding=1)

        # self.channel = ChannelAttention(filters[3]*2, filters[3])
        # self.fuse4 = torch.nn.functional.adaptive_avg_pool2d((1, 1))
        # self.fuse3 = torch.nn.functional.adaptive_avg_pool2d(ChannelAttention(filters[2]*2, filters[2]))
        # self.fuse2 = torch.nn.functional.adaptive_avg_pool2d(ChannelAttention(filters[1]*2, filters[0]))

        self.bifpn4 = BifpnConvs(filters[2], filters[2], kernel_size=1, stride=1, padding=0)
        self.bifpn3 = BifpnConvs(filters[1], filters[1], kernel_size=3, stride=1, padding=2)
        self.bifpn2 = BifpnConvs(filters[0], filters[0], kernel_size=3, stride=1, padding=2)

        self.bifpn3up = BifpnConvs(filters[2], filters[2], kernel_size=3, stride=1, padding=1)
        self.bifpn2up = BifpnConvs(filters[1], filters[1], kernel_size=3, stride=1, padding=1)

        self.up1 = nn.Conv2d(filters[0], filters[1], kernel_size=1, stride=1, padding=0)
        self.up2 = nn.Conv2d(filters[1], filters[2], kernel_size=1, stride=1, padding=0)
        self.final = nn.Conv2d(filters[2], 256, kernel_size=1, stride=1, padding=0)

        self.fuse = ConvBNReLU(filters[0], 64, ks=3, padding=1)
        self.conv_out = nn.Conv2d(64, n_classes, kernel_size=1, bias=False)

        self.init_weight()

    def forward(self, x1, x2, x3, x4):  # feat_lkpp:[4, 256, 26, 26] feat16:[4,1024,24,24] feat8:[4,512,48,48] feat4:[4,256,96,96] feat2:[4,64,192,192]
        # assert len(x1, x2, x3, x4) == self.levels
        # build top-down and down-top path with stack
        # levels = self.levels

        # weighted
        # w relu
        w1 = self.relu1(self.w1)  # 0.5 [[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5]]
        w1 = w1 / torch.sum(w1, dim=0) + self.eps  # normalize  维度为0：因为w1为[2,4],维度0即为2：跨行求和，torch.sum(w1, dim=0)=[1,1,1,1]  [[0.4999,0.4999,0.4999,0.4999,0.4999],[0.4999,0.4999,0.4999,0.4999,0.4999]]
        w2 = self.relu2(self.w2)  # 0.5 [[0.5,0.5],[0.5,0.5],[0.5,0.5]]  tensor([[0.3333, 0.3333],[0.3333, 0.3333], [0.3333, 0.3333]])
        w2 = w2 / torch.sum(w2, dim=0) + self.eps  # normalize  tensor([[0.3333, 0.3333],[0.3333, 0.3333], [0.3333, 0.3333]])

        e4 = self.attention4(x4)      # [B,2048,24,24]
        e3 = self.attention3(x3)     #[B,1024, 24, 24]
        e2 = self.attention2(x2)    #[B,512, 48, 48]
        e1 = self.attention1(x1)    #[b, 256, 96,96]

        "e4经过3*3卷积，改变通道数，从2048调整为1024，便于与后面的特征图做相加"
        d4 = self.decoder4(e4)  # 2048 - [B,1024,24,24]

        # weight4 = self.fuse4(torch.cat(d4, e3), dim=1)  #2048*2
        # d4_add = weight4 * d4 + e3 * (1 - weight4)
        d3_add = (w1[0, 0] * d4 + e3 * w1[1, 0]) / (w1[0, 0] + w1[1, 0] + self.eps)  #相加必须是维度一致,大小一致  1024  48  [b,1024,24,24]
        d3_fuse = self.bifpn4(d3_add)   # d3_fuse: [b,1024, 24,24]

        "d3_fuse经过1*1卷积->反卷积->1*1卷积，得到的尺寸大小增加两倍，通道数减少一半"
        d3 = self.decoder3(d3_fuse)   # d3: [B, 512, 48, 48]

        # weight3 = self.fuse3(torch.cat(d3, e2), dim=1)
        # d3_add = (weight3 * d3 + e2 * (1 - weight3))
        d2_add = (w1[0, 1] * d3 + e2 * w1[1, 1]) / (w1[0, 1] + w1[1, 1] + self.eps)  # [b,512,48,48]
        d2_fuse = self.bifpn3(d2_add)  # d2_fise: [b,512,48,48]

        d2 = self.decoder2(d2_fuse)  #d2:[2, 256, 96, 96]

        "采用门控的权重分配形式"
        "调整为自上而下的融合方式,以获得更多的深层特征"
        # weight2 = self.fuse2(torch.cat(d2, e1), dim=1)
        "将权重应用于特征融合"
        # d2_add = weight2 * d2 + e1 * (1 - weight2)
        d1_add = (w1[0,2] * d2 + e1 * w1[1, 2]) / (w1[0, 2] + w1[1, 2] + self.eps)  #[2, 256, 96, 96]
        "经过1*1卷积,以及多尺度特征融合模块,最后经过1*1卷积,便是sebifpn的完整过程"
        d1_fuse = self.bifpn2(d1_add)  #[2, 256, 96, 96]

        # 自底向上融合,三个特征图的融合
        # 门控的实现：经过nn.Conv2d(256, 1, kernel_size=1)，以及sigmoid函数，作为一个权重
        d1_fuse_up = self.up1(d1_fuse)  #[2, 512, 96, 96]
        dup2 = (w2[0, 0] * F.max_pool2d(d1_fuse_up, kernel_size=2) + w2[1, 0] * d2_fuse +
                w2[2, 0] * e2) / (w2[0,0] + w2[1,0] + w2[2,0] + self.eps)  # 维度都必须为512  [B, 512,48,48]
        dup2_fuse = self.bifpn2up(dup2)   # [B,512,48,48]

        dup2_fuse_up = self.up2(dup2_fuse)#[b,1024,48,48]
        dup3 = (w2[0,1] * F.max_pool2d(dup2_fuse_up, kernel_size=2) + w2[1,1] * d3_fuse +
                w2[2, 1] * e3) / (w2[0,1] + w2[1,1] + w2[2,2] + self.eps)  # 维度都为1024   [2, 1024,24,24]
        dup3_fuse = self.bifpn3up(dup3)  # [n,1024,24,24]

        "上采样"
        "将特征图进行简单粗暴地相加"
        # deco3 = self.conv_fuse3(d4 + dup3_fuse)  #[b,256,24,24]

        deco3 = self.decoder3(d4 + dup3_fuse)  #[b,512,48,48]
        # H, W = dup2_fuse.size()[2:]
        # deco3 = F.interpolate(deco3, (H, W), mode='bilinear') #[b,256,48,48]

        # deco2 = self.conv_fuse2(deco3 + dup2_fuse)
        deco2 = self.decoder2(deco3 + dup2_fuse)  # [b,256,96.96]

        logits = self.conv_out(self.fuse(deco2))  # [b,8,96,96]

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
