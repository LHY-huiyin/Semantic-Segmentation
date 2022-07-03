import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.backbone import build_backbone
from configs import config_factory
from modeling.CBAM import *

cfg = config_factory['resnet_cityscapes']

class DC_Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, BatchNorm=None):  # 传入的参数为：inplanes=64 planes=64  dilation=1
        # dilation=1这个参数决定了是否采用空洞卷积，默认为1（不采用）  从卷积核上的一个参数到另一个参数需要走过的距离
        super(DC_Block, self).__init__()
        # conv1=conv2d(64,64, kernel_size=1,stride=1) 输入通道数为64，输出为64，卷积核大小为1，步长为1上下左右扫描皆为1（每次平移的间隔）
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # -> [batch, planes,  h, w]  主要是改变通道数
        self.bn1 = BatchNorm(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation,
                               bias=False)  # -> 1>[batch, planes,  h/stride, w/stride]:主要是stride，减半  2>[batch, planes, h, w]
        self.bn2 = BatchNorm(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1,
                               bias=False)  # -> [batch, planes * 4,  h', w']  主要是改变通道数
        self.bn3 = BatchNorm(planes * 4)

        self.relu = nn.ReLU(inplace=True)  # 激活函数： 当x>0时，y=x;当x<0时，y=0
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x  # torch.Size([1, 64, 128, 128])  torch.Size([1, 256, 128, 128])...  torch.Size([1, 1024, 32, 32])
        "1*1卷积"
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        "3*3卷积"
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        "1*1卷积"
        out = self.conv3(out)  # torch.Size([1, 256, 128, 128])
        out = self.bn3(out)

        "残差连接"
        out += residual
        out = self.relu(out)

        return out

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


class Decoder_MANet(nn.Module):
    def __init__(self, backbone='resnet', n_classes=8, low_chan=[2048, 1024, 512, 256], sync_bn=True, *args, **kwargs):
        super(Decoder_MANet, self).__init__()
        self.conv_fuse1 = ConvBNReLU(1024, 512, ks=3, padding=1)
        self.conv_fuse2 = ConvBNReLU(512, 256, ks=3, padding=1)
        self.conv_fuse3 = ConvBNReLU(256, 64, ks=3, padding=1)
        self.fuse = ConvBNReLU(64, 64, ks=3, padding=1)

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d  # 每层进行归一化处理
        else:
            BatchNorm = nn.BatchNorm2d  # 数据的归一化处理   y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

        "双注意力机制的添加--通道注意力机制和空间注意力机制的串联CBAM"
        self.cbam_32 = CBAM(low_chan[0])
        self.cbam_16 = CBAM(low_chan[1])
        self.cbam_8 = CBAM(low_chan[2])
        self.cbam_4 = CBAM(low_chan[3])
        "解码器添加一个残差连接模块，信息增强，不改变通道数"
        self.dcblock_32 = DC_Block(2048, 512, BatchNorm=BatchNorm)  # 相差四倍
        self.dcblock_16 = DC_Block(512, 128, BatchNorm=BatchNorm)
        self.dcblock_8 = DC_Block(256, 64, BatchNorm=BatchNorm)
        self.dcblock_4 = DC_Block(64, 16, BatchNorm=BatchNorm)
        self.conv_out = nn.Conv2d(64, n_classes, kernel_size=1, bias=False)

        self.cbam_conv = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.init_weight()

    def forward(self, feat4, feat8, feat16, feat32):  # feat32:[4, 2048, 24, 24] feat16:[4,1024,24,24] feat8:[4,512,48,48] feat4:[4,256,96,96]
        H, W = feat16.size()[2:]
        "对每一个编码器输出的特征图进行双注意力机制"
        feat32_low = self.cbam_32(feat32)   # [4, 2048, 24, 24] -> [4, 2048, 24, 24]
        # feat16_low = self.conv_16(feat16)  # [4,1024,24,24] -> [4, 256, 24, 24]  3*3卷积
        "双注意力机制"
        feat16_low = self.cbam_16(feat16)  # [4,1024,24,24] -> [4, 1024, 24, 24]
        # feat8_low = self.conv_8(feat8)  # [4,512,48,48] -> [4, 128, 48, 48]
        feat8_low = self.cbam_8(feat8)  # [4,512,48,48] -> [4, 512, 48, 48]
        # feat4_low = self.conv_4(feat4)  # [4,256,96,96] -> [4, 64, 96, 96]
        feat4_low = self.cbam_4(feat4)  # [4,256,96,96] -> [4, 256, 96, 96]

        "对低层特征图进行一个残差连接，信息增强DC_Block（不改变通道数）"
        feat32_low = self.dcblock_32(feat32_low)  # [4, 2048, 24, 24] -> [4, 2048, 24, 24]
        "对低层特征图进行2倍上采样"
        feat32_up = F.interpolate(feat32_low, (H, W), mode='bilinear',
                                     align_corners=True)  # [4, 2048, 24, 24] -> [4, 2048, 24, 24]
        feat32_up = self.cbam_conv(feat32_up)  # [4, 2048, 24, 24] -> [4, 1024, 24, 24]
        feat_out = self.conv_fuse1(feat16_low + feat32_up)  # [4, 2014, 24, 24] + [4, 1024, 24, 24] -> [4, 512, 24, 24] 直接相加，尺寸必须一致

        "对融合后的低层特征图进行一个残差连接，信息增强DC_Block（不改变通道数）"
        feat_out = self.dcblock_16(feat_out)  # [4, 512, 24, 24] -> [4, 512, 24, 24]
        H, W = feat8_low.size()[2:]  # 48 48
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear',
                                 align_corners=True)  # [4, 512, 48, 48]
        "卷积改变通道数"
        feat_out = self.conv_fuse2(feat_out + feat8_low)  # -> [4, 512, 48, 48]+[4, 512, 48, 48] -> [4, 512,48,48] -> [4, 256, 48, 48]

        "对融合后的低层特征图进行一个残差连接，信息增强DC_Block（不改变通道数）"
        feat_out = self.dcblock_8(feat_out)  # [4, 256, 48, 48] -> [4, 256, 48, 48]
        H, W = feat4_low.size()[2:]  # 96, 96
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear',
                                 align_corners=True)  # [4, 256, 96, 96]
        feat_out = self.conv_fuse3(feat_out + feat4_low)   # -> [4, 256, 96, 96] + [4, 256, 96, 96] -> [4, 64, 48, 48]

        "对融合后的低层特征图进行一个残差连接，信息增强DC_Block（不改变通道数）"
        feat_out = self.dcblock_4(feat_out)  # [4, 64, 96, 96] -> [4, 64, 96, 96]
        logits = self.conv_out(self.fuse(feat_out))  # [4,8,96,96]
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

class DeepLab_MANet(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=8,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab_MANet, self).__init__()  # 自己搭建的网络Deeplab会继承nn.Module：
        if backbone == 'drn':  # 深度残差网络
            output_stride = 8  # 卷积输出时缩小的倍数  224/7=32

        # 引入输入通道
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        elif backbone == 'resnet':
            inplanes = 2048  # backbone = 'resnet'
        elif backbone == 'ghostnet':
            inplanes = 160
        elif backbone == 'xception':
            inplanes = 2048
        else:
            raise NotImplementedError

        if backbone == 'resnet' or backbone == 'drn':  # backbone = resnet
            low_level_inplanes = 512
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 512
        elif backbone == 'ghostnet':
            low_level_inplanes = 512
        else:
            raise NotImplementedError

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d  # 每层进行归一化处理
        else:
            BatchNorm = nn.BatchNorm2d  # 数据的归一化处理   y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)  # 'resnet' 16 BatchNorm2d
        self.decoder = Decoder_MANet(backbone, cfg.n_classes, low_chan=[2048, 1024, 512, 256])

        self.freeze_bn = freeze_bn

    def forward(self, x):  #
        H, W = x.size()[2:]  # 256 256
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)  # resnet:feat32:[4,2048,24,24] feat16:[4,1024,24,24] feat8:[4,512,48,48] feat4:[4,256,96,96]
        logits = self.decoder(feat4, feat8, feat16, feat32)  # feat_lkpp:[4, 256, 26, 26] feat16:[4,1024,24,24] feat8:[4,512,48,48] feat4:[4,256,96,96]
        logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=True)  # [4, 8, 96, 96]-># [4, 8, 384, 384]

        return logits

    # Given groups=1, weight of size [256, 2048, 3, 3],
    # 代表卷积核的channel 大小为 2048->256 ，大小为3*3
    # expected input[1, 512, 32, 32] to have 2048 channels,
    # 代表现在要卷积的feature的大小，channel为512, 其实我们期望的输入feature大小channel 为2048个通道
    # but got 512 channels instead
    # 代表我们得到512个channels 的feature 与预期2048个channel不一样。

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:  # freeze_bn=false
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


if __name__ == "__main__":
    # model = DeepLab(backbone='mobilenet', output_stride=16)
    model = DeepLab_MANet(backbone='ghostnet', output_stride=16)
    model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化
    input = torch.rand(2, 3, 224, 224)  # RGB是三通道
    output = model(input)
    print(output.size())
