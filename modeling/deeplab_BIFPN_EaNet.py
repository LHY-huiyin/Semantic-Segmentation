import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.backbone import build_backbone
from modeling.decoder_BIFPN_EaNet import *
from configs import config_factory

cfg = config_factory['resnet_cityscapes']

"总体架构"
class DeepLab_EaNet_BIFPN(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=8,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab_EaNet_BIFPN, self).__init__()  # 自己搭建的网络Deeplab会继承nn.Module：
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
        self.lkpp = LKPP(in_chan=2048, out_chan=256, mode='parallel', with_gp=cfg.aspp_global_feature)
        self.decoder = Decoder_BiFPN_EaNet(cfg.n_classes, low_chan=[1024, 512, 256, 64])
        self.conv_out = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        self.Softmax = nn.Softmax()
        self.CoefRefine = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, bias=False),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=False)
        )
        self.freeze_bn = freeze_bn

    def forward(self, x):
        H, W = x.size()[2:]  # 256 256
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)  # resnet:feat32:[4,2048,24,24] feat16:[4,1024,24,24] feat8:[4,512,48,48] feat4:[4,256,96,96]
        feat_lkpp = self.lkpp(feat32)  # [4, 256, 26, 26]
        p2_out, p2_in = self.decoder(feat2, feat4, feat8, feat16, feat_lkpp)  # p2_in p2_out:[4,256,96,96]

        "进行边界上采样"
        p2_out_1 = self.conv_out(p2_out)  # [4, 8, 96,96]
        p2_out_2 = self.Softmax(p2_out_1)  # [4, 8, 96,96]
        coef = torch.max(p2_out_2)
        # p2_t = torch.cat((p2_out, p2_in), dim=1)
        # p2_t = p2_t * (1 - coef)
        p2_t = torch.cat((p2_out, p2_in), dim=1) * (1 - coef)  # [4,512,96,96]
        coef_out = self.CoefRefine(p2_t)  # [4, 64, 96, 96]
        logits = p2_out_1 * coef + coef_out * (1 - coef)
        logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=True)
        # logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=True)  # [4, 8, 96, 96]-># [4, 8, 384, 384]
        # logits = 1
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
        modules = [self.lkpp, self.decoder]
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
    model = DeepLab_EaNet_BIFPN(backbone='resnet', output_stride=16)
    model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化
    input = torch.rand(4, 3, 384, 384)  # RGB是三通道
    output = model(input)
    print(output.size())
