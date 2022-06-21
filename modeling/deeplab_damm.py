import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp_origin import build_aspp
from modeling.decoder_origin import build_decoder
from modeling.backbone import build_backbone

class DeepLab(nn.Module):
    # def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
    #              sync_bn=True, freeze_bn=False):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=8,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__() # 自己搭建的网络Deeplab会继承nn.Module：
        if backbone == 'drn':  # 深度残差网络
            output_stride = 8  # 卷积输出时缩小的倍数  224/7=32

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d  # 每层进行归一化处理
        else:
            BatchNorm = nn.BatchNorm2d  # 数据的归一化处理   y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)  # 'resnet' 16 BatchNorm2d
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):  # input:torch.Size([1, 3, 512, 512])
        x, low_level_feat = self.backbone(input)  # x=[1, 2048, 32, 32] low_level_feat=[1, 256, 128, 128])
        x = self.aspp(x)  # x->[1,256,32,32]
        x = self.decoder(x, low_level_feat)  # x:[1,256,32,32] -> torch.Size([1, 8, 128, 128])
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # 4倍上采样 ->[1,8,512,512]

        return x

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
        modules = [self.aspp, self.decoder]
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
    model = DeepLab(backbone='resnet', output_stride=16)
    model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化
    input = torch.rand(1, 3, 512, 512)
    output = model(input)
    print(output.size())
