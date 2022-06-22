import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PSPModule(nn.Module):  # PPM模块：四个空洞卷积，再上采样并融合
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(PSPModule, self).__init__()   # resnet的out_features=512
    # def __init__(self, features, out_features=80, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
    #     super(PSPModule, self).__init__()   # mobilenet的out_feature=80 * 4=320

        self.stages = []
        # 简易理解：由size大小执行4次
        # self.conv1 = self._make_stage(features, out_features, sizes[0], norm_layer)
        # self.conv2 = self._make_stage(features, out_features, sizes[1], norm_layer)
        # self.conv3 = self._make_stage(features, out_features, sizes[2], norm_layer)
        # self.conv4 = self._make_stage(features, out_features, sizes[3], norm_layer)

        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        # self.a = len(sizes)
        self.bottleneck = nn.Sequential(
            # nn.Conv2d(features + 4 * out_features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=1, padding=0, dilation=1,
                      bias=False),
            # norm_layer(out_features),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        # bn = norm_layer(out_features)
        bn = nn.BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats, m=None):   # torch.Size([1, 2048, 32, 32])   # mobilenet：torch.Size([4, 320, 32, 32])
        h, w = feats.size(2), feats.size(3)
        # 第一个是原始特征图
        priors = [feats]

        # 自适应平均池化->卷积操作->上采样
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]

        # 将以上五个特征图进行concat，然后再减少通道数
        bottle = self.bottleneck(torch.cat(priors, 1))

        return bottle



def PPM(backbone, output_stride, BatchNorm):
    return PSPModule(backbone, output_stride, BatchNorm)