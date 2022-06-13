import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, bilinear=True):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':  # backbone = resnet
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        elif backbone == 'ghostnet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)  # low_level_inplanes = 256 -> [batch, 48, w, h]
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()

        """原本："""
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       # 改变通道数-> [batch, 256, h, w]
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       # -> [batch, 256, h, w]
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1,
                                                 stride=1))  # -> [batch, num_classes, h, w]
        self._init_weight()



    def forward(self, x, low_level_feat):  # x=[2, 256, 7, 7]  低层次特征：low_level_feat=[2, 24, 56, 56]
        low_level_feat = self.conv1(low_level_feat)  # 改变通道数：torch.Size([2, 48, 56, 56])
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        # print('low_level_feat.shape:',low_level_feat.shape)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)  # 8倍上采样 torch.Size([2, 256, 56, 56])
        x = torch.cat((x, low_level_feat), dim=1)  # 拼接：按维数1（列） torch.Size([2, 304, 56, 56])
        x = self.last_conv(x)  # torch.Size([2, 8, 56, 56])
        # print('x.shape:',x.shape)  #torch.Size([4, 8, 129, 129]
        return x


    def _init_weight(self):  # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)