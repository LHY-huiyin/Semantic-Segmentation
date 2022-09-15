'''
New for ResNeXt:
1. Wider bottleneck
2. Add group for conv2
'''

import torch.nn as nn
import math

__all__ = ['ResNeXt', 'resnext18', 'resnext34', 'resnext50', 'resnext101',
           'resnext152']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):  # resnetxt18,34
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes*2, stride)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes*2, planes*2, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):  # resnetxt50,101,152
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(Bottleneck, self).__init__()
        # 1*1卷积
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*2)
        # 3*3卷积 组卷积
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        # 1*1卷积
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # 当stride!=1时，1*1卷积
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # torch.Size([1, 64, 128, 128])

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    """原本：添加了pretrained=True
    def __init__(self, block, layers, num_classes=8, num_group=32):  # resnetxt101 layer:[3, 4, 23, 3] 决定循环多少次
    """
    def __init__(self, block, layers, num_classes=8, num_group=32):  # resnetxt101 layer:[3, 4, 23, 3] 决定循环多少次
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)  # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0], num_group)
        self.layer2 = self._make_layer(block, 128, layers[1], num_group, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_group, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], num_group, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 当stride不为1   或者   输入通道数不等于中间通道数乘4（expansion）
            downsample = nn.Sequential(     # 残差连接
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),  # Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, num_group=num_group))  # 在第一个bock需要残差连接
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=num_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        feat2 = self.conv1(x)  # [4, 64, 256, 256]
        feat2 = self.bn1(feat2)
        feat2 = self.relu(feat2)
        feat2 = self.maxpool(feat2)  # [4, 64, 128, 128]

        feat4 = self.layer1(feat2)  # [4, 256, 128, 128]
        feat8 = self.layer2(feat4)  # [4, 512, 64, 64]
        feat16 = self.layer3(feat8)  # [4, 1024, 32, 32]
        feat32 = self.layer4(feat16)  # [4, 2048, 16, 16]

        return feat2, feat4, feat8, feat16, feat32

    """原本：
    def forward(self, x):
        x = self.conv1(x)  # torch.Size([1, 64, 256, 256])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # torch.Size([1, 64, 128, 128])

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # torch.Size([1, 1024, 32, 32])
        x = self.layer4(x)

        x = self.avgpool(x)  # torch.Size([1, 2048, 16, 16])
        x = x.view(x.size(0), -1)  # torch.Size([1, 204800])
        x = self.fc(x)

        return x
    """


def resnext18( **kwargs):
    """Constructs a ResNeXt-18 model.
    """
    model = ResNeXt(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnext34(**kwargs):
    """Constructs a ResNeXt-34 model.
    """
    model = ResNeXt(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnext50(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    model = ResNeXt(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnext101(**kwargs):
    # Constructs a ResNeXt-101 model.
    model = ResNeXt(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnext152(**kwargs):
    """Constructs a ResNeXt-152 model.
    """
    model = ResNeXt(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

if __name__ == "__main__":
    import torch

    # model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
    # resnext101(num_classes=args.num_class)
    model = resnext101(num_classes=8)
    input = torch.rand(4, 3, 512, 512)  # loveda的图片大小为：1024*1024
    # print('input`1',input)
    feat4, feat8, feat16, feat32 = model(input)
    print(feat4, feat8, feat16, feat32)  # torch.Size([1, 2048, 32, 32])