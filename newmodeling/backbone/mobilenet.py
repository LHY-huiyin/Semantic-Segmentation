import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import torch.utils.model_zoo as model_zoo

def conv_bn(in_channels, out_channels, stride, BatchNorm):
    return nn.Sequential(  # out_channels=32(初始指定),pad=1,stride=2,n=1
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),  # ->[n,32,h/2,w/2]  [1,32,112,112]
        BatchNorm(out_channels),
        nn.ReLU6(inplace=True)
    )


def fixed_padding(inputs, kernel_size, dilation):   # 值得学习：当stride=1时，而kernel_size=3，这种做法不会使wh改变
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)  # 新的卷积核大小
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    # F.pad是pytorch内置的tensor扩充函数，便于对数据集图像或中间层特征进行维度扩充
    # p2d = (左边填充数，右边填充数，上边填充数，下边填充数)扩充维度，用于预先定义出某维度上的扩充参数
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))  # （1，1，1，1）
    # 当dilation=1:kernel_size=1 pad_total=0 pad_beg=0 pad_end=0 （0，0，0，0）
    # 当dilation=1:kernel_size=3 pad_total=2 pad_beg=1 pad_end=1 （1，1，1，1）  (默认kernel_size=3)
    # 当dilation=2:kernel_size=3 pad_total=4 pad_beg=2 pad_end=2 （2，2，2，2）

    return padded_inputs


class InvertedResidual(nn.Module):  # 倒置的残差连接
    def __init__(self, in_channels, out_channels, stride, dilation, expand_ratio, BatchNorm):
        super(InvertedResidual, self).__init__()
        self.stride = stride  # 1
        assert stride in [1, 2]

        hidden_dim = round(in_channels * expand_ratio)  # 中间层维度 in_channels=32  expand_ratio=1（t）
        self.use_res_connect = self.stride == 1 and in_channels == out_channels  # 是否残差连接 (self.stride == 1) and (in_channels == out_channels)
        self.kernel_size = 3
        self.dilation = dilation  # 1 2

        if expand_ratio == 1:  # expand_ratio=t:输入通道的倍增系数（即中间部分的通道数是输入通道数的多少倍）
            self.conv = nn.Sequential(
                # dw  逐通道卷积 3*3卷积 升维  groups就是实现depthwise conv的关键
                # groups=hidden_dim 将输入的每一个通道作为一组，然后分别对其卷积，输出通道数为k，最后再将每组的输出串联，最后通道数为in_channels*K
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear 1*1卷积 降维+linear（不添加relu激活） width multiplier参数来做通道的缩减
                # 将原始维度增加到15或者30再做Relu的输入时，输出回复到原始维度后基本不会丢失很多信息
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, 1, 1, bias=False),   # h*w*(out_channels)
                BatchNorm(out_channels),
            )
        else:  # t=6 -> hidden_dim=6*in_channels
            self.conv = nn.Sequential(
                # pw ： Pointwise Convolution 逐点卷积 1*1卷积 ***窄***
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, 1, bias=False),  # h*w*(hidden_dim)
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),  # 结果是把小于0的变成0，大于6的取6
                # dw  3*3卷积 逐通道卷积  ***宽***
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),  # (h/s)*(w/s)*(hidden_dim)
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear  1*1卷积（不添加relu激活）  ***窄***
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, 1, bias=False),  # (h/s)*(w/s)*(out_channels)
                BatchNorm(out_channels),
            )

    def forward(self, x):
        x_pad = fixed_padding(x, self.kernel_size, dilation=self.dilation)
        if self.use_res_connect:  # 如果self.stride = 1 以及 in_channels = out_channels
            x = x + self.conv(x_pad)  # 使用残差连接  将输出与输入相加
        else:
            x = self.conv(x_pad)   # if决定的：是否残差，而它们都是要做expand_ratio = 6
        return x

"""原本"""
class MobileNetV2(nn.Module):
    def __init__(self, output_stride=8, BatchNorm=None, width_mult=1., pretrained=True):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        current_stride = 1
        rate = 1
        interverted_residual_setting = [
            # t:输入通道的倍增系数（即中间部分的通道数是输入通道数的多少倍）
            # s:该模块第一次重复时的 stride（后面重复都是 stride 1）
            # n:该模块重复次数  c:输出通道数
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)  # input_channel=32 width_mult=1
        self.features = [conv_bn(3, input_channel, 2, BatchNorm)]  # [1,3,224,224] -> [1,32,112,112]

        current_stride *= 2  # 2
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            if current_stride == output_stride:  # @ 当current_stride=16后，stride固定为1，dilation会改变
                # ￥ 若output_stride=8时，stride固定为1，dilation会改变
                stride = 1  # ￥ s=2 1 2 1
                dilation = rate  # @ 1 1 2    #￥ 1 2 2 4
                rate *= s  # @ 1 2 2  #￥ 2 2 4 4
            else:  # current_stride=2 output_stride=16
                stride = s  # 1 2 2 2
                dilation = 1
                current_stride *= s  # 2 4 8 16
            output_channel = int(c * width_mult)   # 16 24 32 64 & 96 160 320
            for i in range(n):  # n为重复次数 1 2 3 4 3 3 1 1
                if i == 0:  # 与ResNet类似，每层Bottleneck单独处理，指定stride。此层外的stride均为1
                    self.features.append(block(input_channel, output_channel, stride, dilation, t, BatchNorm))
                else:  # 这些叠加层stride均为1，in_channels = out_channels
                    self.features.append(block(input_channel, output_channel, 1, dilation, t, BatchNorm))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        self._initialize_weights()

        if pretrained:
            self._load_pretrained_model()

        self.low_level_features = self.features[0:4]  # 0,1,2,3
        self.high_level_features = self.features[4:]  # 4,5,6,7,8,9,10,11,12,13,14,15,16,17

    def forward(self, x):
        low_level_feat = self.low_level_features(x)   # torch.Size([1, 24, 56, 56])  获得特征图
        x = self.high_level_features(low_level_feat)  # torch.Size([1, 320, 14, 14])
        return x, low_level_feat

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://jeff95.me/models/mobilenet_v2-6a65762b.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    # 初始化权重操作
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

"""
class MobileNetV2(nn.Module):
    def __init__(self, output_stride=8, BatchNorm=None, width_mult=1., pretrained=True):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        current_stride = 1
        rate = 1
        interverted_residual_setting = [
            # t:输入通道的倍增系数（即中间部分的通道数是输入通道数的多少倍）
            # s:该模块第一次重复时的 stride（后面重复都是 stride 1）
            # n:该模块重复次数  c:输出通道数
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)  # input_channel=32 width_mult=1
        self.features = [conv_bn(3, input_channel, 2, BatchNorm)]  # [1,3,224,224] -> [1,32,112,112]

        current_stride *= 2  # 2
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            if current_stride == output_stride:  # @ 当current_stride=16后，stride固定为1，dilation会改变
                # ￥ 若output_stride=8时，stride固定为1，dilation会改变
                stride = 1  # ￥ s=2 1 2 1
                dilation = rate  # @ 1 1 2    #￥ 1 2 2 4
                rate *= s  # @ 1 2 2  #￥ 2 2 4 4
            else:  # current_stride=2 output_stride=16
                stride = s  # 1 2 2 2
                dilation = 1
                current_stride *= s  # 2 4 8 16
            output_channel = int(c * width_mult)   # 16 24 32 64 & 96 160 320
            for i in range(n):  # n为重复次数 1 2 3 4 3 3 1 1
                if i == 0:  # 与ResNet类似，每层Bottleneck单独处理，指定stride。此层外的stride均为1
                    self.features.append(block(input_channel, output_channel, stride, dilation, t, BatchNorm))
                else:  # 这些叠加层stride均为1，in_channels = out_channels
                    self.features.append(block(input_channel, output_channel, 1, dilation, t, BatchNorm))
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        self._initialize_weights()

        if pretrained:
            self._load_pretrained_model()

        # self.low_level_features = self.features[0:4]  # 0,1,2,3
        # self.high_level_features = self.features[4:]  # 4,5,6,7,8,9,10,11,12,13,14,15,16,17
        self.layer1 = self.features[0:2]  # [B,16,256,256]
        self.layer2 = self.features[2:4]  # [B,24,128,128]
        self.layer3 = self.features[4:7]  # [B,32,64,64]
        self.layer4 = self.features[7:18]  # [B,320,32,32]

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://jeff95.me/models/mobilenet_v2-6a65762b.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    # 初始化权重操作
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
"""

if __name__ == "__main__":
    input = torch.rand(1, 3, 512, 512)
    model = MobileNetV2(output_stride=8, BatchNorm=nn.BatchNorm2d)  # out_stride 意味着输出的特征图是16倍降采样的结果
    output, low_level_feat = model(input)  # low_level_feat : torch.Size([1, 24, 56, 56]) output : torch.Size([1, 320, 14, 14])
    print(output.size())  # torch.Size([1, 320, 14, 14])
    print(low_level_feat.size())  # torch.Size([1, 24, 56, 56])


# Sequential(
#   (0): Sequential(     # [1,3,224,224] -> [1,32,112,112]  wh减半
#     (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)   # (224-3+2)/2+1=112
#     (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU6(inplace=True)
#   )
#                   经过pad后torch.Size([1, 32, 114, 114])
#   (1): InvertedResidual(   # 32 -> 16 stride=1  n=1  wh不变
#     (conv): Sequential(
#       (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), groups=32, bias=False)           # [1,32,112,112]  逐通道卷积 (114-3)/1+1=112 32个特征图
#       (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)                      # [1,16,112,112]        16个32通道的kernel_size
#       (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#                经过pad后torch.Size([1, 32, 114, 114])
#   (2): InvertedResidual(  # 16->24 stride=2 n=2  wh减半
#     (conv): Sequential(
#       (0): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)                      # [1,16,112,112] -> [1,96,114,114]
#       (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), groups=96, bias=False)           # [1,96,114,114] -> [1,96,56，56]
#       (4): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)                      # [1,96,56，56] -> [1,24,56,56]
#       (7): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#               经过pad后torch.Size([1, 24, 58, 58])
#   (3): InvertedResidual(    #  stride=1  做跳跃连接（in_channels=out_channels stride=1）
#     (conv): Sequential(
#       (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1,24,58,58] -> [1,144,58,58]
#       (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), groups=144, bias=False)        # [1,144,56,56] -> [1,144,56,56]   (58-3)/1+1=56
#       (4): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1,144,56,56] -> [1,24,56,56]
#       (7): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#                               残差连接  [1,24,56,56]
#                 经过pad后torch.Size([1, 24, 58, 58])
#   (4): InvertedResidual(    # 24->32 stride=2 n=3  wh减半
#     (conv): Sequential(
#       (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1,24,58,58] -> [1,144,58,58]
#       (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), groups=144, bias=False)        # [1,144,58,58] -> [1,144,28,28]     (58-3)/2+1=28.5
#       (4): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1,144,28,28] -> [1,32,28,28]
#       (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#                 经过pad后torch.Size([1, 32, 30, 30])
#   (5): InvertedResidual(  # stride=1 做跳跃连接（in_channels=out_channels stride=1）
#     (conv): Sequential(
#       (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 32, 30, 30] -> [1,192,30,30]
#       (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), groups=192, bias=False)        # [1,192,30,30] -> [1,192,28,28]         (30-3)/1+1=28
#       (4): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)                    # [1,192,28,28] -> [1,32,28,28]
#       (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#                               残差连接  torch.Size([1, 32, 28, 28])
#                 经过pad后torch.Size([1, 32, 30, 30])
#   (6): InvertedResidual(  # stride=1 做跳跃连接（in_channels=out_channels stride=1）
#     (conv): Sequential(
#       (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 32, 30, 30] -> [1,192,30,30]
#       (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), groups=192, bias=False)        # [1,192,30,30] -> [1,192,28,28]         (30-3)/1+1=28
#       (4): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1,192,28,28] -> [1,32,28,28]
#       (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#                               残差连接  torch.Size([1, 32, 28, 28])
#                 经过pad后torch.Size([1, 32, 30, 30])
#   (7): InvertedResidual(  # 32->64 stride=2 n=4  wh减半  1/4
#     (conv): Sequential(
#       (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 32, 30, 30] -> [1,192,30,30]
#       (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), groups=192, bias=False)        # [1,192,30,30] -> [1,192,14,14]   (30-3)/2+1=14.5
#       (4): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1,192,14,14] -> [1,64,14,14]
#       (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#                 经过pad后torch.Size([1, 64, 16, 16])
#   (8): InvertedResidual(  # stride=1  做跳跃连接（in_channels=out_channels stride=1）2/4
#     (conv): Sequential(
#       (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 64, 16, 16] ->[1, 384, 16, 16]
#       (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), groups=384, bias=False)        # [1, 384, 16, 16] -> [1,384,14,14]      (16-3)/1+1=14
#       (4): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1,384,14,14] -> [1,64,14,14]
#       (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#                               残差连接[1,64,14,14]
#                 经过pad后torch.Size([1, 64, 16, 16])
#   (9): InvertedResidual(  # stride=1  做跳跃连接（in_channels=out_channels stride=1） 3/4
#     (conv): Sequential(
#       (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)                     #  [1, 64, 16, 16] ->[1, 384, 16, 16]
#       (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), groups=384, bias=False)        # [1, 384, 16, 16] -> [1,384,14,14]
#       (4): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1,384,14,14] -> [1,64,14,14]
#       (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#                               残差连接torch.Size([1, 64, 14, 14])
#                 经过pad后torch.Size([1, 64, 16, 16])
#   (10): InvertedResidual(  # stride=1  做跳跃连接（in_channels=out_channels stride=1） 4/4
#     (conv): Sequential(
#       (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 64, 16, 16] -> [1, 384, 16, 16]
#       (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), groups=384, bias=False)        # [1, 384, 16, 16] -> [1,384,14,14]
#       (4): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1,384,14,14] -> [1,64,14,14]
#       (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#                 经过pad后torch.Size([1, 64, 16, 16])
#   (11): InvertedResidual(  # 64->96 n=3 here:stride=1    dilation会变(最后一个)  1/3
#     (conv): Sequential(
#       (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 64, 16, 16] -> [1, 384, 16, 16]
#       (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), groups=384, bias=False)        # [1, 384, 16, 16] ->  [1, 384, 14, 14]
#       (4): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 384, 14, 14] -> [1,96,14,14]
#       (7): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#                 经过pad后torch.Size([1, 64, 16, 16])
#   (12): InvertedResidual(  #  跳跃连接（stride=1）  2/3
#     (conv): Sequential(
#       (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 96, 16, 16] -> [1, 576, 16, 16]
#       (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), groups=576, bias=False)        # [1, 576, 16, 16] -> [1, 576, 14, 14]
#       (4): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 576, 14, 14] -> [1, 96, 14, 14]
#       (7): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#                       跳跃连接torch.Size([1, 96, 14, 14])
#                 经过pad后torch.Size([1, 96, 16, 16])
#   (13): InvertedResidual(  #  跳跃连接（stride=1）  3/3
#     (conv): Sequential(
#       (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 96, 16, 16] -> [1, 576, 16, 16]
#       (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), groups=576, bias=False)        # [1, 576, 16, 16] -> [1, 576, 14, 14]
#       (4): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 576, 14, 14] -> [1, 96, 14, 14]
#       (7): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#                       跳跃连接torch.Size([1, 96, 14, 14])
#                 经过pad后torch.Size([1, 96, 16, 16])
#   (14): InvertedResidual(    # 96->160 n=3 跳跃连接   1/3
#     (conv): Sequential(
#       (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 96, 16, 16] -> [1, 576, 16, 16]
#       (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), groups=576, bias=False)        # [1, 576, 16, 16] -> [1, 576, 14, 14]
#       (4): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 576, 14, 14] -> [1, 160, 14, 14]
#       (7): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#                       跳跃连接torch.Size([1, 160, 14, 14])
#                 经过pad后torch.Size([1, 160, 16, 16])
#   (15): InvertedResidual(  #  跳跃连接（stride=1） 2/3
#     (conv): Sequential(
#       (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 160, 16, 16] -> [1, 960, 16, 16]
#       (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), groups=960, bias=False)        # [1, 960, 16, 16] -> [1, 960, 14, 14]
#       (4): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 960, 14, 14] -> [1, 160, 14, 14]
#       (7): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#                       跳跃连接torch.Size([1, 160, 14, 14])
#                 经过pad后torch.Size([1, 160, 16, 16])
#   (16): InvertedResidual(  #  跳跃连接（stride=1） 3/3
#     (conv): Sequential(
#       (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 160, 16, 16] -> [1, 960, 16, 16]
#       (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), groups=960, bias=False)        # [1, 960, 16, 16] -> [1, 960, 14, 14]
#       (4): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1, 960, 14, 14] -> [1, 160, 14, 14]
#       (7): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#                 经过pad后torch.Size([1, 160, 18, 18])  (dilation=2 pad=(2,2,2,2))
#   (17): InvertedResidual(  # 160->320  n=1 dilation=(2, 2) 跳跃连接 stride=1 wh不变
#     (conv): Sequential(
#       (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)                     #  [1, 160, 18, 18] -> [1,160,18,18]
#       (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (2): ReLU6(inplace=True)
#       (3): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), dilation=(2, 2), groups=960, bias=False)  # [1,160,18,18] -> [1,160,14,14]  k=k+(k-1)(d-1)=5 (18-5)/1+1=14
#       (4): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (5): ReLU6(inplace=True)
#       (6): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)                     # [1,160,14,14] -> [1,320,14,14]
#       (7): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
# )
