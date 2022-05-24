# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# __all__ = ['ghost_net']  # 设置可被其他文件import的变量或函数。


def _make_divisible(v, divisor, min_value=None):    # divisor:4（固定）  v:24
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

    min_value: 是out_channel的最小值，
               如果int(v + divisor / 2) // divisor * divisor的结果小于min_value的话，
               out_channel的值就是min_value
    这个函数的作用就是对in_channel: v做一个最低限度的通道变换，使得mew_v可以被divisor整除
    e.g: 30 ---> 32 生成的通道数量能被 4 整除
    确保所有层的通道数能被8整除 16 24 40 80 112 160 960
    """
    if min_value is None:
        min_value = divisor   # min_value:4
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)   # new_v = max(4, (v + 2) // 4 * 4)  v=输入通道数*0.25
    # a = int(v + divisor / 2)// divisor      重点：整除！！！ v+2后无法整除4，取整然后再乘4《输出通道数确保被4整除》
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# 就是一个带有截断的sigmoid
def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)  # clamp_():将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn

        # 通道压缩， se_ratio=0.25
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)   # se_ratio=0.25 通道数压缩成原来的1/4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # wh变成1*1
        # 通道压缩为之前的1/4
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        # 激活函数
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)  # 自适应平均池化
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)  # 激活函数relu
        x_se = self.conv_expand(x_se)
        # x_se就是通道注意力
        x = x * self.gate_fn(x_se)   # 通道权重相乘   参数量增加 将1*1（注意力，权重大小）
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):  # 只传入inp,oup;影响的是中间层的大小init_channesl;变更的是逐通道卷积，即groups
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()

        # 这里采用了分离卷积
        # self.primary_conv： kernel_size = 1
        # self.cheap_operation： groups=init_channels

        self.oup = oup
        # 中间层的channel(是输入层的1 / 2)
        init_channels = math.ceil(oup / ratio)   # (16/2)=8 math.ceil():“向上取整”， 即小数部分直接舍去，并向正数部分进1
        # 输出层的channel
        new_channels = init_channels*(ratio-1)  # 8

        # 1×1的卷积用来降维
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),  # 改变通道数：输出通道数的一半
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # 3×3的分组卷积进行线性映射
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),   # 逐通道卷积
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        # ghost1: 输出通道A为隐藏输出通道数 进入后，输出B为A/2
        # ghost2: 输出通道A为输出通道数  进入后，输出B为A/2
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        # 将x1和x2沿着通道数堆叠
        out = torch.cat([x1, x2], dim=1)  # 前面将init_channels=oup / 2,此处将两个特征图进行叠加，增加通道数至原来大小

        # 只返回需要的通道数
        return out[:, :self.oup, :, :]


"""
Ghost bottleneck主要由两个堆叠的Ghost模块组成。
第一个Ghost模块用作扩展层，增加了通道数。这里将输出通道数与输入通道数之比称为expansion ratio。
第二个Ghost模块减少通道数，以与shortcut路径匹配。然后，使用shortcut连接这两个Ghost模块的输入和输出。
"""
class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()

        # 通道压缩系数
        has_se = se_ratio is not None and se_ratio > 0.  # false
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)   # 输出通道数为中间通道数

        # Depth-wise convolution
        if self.stride > 1:  # wh减半
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)  # 根据se_ratio=0.25做了一个压缩与扩展，以及激活的操作，通道数先减后增，最终不变
        else:  # false
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)   # 输出通道数为输出通道数

        # shortcut
        # 如果是以下的参数的话， 不需要使用额外的卷积层进行通道和尺寸的变换
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(  # 当in_chs /= out_chs时，调整residual大小
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),  # stride=2时：wh减半
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),    # 通道数改变
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        # 每一个block包含两个ghostmodule
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # 当stride=1时，x + (ghost module->ghost module->x1)
        # 当stride=2时，x + (ghost module->DWConv stride=2->ghost module->x2)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)  # 将对应像素值相加---不改变通道数

        return x


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=8, width=1.0, dropout=0.2):
        """
        width: 1.0
        dropout: 0.2
        """
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs  # 初始参数
        self.dropout = dropout  # 0.2

        # building first layer
        # 计算输出的channel大小
        output_channel = _make_divisible(16 * width, 4)
        # stem是起源的意思
        self.conv_stem = nn.Conv2d(8, output_channel, 3, 2, 1, bias=False)   # 指定了输入通道数：3   out=(in-3+2*1)/2+1=in/2  ->[b,16,w/2,h/2]
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck  # 网络大模块
        for cfg in self.cfgs:
            # 每一个layer就是一个stage
            layers = []
            """
            c: 控制输出层
            exp_size: 控制影藏层  
            """
            for k, exp_size, c, se_ratio, s in cfg:
                # print(k, exp_size, c, se_ratio, s)
                # 得到输出层和隐藏层的channel, 这些channel都要能被4整除
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                              se_ratio=se_ratio))
                # 更新下一个block的in_channel
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)         # exp_size=960
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        # ------主要的网络层--------
        x = self.blocks(x)
        # ------------------------
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)  # x.view(batchsize, -1)   batchsize指转换后有几行，而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)  # F.dropout:将输入Tensor的元素按伯努利分布随机置0
            # 直接将self.training传入函数，就可以在训练时应用dropout，评估时关闭dropout
        x = self.classifier(x)
        return x


def my_ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k, t, c, SE, s
        # 第二个值是t:exp:expasion size是中间隐藏层大小  hidden_channel = _make_divisible(exp_size * width, 4)
        # 第三个值是c:是输出大小  output_channel = _make_divisible(c * width, 4)
        # stage1        输入(2,8,224,224) -> (2,16,112,112)
        [[3,  16,  16, 0, 1]],  # (2,16,112,112) -> (2,16,112,112)
        # stage2
        [[3,  48,  24, 0, 2]],  # 原文中每一个stage的最后一个卷积是stride=2 (2,16,112,112)->(1,24,56,56)
        [[3,  72,  24, 0, 1]],  # (1,24,56,56) -> (2,24,56,56)
        # stage3
        [[5,  72,  40, 0.25, 2]],  # (2,24,56,56)->(2,40,28,28)
        [[5, 120,  40, 0.25, 1]],  # (2,40,28,28)->(2,40,28,28)
        # stage4
        [[3, 240,  80, 0, 2]],     # (2,40,28,28)->(2,80,14,14)
        [[3, 200,  80, 0, 1],      # (2,80,14,14)->(2,80,14,14)
         [3, 184,  80, 0, 1],      # (2,80,14,14)->(2,80,14,14)
         [3, 184,  80, 0, 1],      # (2,80,14,14)->(2,80,14,14)
         [3, 480, 112, 0.25, 1],    # (2,80,14,14)->(2,112,14,14)
         [3, 672, 112, 0.25, 1]  # 可以拆开，但是代码部分减少循环 (2,112,14,14)->(2,112,14,14)
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],      # (2,112,14,14)->(2,160,7,7)
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    return GhostNet(cfgs, **kwargs)


if __name__=='__main__':
    model = my_ghostnet()
    model.eval()
    # print(model)
    input = torch.randn(2, 8, 224, 224)
    y = model(input)
    print(y.size())  # torch.Size([32, 8])

# GhostNet(                 输入(2,8,224,224)
#   (conv_stem): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)   # (2,8,224,224)->(2,16,112,112)
#   (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (act1): ReLU(inplace=True)
#   (blocks): Sequential(                     k, t, c, SE, s
#     (0): Sequential(              stage1:[[3,  16,  16, 0, 1]]
#       (0): GhostBottleneck(
#                                                           residual =x (2,16,112,112)
#         (ghost1): GhostModule(
#           (primary_conv): Sequential(  # 正常的卷积：压缩特征图的通道
#             (0): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)   # (2,16,112,112)->(2,8,112,112)
#             (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(  # 逐通道卷积：            dw_size=3 dw_size//2=1  stride=1 wh不变
#             (0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)  # (2,8,112,112)->(2,8,112,112)
#             (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                 会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此： (2,16,112,112)
#         )
#         (ghost2): GhostModule(
#           (primary_conv): Sequential(                               # (2,16,112,112)->(2,8,112,112)
#             (0): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
#             (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=8, bias=False)  # (2,8,112,112)->(2,8,112,112)
#             (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                 会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,16,112,112)
#         )
#         (shortcut): Sequential()    # 因为in_chs=16 out_chs=16 且stride=1
#       )
#     )
#
#
#                                stage2:[[3,  48,  24, 0, 2]]  k, t, c, SE, s
#     (1): Sequential(              输入（2，16，112，112）
#       (0): GhostBottleneck(
#         (ghost1): GhostModule(
#           (primary_conv): Sequential(
#             (0): Conv2d(16, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)                    # (2，16，112，112)->(2,24,112,112)
#             (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)  # ->(2,24,112,112)
#             (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                       会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,48,112,112)
#         )
#               stride=2  Depth-wise convolution
#         (conv_dw): Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=48, bias=False)  ->(2,48,56,56)
#         (bn_dw): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (ghost2): GhostModule(
#           (primary_conv): Sequential(
#             (0): Conv2d(48, 12, kernel_size=(1, 1), stride=(1, 1), bias=False)            ->(2,12,56,56)
#             (1): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=12, bias=False)  ->(2,12,56,56)
#             (1): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,24,56,56)
#         )
#                     self.shortcut(residual)   residual=(2,16,112,112)
#         (shortcut): Sequential(  # 因为in_chs=16 out_chs=24 stride=2  wh减半
#           (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)     ->(2,16,56,56)
#           (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (2): Conv2d(16, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)                                 ->(2,24,56,56)
#           (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#                   x+=self.shortcut(residual)   (2,24,56,56)+(2,24,56,56)=(1,24,56,56)
#       )
#     )
#     (2): Sequential(              stage2.2: [[3,  72,  24, 0, 1]]
#       (0): GhostBottleneck(   #  输入：residual=x=(2,24,56,56)
#         (ghost1): GhostModule(
#           (primary_conv): Sequential(
#             (0): Conv2d(24, 36, kernel_size=(1, 1), stride=(1, 1), bias=False)                   (2,24,56,56)->(2,36,56,56)
#             (1): BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=36, bias=False)   ->(2,36,56,56)
#             (1): BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,72,56,56)
#         )
#         (ghost2): GhostModule(
#           (primary_conv): Sequential(
#             (0): Conv2d(72, 12, kernel_size=(1, 1), stride=(1, 1), bias=False)          (2,72,56,56)->(2,12,56,56)
#             (1): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=12, bias=False)  ->(2,12,56,56)
#             (1): BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,24,56,56)
#         )
#         (shortcut): Sequential()   # stride=1 且 输入输出都是24
#       )
#     )
#
#
#                           stage3：[[5,  72,  40, 0.25, 2]]   输入通道数：24 输出通道数：40
#     (3): Sequential(      #  输入：residual=x=(2,24,56,56)
#       (0): GhostBottleneck(
#         (ghost1): GhostModule(
#           (primary_conv): Sequential(
#             (0): Conv2d(24, 36, kernel_size=(1, 1), stride=(1, 1), bias=False)        (2,24,56,56)->(2，36，56，56)
#             (1): BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(36, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=36, bias=False)   ->(2,36,56,56)
#             (1): BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,72,56,56)
#         )
#                ******    stride=2  Depth-wise convolution wh减半
#         (conv_dw): Conv2d(72, 72, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=72, bias=False)  (2,72,56,56)->(2,72,28,28)
#         (bn_dw): BatchNorm2d(72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#                ******    se=0.25  通道注意力模块  x=(2,72,28,28)
#                *****   reduce_chs=_make_divisible(72*0.25,4)  new_v=max(4, (v + 2) // 4 * 4)) =(4,20)  =20  且大于72*0.25*0.9=16.
#         (se): SqueezeExcite(
#           (avg_pool): AdaptiveAvgPool2d(output_size=1)                     x_se = (2,72,1,1)  自适应算法能够自动帮助我们计算核的大小和每次移动的步长。
#           (conv_reduce): Conv2d(72, 20, kernel_size=(1, 1), stride=(1, 1))    x_se = (2,72,28,28)->(2,20,1,1)  压缩
#           (act1): ReLU(inplace=True)
#           (conv_expand): Conv2d(20, 72, kernel_size=(1, 1), stride=(1, 1))    x_se =  (2,20,28,28)->(2,72,1,1)  扩张
#                 ****   x = x * self.gate_fn(x_se)  gate_fn=hard_sigmoid=F.relu6(x + 3.) / 6.  参数量增加   (2,72,28,28)
#         )
#         (ghost2): GhostModule(
#           (primary_conv): Sequential(
#             (0): Conv2d(72, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)         (2,72,28,28)->(2,20,28,28)
#             (1): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=20, bias=False)     (2,20,28,28)->(2,20,28,28)
#             (1): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,40,28,28)
#         )
#         (shortcut): Sequential(       因为stride=2 (wh减半) in_chs =24 不等于 out_chs = 40  所以需要调整residual  residual=x=(2,24,56,56)
#           (0): Conv2d(24, 24, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=24, bias=False)        (2,24,56,56)->(2,24,28,28)
#           (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (2): Conv2d(24, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)                                   (2,24,28,28)—>(2,40,28,28)
#           (3): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#               **x += self.shortcut(residual)  (2,40,28,28) + (2,40,28,28) = (2,40,28,28)
#       )
#     )
#     (4): Sequential(              stage3.2:        [[5, 120,  40, 0.25, 1]]  输入通道数：40 输出通道数：40 stride=1
#       (0): GhostBottleneck(               #  输入：residual=x=(2,40,28,28)
#         (ghost1): GhostModule(
#           (primary_conv): Sequential(   # 中间层init_channels = math.ceil(oup / ratio) =
#             (0): Conv2d(40, 60, kernel_size=(1, 1), stride=(1, 1), bias=False)                (2,40,28,28)->(2,60,28,28) 改变通道数
#             (1): BatchNorm2d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(60, 60, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=60, bias=False)   (2,60,28,28)->(2,60,28,28)
#             (1): BatchNorm2d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,120,28,28)
#         )
#         (se): SqueezeExcite(  reduce_chs=_make_divisible(120*0.25=30,4)  new_v=max(4, (v30 + 2) // 4 * 4)) =(4,32)  =32  且大于120*0.25*0.9=27.
#           (avg_pool): AdaptiveAvgPool2d(output_size=1)
#           (conv_reduce): Conv2d(120, 32, kernel_size=(1, 1), stride=(1, 1))   **** x_se ****  (2,120,28,28)->(2,32,1,1)
#           (act1): ReLU(inplace=True)
#           (conv_expand): Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))   **** x_se ****   (2,32,28,28)->(2,120,1,1)
#                   **** x = x * self.gate_fn(x_se)  gate_fn = hard_sigmoid = F.relu6(x + 3.) / 6.  参数量增加   (2,120,28,28)
#         )
#         (ghost2): GhostModule(
#           (primary_conv): Sequential(
#             (0): Conv2d(120, 20, kernel_size=(1, 1), stride=(1, 1), bias=False)                       (2,120,28,28)->(2,20,28,28)
#             (1): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=20, bias=False)         (2,20,28,28)->(2,20,28,28)
#             (1): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此： (2,40,28,28)
#         )
#         (shortcut): Sequential()
#       )
#     )
#
#
#                               stage4:[[3, 240,  80, 0, 2]],   输入通道数：40 输出通道数：80 stride=2 要shortcut
#     (5): Sequential(          #  输入：residual = x = (2,40,28,28)
#       (0): GhostBottleneck(
#         (ghost1): GhostModule(  #ghost1: 输出通道A为隐藏输出通道数(=240)  进入后，输出B为A/2=120
#           (primary_conv): Sequential(
#             (0): Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)               (2,40,28,28)->(2,120,28,28)
#             (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(120, 120, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=120, bias=False)      (2,120,28,28)->(2,120,28,28)
#             (1): BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,240,28,28)
#         )
#               ******* stride=2  Depth-wise convolution wh减半
#         (conv_dw): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)       (2,240,28,28)->(2,240,14,14)
#         (bn_dw): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (ghost2): GhostModule(       # ghost2: 输出通道A为输出通道数(=80)  进入后，输出B为A/2=40
#           (primary_conv): Sequential(
#             (0): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)                                  (2,240,14,14)->(2,40,14,14)
#             (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)        (2,40,14,14)->(2,40,14,14)
#             (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,80,14,14)
#         )
#         (shortcut): Sequential(           输入通道数：40 输出通道数：80 stride=2 要shortcut  输入：residual = x = (2,40,28,28)
#           (0): Conv2d(40, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=40, bias=False)   residual= (2,40,28,28) ->(2,40,14,14)
#           (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (2): Conv2d(40, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)                              (2,40,14,14)->(2,80,14,14)
#           (3): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       *****  x += self.shortcut(residual)  (2,80,14,14) + (2,80,14,14) = (2,80,14,14)
#       )
#     )

#                           stage4.2:
#                     [[3, 200,  80, 0, 1],
#                      [3, 184,  80, 0, 1],
#                      [3, 184,  80, 0, 1],
#                      [3, 480, 112, 0.25, 1],
#                      [3, 672, 112, 0.25, 1]
#                     ],
#     (6): Sequential(                  [3, 200,  80, 0, 1]  输入通道数：80 输出通道数：80 stride=1
#       (0): GhostBottleneck(           #  输入：residual = x =(2,80,14,14)
#         (ghost1): GhostModule(    # ghost1:输出通道数A是隐藏层200，然后进入程序的输出通道数init_channel=oup/2=100
#           (primary_conv): Sequential(
#             (0): Conv2d(80, 100, kernel_size=(1, 1), stride=(1, 1), bias=False)       (2,80,14,14)->(2,100,14,14)
#             (1): BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(100, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=100, bias=False)  (2,100,14,14)->(2,100,14,14)
#             (1): BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,200,14,14)
#         )
#         (ghost2): GhostModule( # ghost2:输入通道数A是隐藏层200，输出通道数是输出层80，然后进入程序的输出通道数init_channel=oup/2=40
#           (primary_conv): Sequential(
#             (0): Conv2d(200, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)                   (2,200,14,14)->(2,40,14,14)
#             (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)   (2,40,14,14)->(2,40,14,14)
#             (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,80,14,14)
#         )
#         (shortcut): Sequential()
#       )
#
#       (1): GhostBottleneck(           [3, 184,  80, 0, 1],   输入通道数：80 输出通道数：80 stride=1
#         (ghost1): GhostModule(  # ghost1:输出通道数A是隐藏层184，然后进入程序的输出通道数init_channel=oup/2=92,因为会有cat操作
#           (primary_conv): Sequential(
#             (0): Conv2d(80, 92, kernel_size=(1, 1), stride=(1, 1), bias=False)    (2,80,14,14)->(2,92,14,14)
#             (1): BatchNorm2d(92, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(92, 92, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=92, bias=False)  (2,92,14,14)->(2,92,14,14)
#             (1): BatchNorm2d(92, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,184,14,14)
#         )
#         (ghost2): GhostModule(  # ghost2:输入通道数A是隐藏层184，输出通道数是输出层80，然后进入程序的输出通道数init_channel=oup/2=40
#           (primary_conv): Sequential(
#             (0): Conv2d(184, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)                   (2,184,14,14)->(2,40,14,14)
#             (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)  (2,40,14,14)->(2,40,14,14)
#             (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,80,14,14)
#         )
#         (shortcut): Sequential()  输入通道数：80 输出通道数：80 stride=1
#       )
#
#       (2): GhostBottleneck(               [3, 184,  80, 0, 1],    输入通道数：80 输出通道数：80 stride=1
#         (ghost1): GhostModule(# ghost1:输出通道数A是隐藏层184，然后进入程序的输出通道数init_channel=oup/2=92,因为会有cat操作
#           (primary_conv): Sequential(
#             (0): Conv2d(80, 92, kernel_size=(1, 1), stride=(1, 1), bias=False)               (2,80,14,14)->(2,92,14,14)
#             (1): BatchNorm2d(92, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(92, 92, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=92, bias=False)   (2,92,14,14)->(2,92,14,14)
#             (1): BatchNorm2d(92, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,80,14,14)
#         )
#         (ghost2): GhostModule(   # ghost2:输入通道数A是隐藏层184，输出通道数是输出层80，然后进入程序的输出通道数init_channel=oup/2=40
#           (primary_conv): Sequential(
#             (0): Conv2d(184, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)                     (2,184,14,14)->(2,40,14,14)
#             (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)  (2,40,14,14)->(2,40,14,14)
#             (1): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,80,14,14)
#         )
#         (shortcut): Sequential()
#       )
#
#       (3): GhostBottleneck(           [3, 480, 112, 0.25, 1],    输入通道数：80 输出通道数：112  stride=1
#         (ghost1): GhostModule(# ghost1:输出通道数A是隐藏层480，然后进入程序的输出通道数init_channel=oup/2=240,因为会有cat操作
#           (primary_conv): Sequential(
#             (0): Conv2d(80, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)                       (2,80,14,14)->(2,240,14,14)
#             (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)  (2,240,14,14)->(2,240,14,14)
#             (1): BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,480,14,14)
#         )
#         (se): SqueezeExcite(  x_se=(2,480,14,14)
#           (avg_pool): AdaptiveAvgPool2d(output_size=1)                          ->  (2,480,1,1)
#           (conv_reduce): Conv2d(480, 120, kernel_size=(1, 1), stride=(1, 1))    ->  (2,120,1,1)
#           (act1): ReLU(inplace=True)
#           (conv_expand): Conv2d(120, 480, kernel_size=(1, 1), stride=(1, 1))     ->  (2,480,1,1)
#               *****    x = x * self.gate_fn(x_se)    (2,480,14,14)
#         )
#         (ghost2): GhostModule(# ghost2:输入通道数A是隐藏层480，输出通道数是输出层112，然后进入程序的输出通道数init_channel=oup/2=56
#           (primary_conv): Sequential(
#             (0): Conv2d(480, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)              (2,480,14,14)->(2,56,14,14)
#             (1): BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(56, 56, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=56, bias=False) (2,56,14,14) ->(2,56,14,14)
#             (1): BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,112,14,14)
#         )
#         (shortcut): Sequential(   输入通道数：80 输出通道数：112 两者不相等  需要调整  residual = (2,80,14,14)
#           (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)     (2,80,14,14)->(2,80,14,14)
#           (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (2): Conv2d(80, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)                                (2,80,14,14)->(2,112,14,14)
#           (3): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#               ****    x += self.shortcut(residual)   (2,112,14,14)
#       )
#
#       (4): GhostBottleneck(               [3, 672, 112, 0.25, 1]  residual = x = (2,112,14,14)  输入：112 输出：112 stride=1
#         (ghost1): GhostModule(  # ghost1:输出通道数A是隐藏层672，然后进入程序的输出通道数init_channel=oup/2=336,因为会有cat操作
#           (primary_conv): Sequential(
#             (0): Conv2d(112, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)                  (2,112,14,14)->(2,336,14,14)
#             (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(336, 336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=336, bias=False)  (2,336,14,14)->(2,336,14,14)
#             (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,672,14,14)
#         )
#         (se): SqueezeExcite(                       x_se = (2,672,14,14)
#           (avg_pool): AdaptiveAvgPool2d(output_size=1)                    (2,672,14,14)->(2,672,1,1)
#           (conv_reduce): Conv2d(672, 168, kernel_size=(1, 1), stride=(1, 1))     ->(2,168,1,1)   压缩
#           (act1): ReLU(inplace=True)
#           (conv_expand): Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))     ->(2,672,1,1)   扩展
#               *****    x = x * self.gate_fn(x_se)  (2,672,14,14)
#         )
#         (ghost2): GhostModule(# ghost2:输入通道数A是隐藏层672，输出通道数是输出层112，然后进入程序的输出通道数init_channel=oup/2=56
#           (primary_conv): Sequential(
#             (0): Conv2d(672, 56, kernel_size=(1, 1), stride=(1, 1), bias=False)               (2,672,14,14)->(2,56,14,14)
#             (1): BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(56, 56, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=56, bias=False)   (2,56,14,14)->(2,56,14,14)
#             (1): BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,112,14,14)
#         )
#         (shortcut): Sequential()  输入：112 输出：112 stride=1
#       )
#     )
#
#
#                                   stage5:[[5, 672, 160, 0.25, 2]],
#     (7): Sequential(  # 输入  residual = x =  (2,112,14,14)
#       (0): GhostBottleneck(  输入通道：112 输出通道160 stride=2
#         (ghost1): GhostModule(# ghost1:输出通道数A是隐藏层672，然后进入程序的输出通道数init_channel=oup/2=336,因为会有cat操作
#           (primary_conv): Sequential(
#             (0): Conv2d(112, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)            (2,112,14,14)->(2,336,14,14)
#             (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(336, 336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=336, bias=False)   (2,336,14,14)->(2,336,14,14)
#             (1): BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,672,14,14)
#         )
#               *****  stride=2 wh减半
#         (conv_dw): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)    (2,672,14,14)->(2,672,7,7)
#         (bn_dw): BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (se): SqueezeExcite(     x_se = (2,672,7,7)     reduced_chs=727*0.25=168
#           (avg_pool): AdaptiveAvgPool2d(output_size=1)                                    ->(2,672,1,1)
#           (conv_reduce): Conv2d(672, 168, kernel_size=(1, 1), stride=(1, 1))              (2,672,1,1)->(2,168,1,1)
#           (act1): ReLU(inplace=True)
#           (conv_expand): Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))              (2,168,1,1)->(2,672,1,1)
#               *****   x = x * self.gate_fn(x_se)   (2,672,7,7)
#         )
#         (ghost2): GhostModule(  # ghost2:输入通道数A是隐藏层672，输出通道数是输出层160，然后进入程序的输出通道数init_channel=oup/2=80
#           (primary_conv): Sequential(
#             (0): Conv2d(672, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)            (2,672,7,7)->(2,80,7,7)
#             (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)    (2,80,7,7)->(2,80,7,7)
#             (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,160,7,7)
#         )
#         (shortcut): Sequential(   ***** 输入通道：112 输出通道160 stride=2    residual = (2,112,14,14)
#           (0): Conv2d(112, 112, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=112, bias=False)    (2,112,14,14)->(2,112,7,7)
#           (1): BatchNorm2d(112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#           (2): Conv2d(112, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)                                (2,112,7,7)->(2,160,7,7)
#           (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#           ****   x += self.shortcut(residual)             (2,160,7,7)
#       )
#     )
#                     stage5:
#                             [[5, 960, 160, 0, 1],
#                              [5, 960, 160, 0.25, 1],
#                              [5, 960, 160, 0, 1],
#                              [5, 960, 160, 0.25, 1]
#                             ]
#     (8): Sequential(             [5, 960, 160, 0, 1]   输入通道：160  输出通道：160 stride=1
#       (0): GhostBottleneck(    输入  residual = x  = (2,160,7,7)
#         (ghost1): GhostModule(  # ghost1:输出通道数A是隐藏层960，然后进入程序的输出通道数init_channel=oup/2=480,因为会有cat操作
#           (primary_conv): Sequential(
#             (0): Conv2d(160, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)              (2,160,7,7) ->(2,480,7,7)
#             (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)  (2,480,7,7)->(2,480,7,7)
#             (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,960,7,7)
#         )
#         (ghost2): GhostModule(  # ghost2:输入通道数A是隐藏层960，输出通道数是输出层160，然后进入程序的输出通道数init_channel=oup/2=80
#           (primary_conv): Sequential(
#             (0): Conv2d(960, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)               (2,960,7,7)->(2,80,7,7)
#             (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)     (2,80,7,7)->(2,80,7,7)
#             (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,160,7,7)
#         )
#         (shortcut): Sequential()  输入通道：160  输出通道：160 stride=1
#       )
#
#       (1): GhostBottleneck(     [5, 960, 160, 0.25, 1],  输入通道：160 输出通道：160  stride=1  residual = x = (2,160,7,7)
#         (ghost1): GhostModule( # ghost1:输出通道数A是隐藏层960，然后进入程序的输出通道数init_channel=oup/2=480,因为会有cat操作
#           (primary_conv): Sequential(
#             (0): Conv2d(160, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)           (2,160,7,7) ->(2,480,7,7)
#             (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)  (2,480,7,7)->(2,480,7,7)
#             (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,960,7,7)
#         )
#         (se): SqueezeExcite(  x_se=(2,960,7,7)     v=in_chs*se_ratio=960*0.25=240     reduced_chs = max(4, (v + 2) // 4 * 4)=242//2*4=60*4=240
#           (avg_pool): AdaptiveAvgPool2d(output_size=1)            (2,960,7,7)->(2,960,1,1)
#           (conv_reduce): Conv2d(960, 240, kernel_size=(1, 1), stride=(1, 1))      (2,960,1,1)->(2,240,1,1)
#           (act1): ReLU(inplace=True)
#           (conv_expand): Conv2d(240, 960, kernel_size=(1, 1), stride=(1, 1))     (2,240,1,1) ->(2,960,1,1)
#                *******     x = x * self.gate_fn(x_se)     (2,960,7,7)
#         )
#         (ghost2): GhostModule( # ghost2:输入通道数A是隐藏层960，输出通道数是输出层160，然后进入程序的输出通道数init_channel=oup/2=80
#           (primary_conv): Sequential(
#             (0): Conv2d(960, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)               (2,960,7,7)->(2,80,7,7)
#             (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)     (2,80,7,7)->(2,80,7,7)
#             (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,160,7,7)
#         )
#         (shortcut): Sequential()  输入通道：160 输出通道：160  stride=1
#       )
#
#       (2): GhostBottleneck(       [5, 960, 160, 0, 1],   输入通道：160 输出通道：160 stride=1
#         (ghost1): GhostModule(# ghost1:输出通道数A是隐藏层960，然后进入程序的输出通道数init_channel=oup/2=480,因为会有cat操作
#           (primary_conv): Sequential(
#             (0): Conv2d(160, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)          (2,160,7,7) ->(2,480,7,7)
#             (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)  (2,480,7,7)->(2,480,7,7)
#             (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,960,7,7)
#         )
#         (ghost2): GhostModule( # ghost2:输入通道数A是隐藏层960，输出通道数是输出层160，然后进入程序的输出通道数init_channel=oup/2=80
#           (primary_conv): Sequential(
#             (0): Conv2d(960, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)           (2,960,7,7)->(2,80,7,7)
#             (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False) (2,80,7,7)->(2,80,7,7)
#             (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,160,7,7)
#         )
#         (shortcut): Sequential()
#       )
#
#       (3): GhostBottleneck(       [5, 960, 160, 0.25, 1]   输入通道：160 输出通道：160 stride=1
#         (ghost1): GhostModule(# ghost1:输出通道数A是隐藏层960，然后进入程序的输出通道数init_channel=oup/2=480,因为会有cat操作
#           (primary_conv): Sequential(
#             (0): Conv2d(160, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)              (2,160,7,7)->(2,480,7,7)
#             (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)  (2,480,7,7)->(2,480,7,7)
#             (1): BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): ReLU(inplace=True)
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,960,7,7)
#         )
#         (se): SqueezeExcite(      x_se = (2,960,7,7)   reduced_chs = max(4, (v + 2) // 4 * 4) = (240+2)//4*4=60*4=240    in_chs * se_ratio =960*0.25=240
#           (avg_pool): AdaptiveAvgPool2d(output_size=1)                (2,960,7,7)->(2,960,1,1)
#           (conv_reduce): Conv2d(960, 240, kernel_size=(1, 1), stride=(1, 1))      (2,960,1,1)->(2,240,1,1)
#           (act1): ReLU(inplace=True)
#           (conv_expand): Conv2d(240, 960, kernel_size=(1, 1), stride=(1, 1))      (2,240,1,1)->(2,960,1,1)
#            *****        x = x * self.gate_fn(x_se)    (2,960,7,7)
#         )
#         (ghost2): GhostModule(  # ghost2:输入通道数A是隐藏层960，输出通道数是输出层160，然后进入程序的输出通道数init_channel=oup/2=80
#           (primary_conv): Sequential(
#             (0): Conv2d(960, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)               (2,960,7,7)->(2,80,7,7)
#             (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#           (cheap_operation): Sequential(
#             (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)  (2,80,7,7)->(2,80,7,7)
#             (1): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#             (2): Sequential()
#           )
#                ******   会有一个操作：cat(x1,x2) x1:primary_conv x2:cheap_operation  因此：    (2,160,7,7)
#         )
#         (shortcut): Sequential()
#       )
#     )
#
#       exp_size=960  hidden_channel=960   input_channel=160  dropout=0.2
#     (9): Sequential(
#       (0): ConvBnAct(
#         (conv): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)            (2,160,7,7)->(2,960,7,7)
#         (bn1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (act1): ReLU(inplace=True)
#       )
#     )
#   )
#   (global_pool): AdaptiveAvgPool2d(output_size=(1, 1))            (2,960,7,7)->(2,960,1,1)
#   (conv_head): Conv2d(960, 1280, kernel_size=(1, 1), stride=(1, 1))           (2,960,1,1)->(2,1280,1,1)
#   (act2): ReLU(inplace=True)
#       ****** x = x.view(x.size(0), -1)   torch.Size([2, 1280])
#       ****** x = F.dropout(x, p=self.dropout, training=self.training)     torch.Size([2, 1280])
#   (classifier): Linear(in_features=1280, out_features=8, bias=True)   # torch.Size([2, 8])
# )