from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _make_divisible(v, divisor, min_value=None):  # divisor:4（固定）  v:24
    if min_value is None:
        min_value = divisor  # min_value:4
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)  # new_v = max(4, (v + 2) // 4 * 4)  v=输入通道数*0.25
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
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)  # se_ratio=0.25 通道数压缩成原来的1/4
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
        x = x * self.gate_fn(x_se)  # 通道权重相乘   参数量增加 将1*1（注意力，权重大小）
        return x


class GhostModule(nn.Module):  # 只传入inp,oup;影响的是中间层的大小init_channesl;变更的是逐通道卷积，即groups
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()

        # 这里采用了分离卷积
        # self.primary_conv： kernel_size = 1
        # self.cheap_operation： groups=init_channels

        self.oup = oup
        # 中间层的channel(是输入层的1 / 2)
        init_channels = math.ceil(oup / ratio)  # (16/2)=8 math.ceil():“向上取整”， 即小数部分直接舍去，并向正数部分进1
        # 输出层的channel
        new_channels = init_channels * (ratio - 1)  # 8

        # 1×1的卷积用来降维
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),  # 改变通道数：输出通道数的一半
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # 3×3的分组卷积进行线性映射
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),  # 逐通道卷积
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
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)  # 输出通道数为中间通道数

        # Depth-wise convolution
        if self.stride > 1:  # wh减半
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)  # 根据se_ratio=0.25做了一个压缩与扩展，以及激活的操作，通道数先减后增，最终不变
        else:  # false
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)  # 输出通道数为输出通道数

        # shortcut
        # 如果是以下的参数的话， 不需要使用额外的卷积层进行通道和尺寸的变换
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(  # 当in_chs /= out_chs时，调整residual大小
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),  # stride=2时：wh减半
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),  # 通道数改变
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
    def __init__(self, num_classes=8, width=1, dropout=0.2):
        """
        width: 1.0
        dropout: 0.2
        """
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        # self.cfgs = cfgs  # 初始参数
        self.dropout = dropout  # 0.2

        # tblocks = [200, 184, 184, 480, 672]
        # cblocks = [80, 80, 80, 112, 112]
        # seblocks = [0, 0, 0, 0.25, 0.25]

        # building first layer
        # 计算输出的channel大小
        output_channel = _make_divisible(16 * width, 4)
        # stem是起源的意思
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)  # 指定了输入通道数：3   out=(in-3+2*1)/2+1=in/2  ->[b,16,w/2,h/2]

        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        # stages = []
        block = GhostBottleneck  # 网络大模块
        # in_chs, mid_chs, out_chs, dw_kernel_size = 3,stride = 1, act_layer = nn.ReLU, se_ratio = 0.
        self.layer1 = nn.Sequential(
            block(16, 16, 16, 3, 1, se_ratio=0),
            block(16, 48, 24, 3, 2, se_ratio=0),
            block(24, 72, 24, 3, 1, se_ratio=0),
        )
        self.layer2 = nn.Sequential(
            block(24, 72, 40, 5, 2, se_ratio=0.25),
            block(40, 120, 40, 5, 1, se_ratio=0.25),
        )
        self.layer3 = nn.Sequential(
            block(40, 240, 80, 3, 2, se_ratio=0),
            block(80, 200, 80, 3, 1, se_ratio=0),
            block(80, 184, 80, 3, 1, se_ratio=0),
            block(80, 184, 80, 3, 1, se_ratio=0),
            block(80, 480, 112, 3, 1, se_ratio=0.25),
            block(112, 672, 112, 3, 1, se_ratio=0.25),
        )
        self.layer4 = nn.Sequential(
            block(112, 672, 160, 5, 2, se_ratio=0.25),
            block(160, 960, 160, 5, 1, se_ratio=0),
            block(160, 960, 160, 5, 1, se_ratio=0.25),
            block(160, 960, 160, 5, 1, se_ratio=0),
            block(160, 960, 160, 5, 1, se_ratio=0.25),
        )


        self.init_weights()

    #     self.layer1 = self._make_layer(block, 16, 3, 16, 16, 0, 1)  # (2,16,112,112) -> (2,16,112,112)
    #     self.layer2 = self._make_layer(block, 16, 3, 48, 24, 0, 2)  # (2,16,112,112)->(1,24,56,56)
    #     self.layer3 = self._make_layer(block, 24, 3, 72, 24, 0, 1)  # (1,24,56,56) -> (2,24,56,56)
    #     self.layer4 = self._make_layer(block, 24, 5, 72, 40, 0.25, 2)  # (2,24,56,56)->(2,40,28,28)
    #     self.layer5 = self._make_layer(block, 40, 5, 120, 40, 0.25, 1)   # (2,40,28,28)->(2,40,14,14)
    #     self.layer6 = self._make_layer(block, 40, 3, 240, 80, 0, 2)  # (2,40,28,28)->(2,80,14,14)
    #     self.layer7 = self._make_MG_unit(block, 80, 3, tblocks=tblocks, cblocks=cblocks, seblocks=seblocks)   # (2,80,14,14)-> (2,112,14,14)
    #     self.layer8 = self._make_layer(block, 112, 5, 672, 160, 0.25, 2)   # (2,112,14,14)
    #
    # def _make_layer(self, block, input_channel,  kernerl_size, exp_size, c, se_ratio, stride=1, width=1):
    #     layers = []
    #     output_channel = _make_divisible(c * width, 4)
    #     hidden_channel = _make_divisible(exp_size * width, 4)
    #     layers.append(block(input_channel, hidden_channel, output_channel, kernerl_size, stride,
    #                   se_ratio=se_ratio))
    #     # 更新下一个block的in_channel
    #     input_channel = output_channel
    #     return nn.Sequential(*layers)
    #
    # def _make_MG_unit(self, block, input_channel,  kernerl_size, tblocks, cblocks, seblocks, stride=1, width=1):
    #     for i in range(0, len(tblocks)):
    #         layers = []
    #         output_channel = _make_divisible(cblocks[i] * width, 4)
    #         hidden_channel = _make_divisible(tblocks[i] * width, 4)
    #         layers.append(block(input_channel, hidden_channel, output_channel, kernerl_size, stride,
    #                             se_ratio=seblocks[i]))
    #         # 更新下一个block的in_channel
    #         input_channel = output_channel
    #
    #     return nn.Sequential(*layers)

    def init_weights(self):
        # 先读取下载的预训练的键，读取模型的键
        checkpoint = torch.load('F:\\Code\\Deeplabv3Plus\\pytorch-deeplab-dualattention\\test\\state_dict_73.98.pth')
        state_dict = OrderedDict()  # 对字典对象中的元素排序
        # convert data_parallal to model 改变键的名字    更改名：将下载的预训练的键进行改名，if判断语句有很多个，因为结构有变化
        i = 0
        for key in checkpoint:
            # 前24个
            if i in range(0, 6):
                # a = "backbone."
                # b = a + key
                # state_dict[b] = checkpoint[key]
                state_dict[key] = checkpoint[key]
            if i in range(6, 30):
                # a = "backbone.layer1.0"
                a = "layer1.0"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            if i in range(30, 72):
                a = "layer1.1"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            if i in range(72, 96):
                a = "layer1.2"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            if i in range(96, 142):
                a = "layer2.0"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            if i in range(142, 170):
                a = "layer2.1"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            if i in range(170, 212):
                a = "layer3.0"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            if i in range(212, 236):
                a = "layer3.1"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            if i in range(236, 260):
                a = "layer3.2"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            if i in range(260, 284):
                a = "layer3.3"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            if i in range(284, 324):
                a = "layer3.4"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            if i in range(324, 352):
                a = "layer3.5"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            if i in range(352, 398):
                a = "layer4.0"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            if i in range(398, 422):
                a = "layer4.1"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            if i in range(422, 450):
                a = "layer4.2"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            if i in range(450, 474):
                a = "layer4.3"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            if i in range(474, 502):
                a = "layer4.4"
                b = a + key[10:]
                state_dict[b] = checkpoint[key]
            i += 1

        # check loaded parameters and created model parameters  去掉module字符
        model_state_dict_ = self.state_dict()
        model_state_dict = OrderedDict()
        for key in model_state_dict_:
            model_state_dict[key] = model_state_dict_[key]

        # 检查权重格式  将不必要的键去掉
        for key in state_dict:
            if key in model_state_dict:
                if state_dict[key].shape != model_state_dict[key].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        key, model_state_dict[key].shape, state_dict[key].shape))
                    state_dict[key] = model_state_dict[key]
            else:
                state_dict.pop(key)
                print('Drop parameter {}.'.format(key))

        for key in model_state_dict:
            if key not in state_dict:
                print('No param {}.'.format(key))
                state_dict[key] = model_state_dict[key]

        # 将权重的key与model的key统一
        model_key = list(model_state_dict_.keys())
        pretrained_key = list(state_dict.keys())
        pre_state_dict = OrderedDict()
        for k in range(len(model_key)):
            pre_state_dict[model_key[k]] = state_dict[pretrained_key[k]]

        self.load_state_dict(pre_state_dict, strict=True)


    def forward(self, x):  # 32倍
        x = self.conv_stem(x)  # ->[8, 16, 512, 512]
        # torch.tensor(x)  仍旧是16位
        # torch.FloatTensor(x) 类型不一致，这是32位的输入函数
        # torch.tensor(x).to(torch.float32)
        # x.to(torch.float32)
        x = self.bn1(x)
        x = self.act1(x)

        x1 = self.layer1(x)   # [8, 16, 512, 512] -> [8, 24, 256, 256]
        x2 = self.layer2(x1)  # [8, 24, 256, 256] -> [8, 40, 128, 128]
        x3 = self.layer3(x2)  # [8, 40, 128, 128] -> [8, 112, 64, 64]
        x4 = self.layer4(x3)  # [8, 112, 64, 64] -> [8, 160, 32, 32]

        return [x1, x2, x3, x4]

"""
点流模块：
    x1 = self.layer1(x)  # (2,16,112,112) -> (2,24,56,56)
    low_level_feat = x1  # (2,24,56,56)
    x2 = self.layer2(x1)  # (2,24,56,56)->(2,40,28,28)
    x3 = self.layer3(x2)  # (2,40,28,28)->(2,112,14,14)
    x4 = self.layer4(x3)  # (2,112,14,14)->(2,160,7,7)
    
    return [x1,x2,x3,x4], low_level_feat
"""
"""
原本：
    def forward(self, x):  # 32倍
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.layer1(x)  # (2,16,112,112) -> (2,24,56,56)
        low_level_feat = x  # (2,24,56,56)
        x = self.layer2(x)  # (2,24,56,56)->(2,40,28,28)
        x = self.layer3(x)  # (2,40,28,28)->(2,112,14,14)
        x = self.layer4(x)  # (2,112,14,14)->(2,160,7,7)
        # x = self.layer5(x)  # (2,40,28,28)->(2,40,28,28)
        # x = self.layer6(x)  # (2,40,28,28)->(2,80,14,14)
        # x = self.layer7(x)  # (2,80,14,14) -> (2,112,14,14)
        # x = self.layer8(x)  # (2,112,14,14)->(2,160,7,7)

        return x, low_level_feat
"""
# def Ghostnet101(**kwargs):
#     cfgs = [
#         # k, t, c, SE, s
#         # 第二个值是t:exp:expasion size是中间隐藏层大小  hidden_channel = _make_divisible(exp_size * width, 4)
#         # 第三个值是c:是输出大小  output_channel = _make_divisible(c * width, 4)
#         # stage1
#         [[3,  16,  16, 0, 1]],
#         [[3,  48,  24, 0, 2]],  # 原文中每一个stage的最后一个卷积是stride=2
#         # stage2
#         [[3,  72,  24, 0, 1]],
#         [[5,  72,  40, 0.25, 2]],
#         # stage3
#         [[5, 120,  40, 0.25, 1]],
#         [[3, 240,  80, 0, 2]],
#         # stage4
#         [[3, 200,  80, 0, 1],
#          [3, 184,  80, 0, 1],
#          [3, 184,  80, 0, 1],
#          [3, 480, 112, 0.25, 1],
#          [3, 672, 112, 0.25, 1]  # 可以拆开，但是代码部分减少循环
#         ]
#         [[5, 672, 160, 0.25, 2]]
#     ]
#     model = GhostNet(**kwargs)
#     return model

if __name__ == '__main__':
    model = GhostNet()
    model.eval()
    # print(model)
    input = torch.randn(12, 3, 1024, 1024)
    # y = model(input)
    output, low_level_feat = model(input)
    # print(y.size())   # torch.Size([32, 8])
    print(output.size())  # torch.Size([2, 160, 7, 7])
    print(low_level_feat.size())  # torch.Size([2, 24, 56, 56])
