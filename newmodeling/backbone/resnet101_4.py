import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Bottleneck(nn.Module):
    expansion = 4  # 输出通道数是输入通道数的4倍

    #  两个1X1fliter分别用于降低和升高特征维度，主要目的是为了减少参数的数量，从而减少计算量，且在降维之后可以更加有效、直观地进行数据的训练和特征提取
    #   out = (in-kernel_size+2*padding)/stride + 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 BatchNorm=None):  # 传入的参数为：inplanes=64 planes=64  dilation=1
        # dilation=1这个参数决定了是否采用空洞卷积，默认为1（不采用）  从卷积核上的一个参数到另一个参数需要走过的距离
        super(Bottleneck, self).__init__()
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
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x  # torch.Size([1, 64, 128, 128])  torch.Size([1, 256, 128, 128])...  torch.Size([1, 1024, 32, 32])

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # torch.Size([1, 256, 128, 128])
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    # out = (in-kernel_size+2*padding)/stride + 1
    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True):  # block:Bottleneck  layers:[3, 4, 23, 3]
        self.inplanes = 64  # 每一个block的输入通道数目
        super(ResNet, self).__init__()  # 自己搭建的网络Resnet会继承nn.Module：
        blocks = [1, 2, 4]  # 仅用于第四层，另外，本文出现了同名的blocks，它是指循环次数，等于layer[*]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)  # -> [batch, planes,  h/2, w/2]
        # 输入通道为3，输出通道为64，输入尺寸为[1,3,512,512]--[1,64,256,256]
        # 高和宽的卷积核为7*7，所以，经过卷积，得到的高：(512-7+2*3)/2 +1 = 256
        # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = BatchNorm(
            64)  # BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) 64是通道数
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # -> [batch, planes,  h'/2, w'/2]
        # 最大池化：与卷积相似，同时，不是相乘，而是取最大值  [1,64,256,256]--[1,64,128,128]
        # 高和宽的卷积核为3*3，所以，经过卷积，得到的高：(256-3+2*1)/2 +1 = 128  在每个3*3的卷积核中选择最大值

        # 输入图片[1,64,128,128]  _make_layer()中的planes参数是“基准通道数”，不是输出通道数！！！
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0],
                                       BatchNorm=BatchNorm)  # 步长为1  空洞卷积为1  layer为3 改变了通道数 ->[batch, 256,  h, w]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1],
                                       BatchNorm=BatchNorm)  # 步长为2  空洞卷积为1  layer为4 -> [batch, 512,  h/2, w/2]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2],
                                       BatchNorm=BatchNorm)  # 步长为2  空洞卷积为1  layer为23 -> [batch, 1024,  h/2, w/2]
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3],
                                         BatchNorm=BatchNorm)  # 步长为1  空洞卷积为2 -> [batch, 1048,  h', w']
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self._init_weight()  # 初始化权重

        if pretrained:  # true
            self._load_pretrained_model()  # 加载权重

    # blocks = layers[*]   block = Bottleneck [1,64,128,128]
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    BatchNorm=None):  # blocks:3(进入循环多少次)  dilation:1  planes:64
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 当步长不为1或者输入通道数不等于输出乘扩张的 stride=1 inplanes=256 planes=64 block.expansion=4
            downsample = nn.Sequential(  # 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),  # -> [batch, planes*4,  h/stride, w/stride]
                #  此处的planes是传入的基准通道数
                BatchNorm(planes * block.expansion),  # BatchNorm(256)  256是通道数
            )  # 一个 BasicBlock的分支x要与output相加时，若x和output的通道数不一样，则要做一个downsample，
            # 剧透一下，在resnet里的downsample就是用一个1x1的卷积核处理，变成想要的通道数。
            # 为什么要这样做？因为最后要x要和output相加啊， 通道不同相加不了。所以downsample是专门用来改变x的通道数的。

        layers = []
        # 第一轮开始，因为stride是可变的此处，还加入了downsample
        layers.append(block(self.inplanes, planes, stride, dilation, downsample,
                            BatchNorm))  # 减少特征图尺寸: 后面的stride=2时此处和下面是不一样的，下面的stride 为1
        # 当步长为2时，结合卷积核分析，会减低2倍特征图尺寸
        self.inplanes = planes * block.expansion  # 由于第一轮中已经放大四倍了
        for i in range(1, blocks):  # 不包含blocks的值  不减少特征图尺寸：stride是用默认步长的，步长为1。
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):  # stride=1 dilation=2
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 当步长不为1或者输入通道数不等于输出乘扩张的
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),  # -> [batch, 2048,  h, w]
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation,
                            downsample=downsample,
                            BatchNorm=BatchNorm))  # dilation 会改变kernel_size的大小：k=k-(k-1)(r-1) r为dilation的大小
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i] * dilation,
                                BatchNorm=BatchNorm))  # dilation的大小会影响padding的大小，所以图片大小没有改变

        return nn.Sequential(*layers)

    def forward(self, input):  # 16倍
        x = self.conv1(input)  # [1,3,513,513]-[1,64,256,256]
        x = self.bn1(x)  # [1,64,256,256]
        x = self.relu(x)
        x = self.maxpool(x)  # [1,64,256,256]-[1,64,128,128]

        # x = self.layer1(x)  # [1,64,128,128]-[1,256,128,128]
        # low_level_feat = x
        # x = self.layer2(x)  # [1,256,128,128]-[1,512,64,64]
        # x = self.layer3(x)  # [1,512,64,64]-[1,1024,32,32]
        # x = self.layer4(x)  # [1,1024,32,32]-[1,2048,32,32]
        # return x, low_level_feat
        x1 = self.layer1(x)  # [4,64,128,128]-[4,256,128,128]
        x2 = self.layer2(x1)  # 4,256,128,128]-[4,512,64,64]
        x3 = self.layer3(x2)  # [4,512,64,64]-[4,1024,32,32]
        x4 = self.layer4(x3)  # [4,1024,32,32]-[4,2048,32,32]
        return [x1, x2, x3, x4]

    """Unet
    def forward(self, input):  # 16倍
        #  print('input2', input) 此处的值为传入值
        feat2 = self.conv1(input)  # [4,3,513,513]-[4,64,256,256]
        feat2 = self.bn1(feat2)  # [4,64,256,256]
        feat2 = self.relu(feat2)
        feat2 = self.maxpool(feat2)  # [4,64,256,256]-[4,64,128,128]

        # EaNet中写法改进一点，直观
        feat4 = self.layer1(feat2)  # [4,64,128,128]-[4,256,128,128]
        feat8 = self.layer2(feat4)  # 4,256,128,128]-[4,512,64,64]
        feat16 = self.layer3(feat8)  # [4,512,64,64]-[4,1024,32,32]
        feat32 = self.layer4(feat16)  # [4,1024,32,32]-[4,2048,32,32]

        return feat2, feat4, feat8, feat16, feat32  # (384->192)->96->48->24
    """

    """EaNet
    def forward(self, input):  # 16倍
        #  print('input2', input) 此处的值为传入值
        x = self.conv1(input)  # [4,3,513,513]-[4,64,256,256]
        x = self.bn1(x)  # [4,64,256,256]
        x = self.relu(x)
        x = self.maxpool(x)  # [4,64,256,256]-[4,64,128,128]

        # EaNet中写法改进一点，直观
        feat4 = self.layer1(x)  # [4,64,128,128]-[4,256,128,128]
        feat8 = self.layer2(feat4)  # 4,256,128,128]-[4,512,64,64]
        feat16 = self.layer3(feat8)  # [4,512,64,64]-[4,1024,32,32]
        feat32 = self.layer4(feat16)  # [4,1024,32,32]-[4,2048,32,32]
        return feat4, feat8, feat16, feat32
    """

    """ MASPP中返回的是列表
    def forward(self, input):  # 16倍
        #  print('input2', input) 此处的值为传入值
        x = self.conv1(input)  # [4,3,513,513]-[4,64,256,256]
        x = self.bn1(x)  # [4,64,256,256]
        x = self.relu(x)
        x = self.maxpool(x)  # [4,64,256,256]-[4,64,128,128]
        x1 = self.layer1(x)  # [4,64,128,128]-[4,256,128,128]
        x2 = self.layer2(x1)  # 4,256,128,128]-[4,512,64,64]
        x3 = self.layer3(x2)  # [4,512,64,64]-[4,1024,32,32]
        x4 = self.layer4(x3)  # [4,1024,32,32]-[4,2048,32,32]
        return [x1, x2, x3, x4]
    """

    """原本：
     #  print('input2', input) 此处的值为传入值
        x = self.conv1(input)  # [1,3,513,513]-[1,64,256,256]
        x = self.bn1(x)  # [1,64,256,256]
        x = self.relu(x)
        x = self.maxpool(x)  # [1,64,256,256]-[1,64,128,128]

        x = self.layer1(x)  # [1,64,128,128]-[1,256,128,128]
        low_level_feat = x
        x = self.layer2(x)  # [1,256,128,128]-[1,512,64,64]
        x = self.layer3(x)  # [1,512,64,64]-[1,1024,32,32]
        x = self.layer4(x)  # [1,1024,32,32]-[1,2048,32,32]
        return x, low_level_feat
    """

    def _init_weight(self):  # 初始化
        for m in self.modules():  # 对每一个符合Conv2d都需要初始化
            if isinstance(m, nn.Conv2d):  # isinstance():判断一个对象是否是一个已知的类型
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # n:3136
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)  # 将权重的数值大小填充为1
                m.bias.data.zero_()  # 将偏差的数值大小填充为0
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)  # 将权重的数值大小填充为1
                m.bias.data.zero_()  # 将偏差的数值大小填充为0
        # 先执行这一步 print("weight****")

    def _load_pretrained_model(self):  # 查看预训练模型的参数
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')  # 下载预训练权重resnet101
        # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')   # 下载预训练权重resnet50
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            # 再执行这一步  print(k)  # k：conv.weight bn.running_mean bn.running_var  bn.weight  bn.bias fc.weight  fc.bias
            if k in state_dict:
                model_dict[k] = v  # 新字典的key值对应的value为一一对应的值。
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)  # 重新加载这个模型。就可以将模型参数load进模型。
        # 执行这一步 print('model_pretrained****')


def ResNet101(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm,
                   pretrained=pretrained)  # Bottleneck输入模块   pretrained：True
    return model


"""
新增的resnet50
"""

def ResNet50(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm,
                   pretrained=pretrained)  # Bottleneck输入模块   pretrained：True
    return model


if __name__ == "__main__":
    import torch

    # model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
    model = ResNet101(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=16)
    input = torch.rand(1, 3, 512, 512)  # loveda的图片大小为：1024*1024
    # print('input`1',input)
    output, low_level_feat = model(input)
    print(output.size())  # torch.Size([1, 2048, 32, 32])
    print(low_level_feat.size())  # torch.Size([1, 256, 128, 128])
