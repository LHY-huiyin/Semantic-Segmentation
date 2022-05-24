import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,  # k=2*5-3=7 out=[in-7(k)+6(padding))/1(stride)]+1=in    out = in
                                            stride=1, padding=padding, dilation=dilation, bias=False)  # 空洞卷积  -> [batch, 256,  h, w]
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)  # -> [batch, 256,  h, w]
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        elif backbone == 'ghostnet':
            inplanes = 160
        else:
            inplanes = 2048  # backbone = 'resnet'

        if output_stride == 16:  # output_stride = 16
            dilations = [1, 6, 12, 18]
            # dilations = [1, 3, 5]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0],
                                 BatchNorm=BatchNorm)  # padding=0 dilation=1   -> [batch, 256,  h, w]
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1],
                                 BatchNorm=BatchNorm)  # padding=6 dilation=6  -> [batch, 256,  h, w]
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2],
                                 BatchNorm=BatchNorm)  # padding=12 dilation=12 out=in -> [batch, 256,  h, w]
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        # MSPP
        # 膨胀后卷积核尺寸 = 膨胀系数 * (原始卷积核尺寸 - 1) + 1  = 2x(3-1)+1 =5
        # k=2*5-3=7 out=[in-k+2*padding]/stride+1=[(in-7+6)/1]+1=in    out = in
        # self.aspp1 = nn.Sequential(nn.Conv2d(inplanes, 256, 1, stride=1, padding=0, dilation=dilations[0], bias=False),  # conv1*1
        #                             BatchNorm(256),
        #                             nn.ReLU())   # con1*1   out=[in-1+2*0]/1+1=in  ->[b,256,h,w]
        # self.aspp2 = nn.Sequential(nn.Conv2d(inplanes, 256, 3, stride=1, padding=dilations[0], dilation=dilations[0], bias=False),  # conv3*3 rate=1
        #                             BatchNorm(256),
        #                             nn.ReLU())   # con3*3 rate=1  k=3 out=[in-3+2]/1+1=in     ->[b,256,h,w]
        # self.aspp3 = nn.Sequential(nn.Conv2d(inplanes, 256, 3, stride=1, padding=dilations[0], dilation=dilations[0], bias=False),  # conv3*3 rate=1
        #                             BatchNorm(256),
        #                             nn.ReLU(),
        #                             nn.Conv2d(256, 256, 3, stride=1, padding=dilations[1], dilation=dilations[1], bias=False),  # conv3*3 rate=3
        #                             BatchNorm(256),
        #                             nn.ReLU())   # con3*3 rate=1 -> conv3*3 rate=3    rate=3:k=3*(3-1)+1=5 out=[in-5+6]/1+1=in  ->[b,256,h,w]
        # self.aspp4 = nn.Sequential(nn.Conv2d(inplanes, 256, 3, stride=1, padding=dilations[0], dilation=dilations[0], bias=False),  # conv3*3 rate=1
        #                            BatchNorm(256),
        #                            nn.ReLU(),
        #                            nn.Conv2d(256, 256, 3, stride=1, padding=dilations[1], dilation=dilations[1], bias=False),  # conv3*3 rate=3
        #                            BatchNorm(256),
        #                            nn.ReLU(),
        #                            nn.Conv2d(256, 256, 3, stride=1, padding=dilations[2], dilation=dilations[2], bias=False),  # conv3*3 rate=5
        #                            BatchNorm(256),
        #                            nn.ReLU(),)  # con3*3 rate=1 -> conv3*3 rate=3 -> conv3*3 rate=5     rate=5:k=5*(3-1)+1=11 out=[in-11+5*2]/1+1=in  ->[b,256,h,w]

        """原本：
        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)  # padding=0 dilation=1   -> [batch, 256,  h, w]
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)  # padding=6 dilation=6  -> [batch, 256,  h, w]
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)  # padding=12 dilation=12 out=in -> [batch, 256,  h, w]
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)  # padding=18 dilation=18 out=in -> [batch, 256,  h, w]
        """
        """
        # MSPP
        # 膨胀后卷积核尺寸 = 膨胀系数 * (原始卷积核尺寸 - 1) + 1  = 2x(3-1)+1 =5
        # k=2*5-3=7 out=[in-k+2*padding]/stride+1=[(in-7+6)/1]+1=in    out = in
        self.aspp1 = nn.Sequential(nn.Conv2d(inplanes, 256, 1, stride=1, padding=0, dilation=dilations[0], bias=False),  # conv1*1
                                    BatchNorm(256),
                                    nn.ReLU())   # con1*1   out=[in-1+2*0]/1+1=in  ->[b,256,h,w]
        self.aspp2 = nn.Sequential(nn.Conv2d(inplanes, 256, 3, stride=1, padding=dilations[0], dilation=dilations[0], bias=False),  # conv3*3 rate=1
                                    BatchNorm(256),
                                    nn.ReLU())   # con3*3 rate=1  k=3 out=[in-3+2]/1+1=in     ->[b,256,h,w]
        self.aspp3 = nn.Sequential(nn.Conv2d(inplanes, 256, 3, stride=1, padding=dilations[0], dilation=dilations[0], bias=False),  # conv3*3 rate=1
                                    BatchNorm(256),
                                    nn.ReLU(),
                                    nn.Conv2d(256, 256, 3, stride=1, padding=dilations[1], dilation=dilations[1], bias=False),  # conv3*3 rate=3
                                    BatchNorm(256),
                                    nn.ReLU())   # con3*3 rate=1 -> conv3*3 rate=3    rate=3:k=3*(3-1)+1=5 out=[in-5+6]/1+1=in  ->[b,256,h,w]
        self.aspp4 = nn.Sequential(nn.Conv2d(inplanes, 256, 3, stride=1, padding=dilations[0], dilation=dilations[0], bias=False),  # conv3*3 rate=1
                                   BatchNorm(256),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, 3, stride=1, padding=dilations[1], dilation=dilations[1], bias=False),  # conv3*3 rate=3
                                   BatchNorm(256),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, 3, stride=1, padding=dilations[2], dilation=dilations[2], bias=False),  # conv3*3 rate=5
                                   BatchNorm(256),
                                   nn.ReLU(),)  # con3*3 rate=1 -> conv3*3 rate=3 -> conv3*3 rate=5     rate=5:k=5*(3-1)+1=11 out=[in-11+5*2]/1+1=in  ->[b,256,h,w]
        """


        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化：括号内是输出尺寸1*1：相当于全剧平均池化   对每个特征图，累加所有像素值并求平均  减少参数数量，减少计算量，减少过拟合
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),  # -> [batch, 256, h, w]
                                             BatchNorm(256),
                                             nn.ReLU())  # -> [batch, 256, h, w]
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)  # -> [batch, 256, h, w]
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout:丢掉   表示每个神经元有0.5的可能性不被激活   p为元素被置0的概率，即被‘丢’掉的概率
        self._init_weight()

    def forward(self, x):  # x = [4,1280,32,32]
        x1 = self.aspp1(x)  # ->[4,256,32,32]
        x2 = self.aspp2(x)  # ->[4,256,32,32]
        x3 = self.aspp3(x)  # ->[4,256,32,32]
        x4 = self.aspp4(x)  # ->[4,256,32,32]
        x5 = self.global_avg_pool(x)  # ->[4,256,1,1] 全局平均池化 256个1*1的矩阵
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # ->[4,256,32,32]   将x5的wh还原
        # 把256个1*1矩阵数值copy成7*7
        # align_corners：设置为True，则输入和输出张量由其角像素的中心点对齐，从而保留角像素处的值。
        # bilinear：双线性插值  x4.size()[2:]:按照x4的第三维第四维的尺寸大小（32，32）
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # 将两个张量（tensor）拼接在一起,按维数（1）列对齐 torch.Size([4, 1280, 32, 32])

        x = self.conv1(x)  # 改变通道数 -> [4,256,32,32]
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)  # 防止过拟合：在训练的过程中，随机选择去除一些神经元，在测试的时候用全部的神经元，这样可以使得模型的泛化能力更强，因为它不会依赖某些局部的特征

    def _init_weight(self):
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

    """原本：
    def forward(self, feats, m=None):   # torch.Size([1, 2048, 32, 32])   # mobilenet：torch.Size([4, 320, 32, 32])
        h, w = feats.size(2), feats.size(3)
        
        # 第一个是原始特征图
        priors = [feats]
        # 将特征图进行四次池化卷积【自适应平均池化->卷积操作->上采样】
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]

        # 将以上四个卷积块与原特征图进行concat，再进行1*1卷积，减少通道数
        bottle = self.bottleneck(torch.cat(priors, 1))  # [8, 4, 32, 32]

        return bottle
    """
    """ 扩充理解：
    def forward(self, feats, m=None):   # torch.Size([1, 2048, 32, 32])   # mobilenet：torch.Size([4, 320, 32, 32])
        h, w = feats.size(2), feats.size(3)
        # 第一个是原始特征图
        priors = [feats]

        # 将特征图进行四次池化卷积，然后将这四个卷积块与原特征图进行concat，再进行1*1卷积，减少通道数
        priors1 = self.conv1(feats)   # torch.Size([2, 512, 1, 1])
        priors1 = F.upsample(priors1, size=(h, w), mode='bilinear', align_corners=True)  # torch.Size([2, 512, 32, 32])
        priors.append(priors1)

        priors2 = self.conv2(feats)  # torch.Size([2,512, 2, 2])
        priors2 = F.interpolate(priors2, size=(h, w), mode='bilinear', align_corners=True)  # torch.Size([2, 512, 32, 32])
        priors.append(priors2)

        priors3 = self.conv3(feats)  # torch.Size([2, 512, 3, 3])
        priors3 = F.upsample(priors3, size=(h, w), mode='bilinear', align_corners=True)  # torch.Size([2, 512, 32, 32])
        priors.append(priors3)

        priors4 = self.conv4(feats)  # torch.Size([2, 512, 6, 6])
        priors4 = F.upsample(priors4, size=(h, w), mode='bilinear', align_corners=True)  # torch.Size([2, 512, 32, 32])
        priors.append(priors4)

        out = torch.cat(priors, dim=1)   # torch.Size([2, 4096, 32, 32])
        bottle = self.bottleneck(out)  # torch.Size([2, 512, 32, 32])

        return bottle
    """

    # # 将特征图进行四次池化卷积【自适应平均池化->卷积操作->上采样】
    # priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
    #
    # # 将以上四个卷积块与原特征图进行concat，再进行1*1卷积，减少通道数
    # bottle = self.bottleneck(torch.cat(priors, 1))  # [8, 4, 32, 32]

    """
      PSPModule(
          (stages): ModuleList(
            (0): Sequential(
              (0): AdaptiveAvgPool2d(output_size=(1, 1))
              (1): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Sequential(
              (0): AdaptiveAvgPool2d(output_size=(2, 2))
              (1): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (2): Sequential(
              (0): AdaptiveAvgPool2d(output_size=(3, 3))
              (1): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (3): Sequential(
              (0): AdaptiveAvgPool2d(output_size=(6, 6))
              (1): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          
          (bottleneck): Sequential(
            (0): Conv2d(3072, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Dropout2d(p=0.1, inplace=False)
          )
      )
    """


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)




"""aspp初始：
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,  # k=2*5-3=7 out=[in-7(k)+6(padding))/1(stride)]+1=in    out = in
                                            stride=1, padding=padding, dilation=dilation, bias=False)  # 空洞卷积  -> [batch, 256,  h, w]
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)  # -> [batch, 256,  h, w]
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        elif backbone == 'ghostnet':
            inplanes = 160
        else:
            inplanes = 2048  # backbone = 'resnet'

        if output_stride == 16:  # output_stride = 16
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0],
                                 BatchNorm=BatchNorm)  # padding=0 dilation=1   -> [batch, 256,  h, w]
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1],
                                 BatchNorm=BatchNorm)  # padding=6 dilation=6  -> [batch, 256,  h, w]
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2],
                                 BatchNorm=BatchNorm)  # padding=12 dilation=12 out=in -> [batch, 256,  h, w]
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)


        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化：括号内是输出尺寸1*1：相当于全剧平均池化   对每个特征图，累加所有像素值并求平均  减少参数数量，减少计算量，减少过拟合
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),  # -> [batch, 256, h, w]
                                             BatchNorm(256),
                                             nn.ReLU())  # -> [batch, 256, h, w]
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)  # -> [batch, 256, h, w]
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout:丢掉   表示每个神经元有0.5的可能性不被激活   p为元素被置0的概率，即被‘丢’掉的概率
        self._init_weight()

    def forward(self, x):  # x = [4,1280,32,32]
        x1 = self.aspp1(x)  # ->[4,256,32,32]
        x2 = self.aspp2(x)  # ->[4,256,32,32]
        x3 = self.aspp3(x)  # ->[4,256,32,32]
        x4 = self.aspp4(x)  # ->[4,256,32,32]
        x5 = self.global_avg_pool(x)  # ->[4,256,1,1] 全局平均池化 256个1*1的矩阵
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)  # ->[4,256,32,32]   将x5的wh还原
        # 把256个1*1矩阵数值copy成7*7
        # align_corners：设置为True，则输入和输出张量由其角像素的中心点对齐，从而保留角像素处的值。
        # bilinear：双线性插值  x4.size()[2:]:按照x4的第三维第四维的尺寸大小（32，32）
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # 将两个张量（tensor）拼接在一起,按维数（1）列对齐 torch.Size([4, 1280, 32, 32])

        x = self.conv1(x)  # 改变通道数 -> [4,256,32,32]
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)  # 防止过拟合：在训练的过程中，随机选择去除一些神经元，在测试的时候用全部的神经元，这样可以使得模型的泛化能力更强，因为它不会依赖某些局部的特征

    def _init_weight(self):
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