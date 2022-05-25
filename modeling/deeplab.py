import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.damm import build_damm
from modeling.point_flow import PointFlowModuleWithMaxAvgpool
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp, PSPModule
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        # normal_layer(out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

class UperNetAlignHeadMaxAvgpool(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 fpn_dsn=False, reduce_dim=64, ignore_background=False, max_pool_size=8,
                 avgpool_size=8, edge_points=32):   # resnet50网络  -- 原本
    # def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[24, 40, 112, 160], fpn_dim=40,
    #              fpn_dsn=False, reduce_dim=64, ignore_background=False, max_pool_size=8,
    #              avgpool_size=8, edge_points=32):   # ghostnet网络
    # def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=512,
    #              fpn_dsn=False, reduce_dim=64, ignore_background=False, max_pool_size=8,
    #              avgpool_size=8, edge_points=32):   # resnet101网络  fpd_dim=256:四次切片后融合：1024-2048-256
    # def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[16, 24, 32, 320], fpn_dim=80,
    #              fpn_dsn=False, reduce_dim=64, ignore_background=False, max_pool_size=8,
    #              avgpool_size=8, edge_points=32):   # Mobile net网络
        super(UperNetAlignHeadMaxAvgpool, self).__init__()

        self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)

        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    # norm_layer(fpn_dim),
                    nn.BatchNorm2d(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
            """ fpn_in
            (0): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            )
            """
        self.fpn_in = nn.ModuleList(self.fpn_in)
        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
            """ fpn_out
                (0): Sequential(
                 (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                 (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                 (2): ReLU(inplace=True)
               )
            """

            if ignore_background:  # false
                self.fpn_out_align.append(
                    PointFlowModuleWithMaxAvgpool(fpn_dim, dim=reduce_dim, maxpool_size=max_pool_size,
                                                  avgpool_size=avgpool_size, edge_points=edge_points))
            else:
                self.fpn_out_align.append(
                    PointFlowModuleWithMaxAvgpool(fpn_dim, dim=reduce_dim, maxpool_size=max_pool_size,
                                                  avgpool_size=avgpool_size, edge_points=edge_points))
            """
            (0): PointFlowModuleWithMaxAvgpool(
                (point_matcher): PointMatcher(
                  (match_conv): Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                  (sigmoid): Sigmoid()
                )
                (down_h): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
                (down_l): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
                (softmax): Softmax(dim=-1)
                (max_pool): AdaptiveMaxPool2d(output_size=(8, 8))
                (avg_pool): AdaptiveAvgPool2d(output_size=(8, 8))
                (edge_final): Sequential(
                  (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                  (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): ReLU()
                  (3): Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                )
              )
            """

            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        # 先将最深层的特征图放入网络中  输出是1/4的特征图大小  160/4=40
        psp_out = self.ppm(conv_out[-1])  # x4:[8, 160, 32, 32]    psp_out：[8, 40, 32, 32]

        f = psp_out  # [8, 40, 32, 32]

        # 将卷积后的结果存入列表fpn_feature_list中，[0]是最深层的特征图
        fpn_feature_list = [f]
        edge_preds = []
        out = []

        for i in reversed(range(len(conv_out) - 1)):  # range(len(conv_out) - 1) = range(3)  reversed(3) = 2、1、0
            conv_x = conv_out[i]  # 从2开始，逐层上采样 x3[0]:[8,112,64,64]   x2[1]:[8,40,128,128]  x1[2]:[8,24,256,256]

            # fpn_in：编码器的特征图，进行1*1卷积，改变通道数（都是输入通道图的1/4） 输入分别是112，40，16，输出通道分别是40，40，40
            conv_x = self.fpn_in[i](conv_x)  # [0]:[8, 40, 64, 64]  [1]:[8,40,128,128]  [2]:[8,40,256,256]

            # 将解码器端的输出f和编码器的上一层输出conv_x进行点流最大平均模块的融合，得到两个值：进入下一层f
            f, edge_pred = self.fpn_out_align[i]([f, conv_x])  # f:[8, 40, 32, 32]   conv_x:[8, 40, 64, 64]
            # f:高层特征图，conv_x:低层特征图  两者融合：相加：增强边界信息
            f = conv_x + f  # conv_x：[2, 512, 64, 64]   f：[2, 512, 64, 64]
            edge_preds.append(edge_pred)

            # 将f进入解码器的卷积层（3*3卷积，改变通道数）中，结果放入fpn_feature_list中  输出通道都是40(mo)  256(res)
            fpn_feature_list.append(self.fpn_out[i](f))  # mobilenet:[0]:[2, 40, 32, 32] [1]:[2, 40, 64, 64]  [2]:[2, 40, 128, 128]  [3]:[2, 40, 256, 256]
            # resnet:[3]:[4,256,128,128]  [2]:[4,256,64,64]  [1]:[4,256,32,32]  [0]:[4,256,32,32]

            # false
            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]  mobilenet: [3]:[2, 40, 32, 32] [2]:[2, 40, 64, 64]  [1]:[2, 40, 128, 128]  [0]:[2, 40, 256, 256]
        # resnet: [0]:[4,256,128,128]  [1]:[4,256,64,64]  [2]:[4,256,32,32]  [3]:[4,256,32,32]
        # 最后一个特征图输出
        output_size = fpn_feature_list[0].size()[2:]  #  resnet:[128,128]   mo:torch.Size([256, 256])
        # 将最后一个特征图以列表的形式赋给fusion_list
        fusion_list = [fpn_feature_list[0]]

        # 除了最后一张特征图，对其他每一张特征图进行上采样，双线性插值，再添加到fusion_list中
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,  # 变成最后一张特征图的大小 res:[128,128] [256, 256]
                mode='bilinear', align_corners=True))  # 尺寸大小都变成256,256

        # 将各解码器卷积层输入进行融合
        fusion_out = torch.cat(fusion_list, 1)  # torch.Size([2, 160, 256, 256])

        # 最后卷积，改变通道数，输出通道为类别数
        x = self.conv_last(fusion_out)  # torch.Size([2, 8, 256, 256])

        # return x, edge_preds
        return x



class DeepLab(nn.Module):
    # def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
    #              sync_bn=True, freeze_bn=False):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=8,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()  # 自己搭建的网络Deeplab会继承nn.Module：
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
        self.head = UperNetAlignHeadMaxAvgpool(inplanes,  num_class=num_classes, norm_layer=BatchNorm,
                                               fpn_dsn=False, reduce_dim=64,
                                               ignore_background=False, max_pool_size=8,
                                               avgpool_size=8, edge_points=32)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.damm = build_damm(backbone, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        # out = (in-kernel_size+2*padding)/stride + 1   k =1+(k-1)*dilation  k = 1+2*2=5
        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel_size=3,  # out = in-1+4 = in
                                             stride=1, padding=1, dilation=2, bias=False),  # 扩张率为2的3*3卷积
                                   BatchNorm(inplanes),
                                   nn.ReLU())
        # self.conv1 = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel_size=1,  # out = in-1+4 = in
        #                     stride=1, padding=1, dilation=1, bias=False),  # 扩张率为2的3*3卷积
        #                     BatchNorm(inplanes),
        #                     nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(low_level_inplanes, 256, kernel_size=3,  # out = in-3+2 = in
                                    stride=1,bias=False),
                                   BatchNorm(256),
                                   nn.ReLU())

        self.freeze_bn = freeze_bn


    def forward(self, input):  # aspp->maspp
        [x1, x2, x3, x4] = self.backbone(input)  # resnet:x4:[4,2048,32,32] x3:[4,1024,32,32] x2:[4,512,64,64] x1:[4,256,128,128]
        maspp = self.aspp(x4)  # [4,256,32,32]

        x = self.decoder(maspp, x4, x3, x2, x1)  # [4,8,128,128]
        # 最后进行4倍上采样
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # 4倍上采样 ->[4,8,512,512]

        return x

    """
    原本：
    def forward(self, input): 
        x, low_level_feat = self.backbone(input)  # x=[2, 160, 7, 7]   low_level_feat=[2, 24, 56, 56]
        x = self.aspp(x)  # [2,160,7,7]->[2,256,7,7]
        x = self.decoder(x, low_level_feat)  # [2, 256, 7, 7]   [2, 24, 56, 56] -> [1, 8, 56, 56]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # 4倍上采样 ->[1,8,512,512]
        
        return x
    """
    """
    双注意力机制：
    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.conv1(x)  # torch.Size([4, 320, 64, 64])
        x1 = self.aspp(x)  # ->[2,256,62,62]
        x2 = self.damm(x)  # ->[1,256,62,62]
        # x = torch.cat((x1, x2), dim=1)  # x->[1,512,7,7]
        # x = self.conv2(x)               # x->[1,256,7,7]  期望的输入通道要修改
        x = x1 + x2  # 融合也可能是两个特征图相加 (4, 256, 62, 62)  torch.Size([4, 256, 62, 62])
        x = self.decoder(x, low_level_feat)  # [4, 256, 62, 62]   [4, 24, 128, 128] -> [1, 8, 56, 56]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # 4倍上采样 ->[4,8,512,512]
        
        return x
    """
    """点流模块：
    def forward(self, input):  # input:torch.Size([2, 3, w, h])
        # 点流模块的引入
        [x1, x2, x3, x4] = self.backbone(input)  # 32 <- 64 <- 128 <- 256 <-512
        x = self.head([x1, x2, x3, x4])  # res:[4,8,128,128]  [b,8,最后一张特征图的大小]
        # x = self.decoder(x)  # [2, 256, 7, 7]   [2, 24, 56, 56] -> [1, 8, 56, 56]
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)  # 4倍上采样 ->[1,8,512,512]

        return x
    """
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
        modules = [self.backbone]  # ResNet((conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),
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
    model = DeepLab(backbone='ghostnet', output_stride=16)
    model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化
    input = torch.rand(2, 3, 224, 224)  # RGB是三通道
    output = model(input)
    print(output.size())


