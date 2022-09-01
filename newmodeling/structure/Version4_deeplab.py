from newmodeling.backbone import build_backbone
from newmodeling.structure.Version3_decoder import *
from configs import config_factory
from newmodeling.newidea.aspp_luo import *
cfg = config_factory['resnet_cityscapes']


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class HADCLayer(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, dilation=1, mode='parallel', *args, **kwargs):
        super(HADCLayer, self).__init__()
        self.mode = mode
        self.ks = ks
        if ks > 3:
            padding = int(dilation * ((ks - 1) // 2))
            if mode == 'cascade':
                self.hadc_layer = nn.Sequential(ConvBNReLU(in_chan, out_chan,
                                                           ks=[3, ks], dilation=[1, dilation],
                                                           padding=[1, padding]),
                                                ConvBNReLU(out_chan, out_chan,
                                                           ks=[ks, 3], dilation=[dilation, 1],
                                                           padding=[padding, 1]))
            elif mode == 'parallel':
                self.hadc_layer1 = ConvBNReLU(in_chan, out_chan,
                                              ks=[3, ks], dilation=[1, dilation],
                                              padding=[1,
                                                       padding])  # Conv2d(2048, 256, kernel_size=(3, 7), stride=(1, 1), padding=(1, 3), dilation=(1, 1))
                # Conv2d(256, 256, kernel_size=(3, 7), stride=(1, 1), padding=(1, 6), dilation=(1, 2))
                # Conv2d(256, 256, kernel_size=(3, 7), stride=(1, 1), padding=(1, 9), dilation=(1, 3))
                self.hadc_layer2 = ConvBNReLU(in_chan, out_chan, ks=[ks, 3], dilation=[dilation, 1], padding=[padding,
                                                                                                              1])  # Conv2d(2048, 256, kernel_size=(7, 3), stride=(1, 1), padding=(3, 1), dilation=(1, 1))
                # Conv2d(256, 256, kernel_size=(7, 3), stride=(1, 1), padding=(6, 1), dilation=(2, 1))
                # Conv2d(256, 256, kernel_size=(7, 3), stride=(1, 1), padding=(9, 1), dilation=(3, 1))
            else:
                raise Exception('No %s mode, please choose from cascade and parallel' % mode)

        elif ks == 3:
            self.hadc_layer = ConvBNReLU(in_chan, out_chan, ks=ks, dilation=dilation, padding=dilation)

        else:
            self.hadc_layer = ConvBNReLU(in_chan, out_chan, ks=ks, dilation=1, padding=0)

        self.init_weight()

    def forward(self, x):
        if self.mode == 'cascade' or self.ks <= 3:
            return self.hadc_layer(x)
        elif self.mode == 'parallel' and self.ks > 3:  # 前两个ks=7,ks=5
            x1 = self.hadc_layer1(x)  # 并行的卷积，卷积核大小为(k1,k2)  torch.Size([4, 256, 24, 24])
            x2 = self.hadc_layer2(x)  # 并行的卷积，卷积核大小为(k2,k1)  torch.Size([4, 256, 24, 24])
            return x1 + x2

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class LKPBlock(nn.Module):
    def __init__(self, in_chan, out_chan, ks, dilation=[1, 2, 3], mode='parallel', *args, **kwargs):
        super(LKPBlock, self).__init__()
        if ks >= 3:
            self.lkpblock = nn.Sequential(HADCLayer(in_chan, out_chan,
                                                    ks=ks, dilation=dilation[0], mode=mode),
                                          HADCLayer(out_chan, out_chan,
                                                    ks=ks, dilation=dilation[1], mode=mode),
                                          HADCLayer(out_chan, out_chan,
                                                    ks=ks, dilation=dilation[2], mode=mode))
        else:
            self.lkpblock = HADCLayer(in_chan, out_chan, ks=ks)

        self.init_weight()

    def forward(self, x):
        return self.lkpblock(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class LKPP(nn.Module):
    def __init__(self, in_chan=2048, out_chan=256, ks_list=[7, 5, 3, 1], mode='parallel', with_gp=True, *args,
                 **kwargs):
        super(LKPP, self).__init__()
        self.with_gp = with_gp
        self.conv1 = LKPBlock(in_chan, out_chan, ks=ks_list[0], dilation=[1, 2, 3], mode=mode)
        self.conv2 = LKPBlock(in_chan, out_chan, ks=ks_list[1], dilation=[1, 2, 3], mode=mode)
        self.conv3 = LKPBlock(in_chan, out_chan, ks=ks_list[2], dilation=[1, 2, 3], mode=mode)
        self.conv4 = LKPBlock(in_chan, out_chan, ks=ks_list[3], mode=mode)
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, out_chan,
                                      ks=1)  # Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
            self.conv_out = ConvBNReLU(out_chan * 5, out_chan,
                                       ks=1)  # Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
        else:
            self.conv_out = ConvBNReLU(out_chan * 4, out_chan, ks=1)

        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat1 = self.conv1(x)  # [4,256,24,24]
        feat2 = self.conv2(x)  # [4,256,24,24]
        feat3 = self.conv3(x)  # [4,256,24,24]
        feat4 = self.conv4(x)  # [4,256,24,24]
        if self.with_gp:
            avg = self.avg(x)  # [4,2048,1,1]
            feat5 = self.conv1x1(avg)  # [4,256,1,1]
            feat5 = F.interpolate(feat5, (H, W), mode='bilinear', align_corners=True)  # [4,256,24,24]
            feat = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)  # [4,1280,24,24]

        else:
            feat = torch.cat([feat1, feat2, feat3, feat4], 1)

        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params = []
        non_wd_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'bias' in name:
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        return wd_params, non_wd_params


"总体架构"
class Version3(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=8,
                 sync_bn=True, freeze_bn=False):
        super(Version3, self).__init__()  # 自己搭建的网络Deeplab会继承nn.Module：
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
        self.lkpp = LKPP(in_chan=2048, out_chan=256, mode='parallel', with_gp=cfg.aspp_global_feature)
        # self.lkpp_luo = AsppLuo(in_chan=2048, out_chan=256, mode='parallerl', with_gp=cfg.aspp_global_feature)
        # self.ppm = PPM()
        self.decoder = Decoder_SEBiFPN_EaNet(cfg.n_classes, low_chan=[1024, 512, 256, 64], num_classes=8)
        # self.conv_out = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        self.conv_out = ConvBNReLU(256, num_classes, kernel_size=1, bias=False)
        self.Softmax = nn.Softmax()
        self.CoefRefine = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, bias=False),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=False)
        )
        self.freeze_bn = freeze_bn

    def forward(self, x):
        H, W = x.size()[2:]  # 256 256
        # feat2, feat4, feat8, feat16, feat32 = self.backbone(x)  # resnet:feat32:[4,2048,24,24] feat16:[4,1024,24,24] feat8:[4,512,48,48] feat4:[4,256,96,96]
        # feat_lkpp = self.lkpp(feat32)  # [4, 256, 26, 26]
        x1, x2, x3, x4 = self.backbone(x)
        # x4_lkpp = self.lkpp_luo(x4)
        x4_lkpp = self.lkpp(x4)
        logits = self.decoder(x1, x2, x3, x4_lkpp)  # p2_in p2_out:[4,256,96,96]
        "进行4倍上采样"
        logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=True)  # [4, 8, 96, 96]-># [4, 8, 384, 384]

        return logits

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
        modules = [self.decoder]
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
    model = Version3(backbone='resnet', output_stride=16)
    model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化
    input = torch.rand(4, 3, 384, 384)  # RGB是三通道
    output = model(input)
    print(output.size())
