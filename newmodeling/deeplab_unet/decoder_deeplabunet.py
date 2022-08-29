import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.backbone import build_backbone
from configs import config_factory
from modeling.deeplab_unet.parts import create_decoder

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


class Decoder_deeplabunet(nn.Module):
    def __init__(self, n_classes, low_chan=[1024, 512, 256],  dec_type='unet_scse', num_filters=16, *args, **kwargs):
        super(Decoder_deeplabunet, self).__init__()
        self.dec_type = dec_type
        Decoder = create_decoder(dec_type)

        self.conv_16 = ConvBNReLU(low_chan[0], 256, ks=3, padding=1)  # 此处只是调整通道数，进行3*3卷积，我的想法：调整为注意力机制，边界模块
        self.conv_8 = ConvBNReLU(low_chan[1], 128, ks=3, padding=1)
        self.conv_4 = ConvBNReLU(low_chan[2], 64, ks=3, padding=1)

        # num_filters*2 =32    num_filters * 4 =64      num_filters * 8 =128     num_filters * 16=256      num_filters * 32 = 512      num_filters * 32 * 2 =1024
        self.decoder16 = Decoder(low_chan[0] + num_filters*16, num_filters*32, num_filters*16)  # 1024+256-512-256
        self.decoder8 = Decoder(low_chan[1] + num_filters*16, num_filters*16, num_filters*8)  # 512+256-256-128
        self.decoder4 = Decoder(low_chan[2] + num_filters*8, num_filters*8, num_filters*4)  # 256+128-128-64
        self.decoder2 = Decoder(num_filters*4*2, num_filters*4, num_filters*2)  # 256+64-64-32

        self.fuse = ConvBNReLU(num_filters * (2 + 4 + 8 + 16), 64, ks=3, padding=1)  # 32+64+128+256

        self.conv_out = nn.Conv2d(64, n_classes, kernel_size=1, bias=False)

        self.init_weight()

    def forward(self, x, feat2, feat4, feat8, feat16, feat_lkpp):  # feat_lkpp:[4, 256, 26, 26] feat16:[4,1024,24,24] feat8:[4, 512, 48, 48] feat4:[4, 256, 96, 96]  feat2:[4, 64, 96, 96]
        H, W = feat16.size()[2:]
        img_size = x.shape[2:]

        feat2_up = F.interpolate(feat2, scale_factor=2, mode='bilinear', align_corners=False)  # [4, 64, 192, 192]
        feat_lkpp_up = F.interpolate(feat_lkpp, (H, W), mode='bilinear',
                                     align_corners=True)  # [4, 256, 26, 26] -> [4, 256, 24, 24]

        feat16_out = self.decoder16(feat16, feat_lkpp_up)  # torch.Size([4, 256, 48, 48])  先concat再进行卷积，与注意力机制
        feat8_out = self.decoder8(feat8, feat16_out)  # torch.Size([4, 128, 96, 96])
        feat4_out = self.decoder4(feat4, feat8_out)  # torch.Size([4, 64, 192, 192])
        feat2_out = self.decoder2(feat2_up, feat4_out)  # torch.Size([4, 32, 384, 384])

        u16 = F.interpolate(feat16_out, img_size, mode='bilinear', align_corners=False)  # 上采样 [4, 256, 384, 384]
        u8 = F.interpolate(feat8_out, img_size, mode='bilinear', align_corners=False)  # [4, 128, 384, 384]
        u4 = F.interpolate(feat4_out, img_size, mode='bilinear', align_corners=False)  # [4, 64, 384, 384]

        d = torch.cat((feat2_out, u4, u8, u16), 1)  # torch.cat(feat2_out, u4, u8, u16)超出显存  32+64+128+256
        logits = self.conv_out(self.fuse(d))

        return logits

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

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder_deeplabunet(num_classes, backbone, BatchNorm)