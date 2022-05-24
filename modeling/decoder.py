import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

# 参考U-Net
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.double_conv(x)

class _PAM_Module(nn.Module):
    """Position attention module"""
    # Ref from SAGAN
    def __init__(self, in_dim) -> object:  # 输入通道数
        super(_PAM_Module, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8,kernel_size=1)  # ->(b,in_dim/8,h,w) (2,64,26,26)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8,kernel_size=1)  # ->(b,in_dim/8,h,w) (2,64,26,26)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,kernel_size=1)  # ->(b,in_dim,h,w)  (2,512,26,26)
        self.gamma = nn.Parameter(torch.zeros(1)) #注意此处对$\alpha$是如何初始化和使用的
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs:
                x : input feature maps
            returns:
                out:attention value + input feature
                attention: B * (H*W) * (H*W)
        """
        m_batchsize, C, height, width = x.size()  # 特征图尺寸 A: x(2,512,26,26)

        # reshape & transpose
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)  #  ->(batch,w*h,c) torch.Size([1, 1024, 256])

        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)  # ->(batch,c,w*h)  torch.Size([1, 256, 1024])

        # B*C 矩阵相乘  proj_query:B   proj_key:C
        energy = torch.bmm(proj_query, proj_key)  # torch.Size([1, 1024, 1024]) torch.bmm()是tensor中的一个相乘操作，类似于矩阵中的A*B。
        attention = self.softmax(energy)  # 0.0015 torch.Size([1, 1024, 1024]) S：通过softmax函数一作用，就映射成为(0,1)的值，而这些值的累和为1（满足概率的性质），那么我们就可以将它理解成概率

        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)   # D torch.Size([1, 2048, 1024])  -1：自动调整
        # 将D与S相乘
        out = torch.bmm(proj_value,attention.permute(0,2,1))   # torch.Size([1, 2048, 1024])
        out = out.view(m_batchsize, C, height, width)  # reshape  torch.Size([1, 2048, 32, 32])

        out = self.gamma * out + x  # 1  torch.Size([1, 2048, 32, 32])

        return out

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


class Decoder(nn.Module):
    """ 原本
    def __init__(self, num_classes, backbone, BatchNorm, bilinear=True):
    """
    def __init__(self, num_classes, backbone, BatchNorm, bilinear=True):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':  # backbone = resnet
            low_level_inplanes = 256
            in_channels = 3328
            out_channels = 256
            in_chs = 1024  # maspp：1024 点流：304
            in_chs_ = 2048
        elif backbone == 'xception':
            low_level_inplanes = 128
            in_chs = 304  # 点流模块
        elif backbone == 'mobilenet':
            # low_level_inplanes = 24
            low_level_inplanes = 320
            in_chs = 304
        elif backbone == 'ghostnet':
            low_level_inplanes = 24
            in_chs = 304
        else:
            raise NotImplementedError



        # MASPP+位置注意力模块
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        # 1*1卷积
        self.conv1 = nn.Conv2d(low_level_inplanes, 256, kernel_size=1, stride=1, bias=False)  # 改变通道数-> [batch, 256, h, w]
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        self.pam_3 = _PAM_Module(in_chs)
        self.pam_4 = _PAM_Module(in_chs_)

        """原本：
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)  # low_level_inplanes = 256 -> [batch, 48, w, h]
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        """

        # out = (in-kernel_size+2*padding)/stride + 1
        self.last_conv = nn.Sequential(nn.Conv2d(in_chs, 256, kernel_size=3, stride=1, padding=1, bias=False),  # 改变通道数-> [batch, 256, h, w]
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),  # -> [batch, 256, h, w]
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))  # -> [batch, num_classes, h, w]
        self._init_weight()



    # MASPP+位置注意力模块
    def forward(self, maspp, x4, x3, x2, x1):  # resnet:maspp:[4,256,32,32] x4:[4,2048,32,32] x3:[4,1024,32,32] x2:[4,512,64,64] x1:[4,256,128,128]
        # 对MASPP（高层语义信息）进行1*1卷积
        maspp = self.conv1(maspp)
        maspp = self.bn1(maspp)
        maspp = self.relu(maspp)
        # 将x3,x4进行位置注意力机制
        x3 = self.pam_3(x3)
        x4 = self.pam_4(x4)
        # 将x3,x4,MASPP聚合，并4倍上采样  ->1/4
        m = torch.cat((maspp, x4, x3), dim=1)  # [4,3328,32,32]
        m = self.conv(m)  # [4,256,32,32]
        m = F.interpolate(m, size=x1.size()[2:], mode='bilinear', align_corners=True)  # 4倍上采样  [4,256,128,128]
        # 将x2进行2倍上采样
        m1 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)  # 2倍上采样 [4,512,128,128]
        # 将x1进行1*1卷积
        m2 = self.conv1(x1)
        m2 = self.bn1(m2)
        m2 = self.relu(m2)  # [4,256,128,128]
        # 将x1,x2,x3,x4进行组合
        m3 = torch.cat((m, m1, m2), dim=1)  # [4,1024,128,128]
        # 进行3*3卷积
        m4 = self.last_conv(m3)  # [4,8,128,128]

        return m4

    """ MASPP+位置注意力模块
    def forward(self, maspp, x4, x3, x2, x1):  # resnet:maspp:[4,256,32,32] x4:[4,2048,32,32] x3:[4,1024,32,32] x2:[4,512,64,64] x1:[4,256,128,128]
        # 对MASPP（高层语义信息）进行1*1卷积
        maspp = self.conv1(maspp)
        maspp = self.bn1(maspp)
        maspp = self.relu(maspp)
        # 将x3,x4进行位置注意力机制
        x3 = self.pam_3(x3)
        x4 = self.pam_4(x4)
        # 将x3,x4,MASPP聚合，并4倍上采样  ->1/4
        m = torch.cat((maspp, x4, x3), dim=1)  # [4,3328,32,32]
        m = self.conv(m)  # [4,256,32,32]
        m = F.interpolate(m, size=x1.size()[2:], mode='bilinear', align_corners=True)  # 4倍上采样  [4,256,128,128]
        # 将x2进行2倍上采样
        m1 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)  # 2倍上采样 [4,512,128,128]
        # 将x1进行1*1卷积
        m2 = self.conv1(x1)
        m2 = self.bn1(m2)
        m2 = self.relu(m2)  # [4,256,128,128]
        # 将x1,x2,x3,x4进行组合
        m3 = torch.cat((m, m1, m2), dim=1)  # [4,1024,128,128]
        # 进行3*3卷积
        m4 = self.last_conv(m3)  # [4,8,128,128]

        return m4
    """

    """加入MASPP后:
    def forward(self, maspp, x4, x3, x2, x1):  # resnet:maspp:[4,256,32,32] x4:[4,2048,32,32] x3:[4,1024,32,32] x2:[4,512,64,64] x1:[4,256,128,128]
        # 对MASPP（高层语义信息）进行1*1卷积
        maspp = self.conv1(maspp)
        maspp = self.bn1(maspp)
        maspp = self.relu(maspp)
        # 将x3,x4,MASPP聚合，并4倍上采样  ->1/4
        m = torch.cat((maspp, x4, x3), dim=1)  # [4,3328,32,32]
        m = F.interpolate(m, size=x1.size()[2:], mode='bilinear', align_corners=True)  # 4倍上采样  [4,3328,128,128]
        # 将x2进行2倍上采样
        m1 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)  # 2倍上采样 [4,512,128,128]
        # 将x1进行1*1卷积
        m2 = self.conv1(x1)
        m2 = self.bn1(m2)
        m2 = self.relu(m2)
        # 将x1,x2,x3,x4进行组合
        m3 = torch.cat((m, m1, m2), dim=1)
        # 进行3*3卷积
        m4 = self.last_conv(m3)

        return m4
    """
    """原本：
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
    """


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