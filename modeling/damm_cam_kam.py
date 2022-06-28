import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

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

class _CAM_Module(nn.Module):
    """Channel attention module"""
    def __init__(self, in_dim):
        super(_CAM_Module, self).__init__()
        self.channel_in = in_dim
        # CAM和PAM相比是没有Conv2d层的
        self.gamma = nn.Parameter(torch.zeros(1))  # 注意此处对$\beta$是如何初始化和使用的
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs:
                x : input feature maps
            returns:
                out:attention value + input feature
                attention: B * C * C
        """
        m_batchsize, C, height, width = x.size()  # x:torch.Size([1, 2048, 32, 32])
        proj_query = x.view(m_batchsize, C, -1)  # 维度转换 & 转置 torch.Size([2, 512, 676])
        proj_key = x.view(m_batchsize,C,-1).permute(0,2,1)  # 维度转换 torch.Size([2, 676, 512])
        energy = torch.bmm(proj_query, proj_key)  # 相乘  torch.Size([2, 512, 512])
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy  # torch.Size([2, 512, 512]) 0
        # torch.max()用来求tensor的最大值，0是每列的最大值  keepdim=True：表示输出维度与输入一致  dim=-1与dim=1一致（在行方向上）
        # expand_as()将前面矩阵的大小扩充为后面矩阵的大小
        attention = self.softmax(energy_new)  # 归一化 torch.Size([2, 512, 512])  0.0020
        proj_value = x.view(m_batchsize, C, -1)  # torch.Size([2, 512, 676])

        out = torch.bmm(attention, proj_value)  # 相乘 1
        out = out.view(m_batchsize, C, height, width)  # 维度转换 torch.Size([2, 512, 26, 26])

        out = self.gamma * out + x  # torch.Size([2, 512, 26, 26])
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

class DAMM_CKAM(nn.Module):
    def __init__(self, backbone, inplanes, outplanes):
        super(DAMM_CKAM, self).__init__()

        # if backbone == 'drn':
        #     inplanes = 512
        # elif backbone == 'mobilenet':
        #     inplanes = 320
        # elif backbone == 'ghostnet':
        #     inplanes = 160
        # else:
        #     inplanes = 2048  # backbone = 'resnet'

        self.damm1 = _PAM_Module(inplanes)
        self.damm2 = _CAM_Module(inplanes)
        "调整输出通道数"
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),  # out = (in - k + 2p)/s +1
            nn.BatchNorm2d(outplanes),
            nn.ReLU()
        )

    def forward(self, x):  # torch.Size([1, 2048, 32, 32])
        x1 = self.damm1(x)  # torch.Size([1, 2048, 32, 32])
        x2 = self.damm2(x)
        x = x1 + x2
        x = self.conv1(x)

        return x  # [1,256,32,32]

def build_damm_cam_kam(backbone, inplanes, outplanes):
    return DAMM_CKAM(backbone, inplanes, outplanes)