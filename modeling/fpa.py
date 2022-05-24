import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FPA(nn.Module):
    def __init__(self, channels=160):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels/4)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)  # ->[b,c,h,w]
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)  # ->[b,c,h,w]
        self.bn_gpb = nn.BatchNorm2d(channels)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)  # ->[b,c/4,h/2,w/2]
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)  # ->[b,c/4,h/4,w/4]
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)  # ->[b,c/4,h/8,w/8]
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)  # ->[b,c/4,h/2,w/2]
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)  # ->[b,c/4,h/4,w/4]
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)  # ->[b,c/4,h/8,w/8]
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Convolution Upsample  上采样
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)  # ->[b,c/4,h/4,w/4]
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)  # ->[b,c/4,h/2,w/2]
        self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)  # ->[b,c,h,w]
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # [2,160,512,512]
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch  原图特征图进行1*1卷积，与金字塔融合后的特征图相乘
        x_master = self.conv_master(x)      # [2,160,512,512]->[2,160,512,512]
        x_master = self.bn_master(x_master)

        # Global pooling branch 全局平均池化->1*1卷积->上采样
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)  # [2,160,512,512]->[2,160,1,1]
        x_gpb = self.conv_gpb(x_gpb)   # [2,160,1,1]->[2,160,1,1]
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)   # [2,160,512,512]->[2,40,256,256]
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)  # [2,40,256,256]
        x1_2 = self.bn1_2(x1_2)   # 用于上采样，在经过1*1的卷积后

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)  # [2,40,256,256] ->[2,40,128,128]
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)  # [2,40,128,128]
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)  # [2,40,128,128]->[2,40,64,64]
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)  # [2,40,64,64]
        x3_2 = self.bn3_2(x3_2)

        # Merge branch 1 and 2
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))  # [2,40,64,64]->[2,40,128,128]
        x2_merge = self.relu(x2_2 + x3_upsample)  # [2,40,128,128]
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))  # [2,40,128,128]->[2,40,256,256]
        x1_merge = self.relu(x1_2 + x2_upsample)  # [2,40,256,256]

        # 两特征图相乘 尺寸大小不变
        x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))  # torch.Size([2, 160, 512, 512])

        out = self.relu(x_master + x_gpb)  # [2, 160, 512, 512] + [2, 160, 1, 1]

        return out

def build_fpa(backbone, BatchNorm):  # feature pyramid attention
    return FPA(backbone, BatchNorm)

if __name__=='__main__':
    model = FPA()
    model.eval()
    # print(model)
    input = torch.randn(2, 160, 512, 512)
    y = model(input)
    print(y.size())  # torch.Size([2, 160, 512, 512])