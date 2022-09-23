import torch
import torch.nn as nn
import torch.nn.functional as F

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):

    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()
        # 膨胀后卷积核尺寸 = 膨胀系数 * (原始卷积核尺寸 - 1) + 1
        # out = [in-k+2*p]/s+1
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)  # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)  # Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)  k=1*(2-1)+1=2

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)  # Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)  # Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)  # Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)  # Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)  # Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)  # Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))  k=2*(3-1)+1=5 out=(in-5+2*2)/1+1=in

        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)  # Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  改变通道数

    def forward(self, x):

        hx = x  # 获取图像的大小，如torch.Size([4, 3, 512, 512])
        hxin = self.rebnconvin(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,64,512,512]  hxin 通道数调整

        hx1 = self.rebnconv1(hxin)  # 经过一个3*3卷积，dilate=1 =》 [4,32,512,512]  hx1
        hx = self.pool1(hx1)  # 经过一个2*2卷积，步长为2的下采样 =》 [4,32,256,256]

        hx2 = self.rebnconv2(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,32,256,256]  hx2
        hx = self.pool2(hx2)  # 经过一个2*2卷积，步长为2的下采样 =》 [4,32,128,128]

        hx3 = self.rebnconv3(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,32,128,128]  hx3
        hx = self.pool3(hx3)  # 经过一个2*2卷积，步长为2的下采样 =》 [4,32,64,64]

        hx4 = self.rebnconv4(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,32,64,64]  hx4
        hx = self.pool4(hx4)  # 经过一个2*2卷积，步长为2的下采样 =》 [4,32,32,32]

        hx5 = self.rebnconv5(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,32,32,32]  hx5
        hx = self.pool5(hx5)  # 经过一个2*2卷积，步长为2的下采样 =》 [4,32,16,16]

        hx6 = self.rebnconv6(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,32,16,16]  hx6

        hx7 = self.rebnconv7(hx6)  # 经过一个3*3卷积，dilate=2 =》 [4,32,16,16]  hx7

        hx6d =  self.rebnconv6d(torch.cat((hx7, hx6), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,32,16,16]  hx6d
        hx6dup = _upsample_like(hx6d, hx5)  # 上采样，按hx5的大小 =》 [4,32,32,32]  hx6dup

        hx5d =  self.rebnconv5d(torch.cat((hx6dup, hx5), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,32,32,32]
        hx5dup = _upsample_like(hx5d, hx4)  # 上采样，按hx4的大小 =》 [4,32,64,64]

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,32,64,64]
        hx4dup = _upsample_like(hx4d, hx3)  # 上采样，按hx3的大小 =》 [4,32,128,128]

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,32,128,128]
        hx3dup = _upsample_like(hx3d, hx2)  # 上采样，按hx2的大小 =》 [4,32,256,256]

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,32,256,256]
        hx2dup = _upsample_like(hx2d, hx1)  # 上采样，按hx1的大小 =》 [4,32,512,512]

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,64,512,512]

        return hx1d + hxin  # [4,64,512,512]

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)  # Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)  # Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)  # Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)  # Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)  # Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)  # Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)  # Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))

        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)  # Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  改变通道数

    def forward(self,x):

        hx = x  # torch.Size([4, 64, 256, 256])

        hxin = self.rebnconvin(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,128,256,256] hxin 改变通道数

        hx1 = self.rebnconv1(hxin)  # 经过一个3*3卷积，dilate=1 =》 [4,32,256,256]  hx1
        hx = self.pool1(hx1)  # 经过一个2*2卷积，步长为2的下采样 =》 [4,32,128,128]

        hx2 = self.rebnconv2(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,32,128,128]  hx2
        hx = self.pool2(hx2)  # 经过一个2*2卷积，步长为2的下采样 =》 [4,32,64,64]

        hx3 = self.rebnconv3(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,32,64,64]  hx3
        hx = self.pool3(hx3)  # 经过一个2*2卷积，步长为2的下采样 =》 [4,32,32,32]

        hx4 = self.rebnconv4(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,32,32,32]  hx4
        hx = self.pool4(hx4)  # 经过一个2*2卷积，步长为2的下采样 =》 [4,32,16,16]

        hx5 = self.rebnconv5(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,32,16,16]  hx5

        hx6 = self.rebnconv6(hx5)  # 经过一个3*3卷积，dilate=2 =》 [4,32,16,16]  hx6


        hx5d =  self.rebnconv5d(torch.cat((hx6, hx5), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,32,16,16]  hx5d
        hx5dup = _upsample_like(hx5d, hx4)  # 上采样，按hx4的大小 =》 [4,32,32,32]

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,32,32,32]  hx4d
        hx4dup = _upsample_like(hx4d, hx3)  # 上采样，按hx3的大小 =》 [4,32,64,64]

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,32,64,64]  hx3d
        hx3dup = _upsample_like(hx3d, hx2)  # 上采样，按hx2的大小 =》 [4,32,128,128]

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,32,128,128]  hx2d
        hx2dup = _upsample_like(hx2d, hx1)  # 上采样，按hx1的大小 =》 [4,32,256,256]

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,128,256,256]  hx1d

        return hx1d + hxin  # [4,128,256,256]

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)  # Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)  # Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)  # Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)  # Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)  # Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)  # Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))

        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)  # Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  改变通道数

    def forward(self, x):

        hx = x  # [4, 128, 128, 128]

        hxin = self.rebnconvin(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,256,256,256] hxin 改变通道数

        hx1 = self.rebnconv1(hxin)  # 经过一个3*3卷积，dilate=1 =》 [4,64，128，128]  hx1
        hx = self.pool1(hx1)  # 经过一个2*2卷积，步长为2的下采样 =》 [4,64,64,64]

        hx2 = self.rebnconv2(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,64,64,64]  hx2
        hx = self.pool2(hx2)  # 经过一个2*2卷积，步长为2的下采样 =》 [4,64,32,32]

        hx3 = self.rebnconv3(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,64,32,32]  hx3
        hx = self.pool3(hx3)  # 经过一个2*2卷积，步长为2的下采样 =》 [4,64,16,16]

        hx4 = self.rebnconv4(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,64,16,16]  hx4

        hx5 = self.rebnconv5(hx4)  # 经过一个3*3卷积，dilate=2 =》 [4,64,16,16]  hx5

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,64,16,16] hx4d
        hx4dup = _upsample_like(hx4d, hx3)  # 上采样，按hx3的大小 =》 [4,64,32,32]

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,64,32,32] hx3d
        hx3dup = _upsample_like(hx3d, hx2)  # 上采样，按hx2的大小 =》 [4,64,64,64]

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》[4,64,64,64]  hx2d
        hx2dup = _upsample_like(hx2d, hx1)  # 上采样，按hx1的大小 =》 [4,64，128，128]

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,256，128，128]  hx1d

        return hx1d + hxin  # [4,256，128，128]

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)  # Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)  # Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)  # Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)  # Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)  # Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)  # Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)  # Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  改变通道数

    def forward(self, x):

        hx = x  # [4, 256, 64, 64]

        hxin = self.rebnconvin(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,512,64,64]  hxin 改变通道数

        hx1 = self.rebnconv1(hxin)  # 经过一个3*3卷积，dilate=1 =》 [4,128,64,64]  hx1
        hx = self.pool1(hx1)  # 经过一个2*2卷积，步长为2的下采样 =》 [4,128,32,32]

        hx2 = self.rebnconv2(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,128,32,32] hx2
        hx = self.pool2(hx2)  # 经过一个2*2卷积，步长为2的下采样 =》 [4,128,16,16]

        hx3 = self.rebnconv3(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,128,16,16] hx3

        hx4 = self.rebnconv4(hx3)  # 经过一个3*3卷积，dilate=2 =》 [4,128,16,16] hx4

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》[4,128,16,16]  hx3d
        hx3dup = _upsample_like(hx3d, hx2)  # 上采样，按hx2的大小 =》 [4,128,32,32]

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,128,32,32]  hx2d
        hx2dup = _upsample_like(hx2d, hx1)  # 上采样，按hx1的大小 =》 [4,128,64,64]

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,512,64,64]  hx1d

        return hx1d + hxin  # [4,512,64,64]

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F,self).__init__()
        # 膨胀后卷积核尺寸 = 膨胀系数 * (原始卷积核尺寸 - 1) + 1
        # out = [in-k+2*p]/s+1
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)  # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)  # Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)  # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))  k=2*(3-1)+1=5 out=(in-5+2*2)/1+1=in
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)  # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))  k=4*(3-1)+1=9 out=(in-9+2*4)/1+1=in

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)  # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8))  k=8*(3-1)+1=17 out=(in-11+2*8)/1+1=in

        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)  # Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)  # Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)  # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):

        hx = x  # [4,512,32，32]  [4,512,16,16]  [4,1024,32,32]

        hxin = self.rebnconvin(hx)  # 经过一个3*3卷积，dilate=1 =》 [4,512,32，32]  hxin  [4,512,16,16]

        hx1 = self.rebnconv1(hxin)  # 经过一个3*3卷积，dilate=1 =》 [4,256,32，32]  hx1  [4,256,16,16]
        hx2 = self.rebnconv2(hx1)  # 经过一个3*3卷积，dilate=2 =》 [4,256,32，32]  hx2  [4,256,16,16]
        hx3 = self.rebnconv3(hx2)  # 经过一个3*3卷积，dilate=4 =》 [4,256,32，32]  hx3  [4,256,16,16]

        hx4 = self.rebnconv4(hx3)  # 经过一个3*3卷积，dilate=8 =》 [4,256,32，32]  hx4  [4,256,16,16]

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))  # 先cat然后经过一个3*3卷积，dilate=4 =》 [4,256,32，32] hx3d  [4,256,16,16]
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))  # 先cat然后经过一个3*3卷积，dilate=2 =》 [4,256,32，32]  hx2d  [4,256,16,16]
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))  # 先cat然后经过一个3*3卷积，dilate=1 =》 [4,512,32，32]  hx1d  [4,512,16,16]

        return hx1d + hxin  # [4,512,32，32]  [4,512,16,16]


##### U^2-Net ####
class U2NET(nn.Module):

    def __init__(self, in_ch=3, out_ch=8):
        super(U2NET, self).__init__()

        self.stage1 = RSU7(in_ch, 32, 64)   # 输出通道数为64，图片尺寸不变
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 2*2卷积，下采样两倍

        self.stage2 = RSU6(64, 32, 128)  # 输出通道数为128，图片尺寸不变
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 2*2卷积，下采样两倍

        self.stage3 = RSU5(128, 64, 256)  # 输出通道数为256，图片尺寸不变
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 2*2卷积，下采样两倍

        self.stage4 = RSU4(256, 128, 512)  # 输出通道数为512，图片尺寸不变
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 2*2卷积，下采样两倍

        self.stage5 = RSU4F(512, 256, 512)  # 输出通道数为512，图片尺寸不变
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 2*2卷积，下采样两倍

        self.stage6 = RSU4F(512, 256, 512)  # 输出通道数为512，图片尺寸不变

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)  # 与stage5的尺寸变化一致
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)

    def forward(self,x):

        hx = x

        # stage 1
        hx1 = self.stage1(hx)  # 经过一个编解码结构的结果：[4,64,512,512]  hx1 改变通道数
        hx = self.pool12(hx1)  # 下采样两倍：[4,64,256,256]

        # stage 2
        hx2 = self.stage2(hx)  # 经过一个编解码结构的结果：[4,128,256,256]  hx2 改变通道数
        hx = self.pool23(hx2)  # 下采样两倍：[4,128,128,128]

        # stage 3
        hx3 = self.stage3(hx)  # 经过一个编解码结构的结果：[4,256,128,128]  hx3 改变通道数
        hx = self.pool34(hx3)  # 下采样两倍：[4,256,64,64]

        # stage 4
        hx4 = self.stage4(hx)  # 经过一个编解码结构的结果：[4, 512, 64, 64]  hx4 改变通道数
        hx = self.pool45(hx4)  # 下采样两倍：[4,512,32，32]

        # stage 5
        hx5 = self.stage5(hx)  # 经过一个编解码结构的结果：[4,512,32，32]  hx5
        hx = self.pool56(hx5)  # 下采样两倍：[4,512,16,16]

        # stage 6
        hx6 = self.stage6(hx)  # 经过一个编解码结构的结果：[4,512,16,16]
        hx6up = _upsample_like(hx6, hx5)  # 下采样两倍：[4,512,32，32]

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))  # 经过一个编解码结构的结果：[4,512,32，32] hx5d
        hx5dup = _upsample_like(hx5d, hx4)  # 下采样两倍：

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))


        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

### U^2-Net small ###
class U2NETP(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(U2NETP,self).__init__()

        self.stage1 = RSU7(in_ch,16,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,16,64)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(64,16,64)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(64,16,64)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(64,16,64)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(64,16,64)

        # decoder
        self.stage5d = RSU4F(128,16,64)
        self.stage4d = RSU4(128,16,64)
        self.stage3d = RSU5(128,16,64)
        self.stage2d = RSU6(128,16,64)
        self.stage1d = RSU7(128,16,64)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(64,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self,x):

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)

        #decoder
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        #side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)
