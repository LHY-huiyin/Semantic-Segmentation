import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np


class EDA(nn.Module):
    def __init__(self, in_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(EDA, self).__init__()
        self.conv_1 = nn.Conv2d(in_planes, in_planes // 2, kernel_size=1)
        self.conv_2 = nn.Conv2d(in_planes, in_planes // 2, kernel_size=1)
        self.conv_3 = nn.Conv2d(in_planes, in_planes // 2, kernel_size=1)

        # self.out_channels = out_planes
        # self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
        #                       dilation=dilation, groups=groups, bias=bias)
        self.conv_out = nn.Conv2d(in_planes // 2, in_planes, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
        self.softmax = nn.Softmax()

    def forward(self, x):  # B*C*H*W
        x_c = self.conv_1(x)
        # edge_c = cv.Canny(x_c, 80, 300) 的实现：
        len_c = x_c.size(1)
        len_b = x_c.size(0)
        x_c = x_c.transpose(0, 1)  # tensor数据类型
        x_c = x_c.detach().numpy().astype(np.uint8)  # 转为numpy的形式
        for c in range(0, len_c):
            if c == 0:
                for b in range(0, len_b):
                    if b == 0:
                        edge_c = cv.Canny(x_c[c][b], 80, 300)
                        edge_c = np.expand_dims(edge_c, axis=0)  # 增加维度
                    else:
                        edge_c_1 = cv.Canny(x_c[c][b], 80, 300)
                        edge_c_1 = np.expand_dims(edge_c_1, axis=0)
                        edge_c = np.concatenate((edge_c, edge_c_1), 0)
                edge_c = np.expand_dims(edge_c, axis=0)
            else:
                for b in range(0, len_b):
                    if b == 0:
                        edge_c_2 = cv.Canny(x_c[c][b], 80, 300)
                        edge_c_2 = np.expand_dims(edge_c_2, axis=0)
                    else:
                        edge_c_2_ = cv.Canny(x_c[c][b], 80, 300)
                        edge_c_2_ = np.expand_dims(edge_c_2_, axis=0)
                        edge_c_2 = np.concatenate((edge_c_2, edge_c_2_), 0)
                edge_c_2 = np.expand_dims(edge_c_2, axis=0)
                edge_c = np.concatenate((edge_c, edge_c_2), 0)
        # edge_c = torch.tensor(edge_c)
        edge_c = torch.from_numpy(edge_c)  # numpy转为tensor

        x_n = self.conv_2(x)
        # edge_n = cv.Canny(x_n, 80, 300)
        len_c = x_n.size(1)
        len_b = x_n.size(0)
        x_n = x_n.transpose(0, 1)
        x_n = x_n.detach().numpy().astype(np.uint8)
        for c in range(0, len_c):
            if c == 0:
                for b in range(0, len_b):
                    if b == 0:
                        edge_n = cv.Canny(x_n[c][b], 80, 300)
                        edge_n = np.expand_dims(edge_n, axis=0)  # 增加维度
                    else:
                        edge_n_1 = cv.Canny(x_n[c][b], 80, 300)
                        edge_n_1 = np.expand_dims(edge_n_1, axis=0)
                        edge_n = np.concatenate((edge_n, edge_n_1), 0)
                edge_n = np.expand_dims(edge_n, axis=0)
            else:
                for b in range(0, len_b):
                    if b == 0:
                        edge_n_2 = cv.Canny(x_n[c][b], 80, 300)
                        edge_n_2 = np.expand_dims(edge_n_2, axis=0)
                    else:
                        edge_n_2_ = cv.Canny(x_n[c][b], 80, 300)
                        edge_n_2_ = np.expand_dims(edge_n_2_, axis=0)
                        edge_n_2 = np.concatenate((edge_n_2, edge_n_2_), 0)
                edge_n_2 = np.expand_dims(edge_n_2, axis=0)
                edge_n = np.concatenate((edge_n, edge_n_2), 0)
        edge_n = torch.from_numpy(edge_n)
        edge_n = edge_n.transpose(0, 1)  # B C H W

        x_r = self.conv_3(x)
        # edge_r = cv.Canny(x_r, 80, 300)
        len_c = x_r.size(1)
        len_b = x_r.size(0)
        x_r = x_r.transpose(0, 1)
        x_r = x_r.detach().numpy().astype(np.uint8)
        for c in range(0, len_c):
            if c == 0:
                for b in range(0, len_b):
                    if b == 0:
                        edge_r = cv.Canny(x_r[c][b], 80, 300)
                        edge_r = np.expand_dims(edge_r, axis=0)  # 增加维度
                    else:
                        edge_r_1 = cv.Canny(x_r[c][b], 80, 300)
                        edge_r_1 = np.expand_dims(edge_r_1, axis=0)
                        edge_r = np.concatenate((edge_r, edge_r_1), 0)
                edge_r = np.expand_dims(edge_r, axis=0)
            else:
                for b in range(0, len_b):
                    if b == 0:
                        edge_r_2 = cv.Canny(x_r[c][b], 80, 300)
                        edge_r_2 = np.expand_dims(edge_r_2, axis=0)
                    else:
                        edge_r_2_ = cv.Canny(x_r[c][b], 80, 300)
                        edge_r_2_ = np.expand_dims(edge_r_2_, axis=0)
                        edge_r_2 = np.concatenate((edge_r_2, edge_r_2_), 0)
                edge_r_2 = np.expand_dims(edge_r_2, axis=0)
                edge_r = np.concatenate((edge_r, edge_c_2), 0)
        edge_r = torch.from_numpy(edge_r)  # numpy 转 tensor

        # edge_c = edge_c.transpose(0, 1)  # C B H W
        f_c_aver = edge_c[0, :, :, :]  # 初始化，第一个赋值
        len_c = edge_c.shape[0]
        for c in range(1, len_c):
            f_c_aver = f_c_aver + edge_c[c, :, :, :]  # 求和
        f_c_aver = f_c_aver / len_c  # 平均值
        for c in range(0, len_c):
            # 获得完整的tensor参数
            if c == 0:
                cov_c = (edge_c[c, b, :, :] - f_c_aver).transpose(1, 2) * (edge_c[c, b, :, :] - f_c_aver)  # tensor类型的转置
                cov_c = cov_c.unsqueeze(0)  # tensor 增加一维
            else:
                cov_c_ = (edge_c[c, b, :, :] - f_c_aver).transpose(1, 2) * (edge_c[c, b, :, :] - f_c_aver)
                cov_c_ = cov_c_.unsqueeze(0)
                cov_c = torch.cat((cov_c, cov_c_), 0)  # tensor类型的拼接
        cov_c = cov_c / len_c
        cov_c = cov_c.transpose(0, 1)  # B C H W

        # edge_r = edge_r.transpose(0, 1)  # C B H W
        len_r = edge_r.shape[0]  # 数组形式
        f_r_aver = edge_r[0, :, :, :]
        for c in range(1, len_r):
            f_r_aver = f_r_aver + edge_r[c, :, :, :]  # 求和
        f_r_aver = f_r_aver / len_r
        for c in range(0, len_r):
            # cov_r = (edge_r[c, :, :, :] - f_r_aver).transpose(1, 2) * (edge_r[c, :, :, :] - f_r_aver)
            # 获得完整的tensor参数
            if c == 0:
                cov_r = (edge_r[c, b, :, :] - f_r_aver).transpose(1, 2) * (edge_r[c, b, :, :] - f_r_aver)  # tensor类型的转置
                cov_r = cov_r.unsqueeze(0)  # tensor 增加一维
            else:
                cov_r_ = (edge_r[c, b, :, :] - f_r_aver).transpose(1, 2) * (edge_r[c, b, :, :] - f_r_aver)
                cov_r_ = cov_r_.unsqueeze(0)
                cov_r = torch.cat((cov_r, cov_r_), 0)  # tensor类型的拼接
        cov_r = cov_r / len_r
        cov_r = cov_r.transpose(0, 1)  # B C H W

        # softmax函数
        softmax_r = self.softmax(cov_r)
        softmax_c = self.softmax(cov_c)

        logits = edge_n * softmax_c * softmax_r

        # 通道数回来
        logits = self.conv_out(logits)

        return logits

class SAM(nn.Module):
    """ Parallel CBAM """

    def __init__(self, in_ch):
        super(SAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """ Spatial Attention Module """
        x_attention = self.conv(x)

        return x * x_attention

class HAM(nn.Module):
    def __init__(self, in_planes, kernel_size=3, stride=1):
        super(HAM, self).__init__()
        # if backbone == 'drn':
        #     inplanes = 512
        # elif backbone == 'mobilenet':
        #     inplanes = 320
        # elif backbone == 'ghostnet':
        #     inplanes = 160
        # else:
        #     inplanes = 2048  # backbone = 'resnet'

        self.ham1 = EDA(in_planes)
        self.ham2 = SAM(in_planes)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, bias=False),  # out = (in - k + 2p)/s +1
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )

    def forward(self, x):  # torch.Size([1, 2048, 32, 32])
        x1 = self.ham1(x)  # torch.Size([1, 2048, 32, 32])
        x2 = self.ham2(x)
        x = x1 + x2
        x = self.conv1(x)

        return x

if __name__ == "__main__":
    model = HAM(in_planes=256)
    input = torch.rand(2, 256, 224, 224)  # RGB是三通道
    # input = cv.imread('C:\\Remote sensing semantic segmentation\\loveda_a\\JPEGImages\\1366_0.jpg')
    output = model(input)
    print(output.size())

