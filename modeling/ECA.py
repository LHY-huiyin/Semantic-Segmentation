# coding=utf-8
import os
import sys

import math

import torch
from torch import nn


class ECAnet(nn.Module):
    def __init__(self, channel, gamma=2, b=1, ratio=4):
        super(ECAnet, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        padding = kernel_size // 2
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLu(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])
        fc = self.fc(out)
        return x * fc


model = ECAnet(512)
print(model)
inputs = torch.ones([2, 512, 128, 128])
outputs = model(inputs)