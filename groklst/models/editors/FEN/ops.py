import math
import torch
import torch.nn as nn
# import torch.nn.init as init
import torch.nn.functional as F
# import cv2
# from torch.autograd import Variable
# import numpy as np
import torch
# import os



class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale, wn, group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, wn=wn, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, wn=wn, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, wn=wn, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, wn=wn, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, wn,  group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []

        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [wn(nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group)),
                            nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]

        elif scale == 3:
            modules += [wn(nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group)),
            nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        elif scale == 5:
            modules += [wn(nn.Conv2d(n_channels, 25 * n_channels, 3, 1, 1, groups=group)),
            nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(5)]

        self.body = nn.Sequential(*modules)


    def forward(self, x):
        out = self.body(x)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, wn, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = wn(nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True))

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


#Frequency Enhancement (FE) Operation
class FE(nn.Module):
    def __init__(self,
                 wn, in_channels, channels):
        super(FE, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=4)

        self.k2 = wn(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1,padding=1, bias=False))
        self.k3 = wn(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1,padding=1, bias=False))
        self.k4 = wn(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1,padding=1, bias=False))

    def forward(self, x):

        h1 = F.interpolate(self.pool(x), (x.size(-2), x.size(-1)), mode='nearest')
        h2 = x - h1
        F2 = torch.sigmoid(torch.add(self.k2(h2), x))
        out = torch.mul(self.k3(x), F2)
        out = self.k4(out)

        return out

#Frequency-based Enhancement Block (FEB)
class FEB(nn.Module):
    def __init__(self,
                 wn, in_channels, out_channels):
        super(FEB, self).__init__()
        channels = in_channels // 2
        self.path_1 = wn(nn.Conv2d(in_channels, channels, kernel_size=1, bias=False))
        self.path_2 = wn(nn.Conv2d(in_channels, channels, kernel_size=1, bias=False))
        self.relu = nn.ReLU(inplace=True)
        self.k1 = wn(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.HConv = FE(wn, 32, 32)
        self.conv = wn(nn.Conv2d(channels*2, in_channels, kernel_size=1, bias=False))


    def forward(self, x):
        #Low-Frequency Path
        path_1 = self.path_1(x)
        path_1 = self.relu(path_1)
        path_1 = self.k1(path_1)
        path_1  = self.relu(path_1)

        #High-Frequency Path
        path_2 = self.path_2(x)
        path_2 = self.relu(path_2)
        path_2 = self.HConv(path_2)
        path_2 = self.relu(path_2)

        output = self.conv(torch.cat([path_1, path_2], dim=1))
        output = output + x

        return output
