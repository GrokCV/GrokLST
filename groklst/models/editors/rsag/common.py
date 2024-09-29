import math
import numpy as np
import torch
import torch.nn as nn

# import torch.nn.functional as F
# import torch.nn.init as init


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))

                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))

            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class DownBlock(nn.Module):
    def __init__(self, opt, scale, nFeat=None, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()
        negval = opt.negval  # LeakyReLU的参数
        if nFeat is None:
            nFeat = opt.n_feats
        if in_channels is None:
            in_channels = opt.n_colors
        if out_channels is None:
            out_channels = opt.n_colors
        dual_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True),
            )
        ]
        for _ in range(1, int(np.log2(scale))):
            dual_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=negval, inplace=True),
                )
            )
        dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return self.sigmoid(y) * x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SAttention(nn.Module):
    def __init__(self, nfeat):
        super(SAttention, self).__init__()
        self.trans_1 = nn.Conv2d(in_channels=nfeat, out_channels=nfeat, kernel_size=1)
        self.trans_2 = nn.Conv2d(in_channels=nfeat, out_channels=nfeat, kernel_size=3, padding=1)
        self.trans_3 = nn.Conv2d(in_channels=nfeat, out_channels=nfeat, kernel_size=1)
        self.trans_4 = nn.Conv2d(in_channels=nfeat, out_channels=nfeat, kernel_size=5, padding=2)
        self.trans_5 = nn.Conv2d(in_channels=nfeat, out_channels=nfeat, kernel_size=3, padding=1)
        self.trans_6 = nn.Conv2d(in_channels=nfeat, out_channels=nfeat, kernel_size=1)
        self.downsample = nn.AvgPool2d(2)
        self.ca1 = CALayer(nfeat, 16)
        self.ca2 = CALayer(nfeat, 16)
        self.sp1 = SpatialAttention()
        self.sp2 = SpatialAttention()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.relu3 = nn.PReLU()
        self.relu4 = nn.PReLU()
        self.relu5 = nn.PReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, depth, rgb):
        rgb_1 = self.relu1(self.trans_3(rgb))
        rgb_2 = self.relu5(self.trans_1(rgb))
        rgb_3 = self.relu2(self.trans_2(rgb))
        depth_4 = self.relu3(self.trans_6(depth))
        depth_5 = self.relu4(self.trans_5(depth))
        rgb_att = torch.abs(rgb_2 - rgb_3)
        depth_att = torch.abs(depth_4 - depth_5)
        rgb_att = self.sp1(self.ca1(rgb_att + depth_att))
        return rgb_att + self.gamma * rgb_1


class DCN(nn.Module):
    def __init__(self, nChannels=1):
        super(DCN, self).__init__()
        self.conv1 = nn.Conv2d(nChannels, nChannels, kernel_size=5, padding=2, bias=False)
        self.conv2 = nn.Conv2d(nChannels, nChannels, kernel_size=7, padding=3, bias=False)
        self.conv3 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(nChannels, nChannels, kernel_size=5, padding=2, bias=False)
        self.conv5 = nn.Conv2d(nChannels, nChannels, kernel_size=1, padding=0, bias=False)
        self.conv6 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv7 = nn.Conv2d(nChannels * 2, nChannels, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.relu3 = nn.PReLU()
        self.relu4 = nn.PReLU()
        self.relu5 = nn.PReLU()
        self.relu6 = nn.PReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y1 = self.relu3(self.conv1(x))
        y2 = self.relu4(self.conv2(x))
        y = torch.abs(y2 - y1)
        edge1 = self.sig(y)

        y1 = self.relu3(self.conv3(edge1))
        y2 = self.relu4(self.conv4(edge1))
        y = torch.abs(y1 - y2)
        edge2 = self.sig(y)

        y1 = self.relu5(self.conv5(edge2))
        y2 = self.relu6(self.conv6(edge2))
        y = torch.abs(y1 - y2)
        edge3 = self.sig(y)

        hr_2 = edge3
        lf_2 = x - hr_2
        return hr_2, lf_2


class CBAM(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(CBAM, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        modules_body.append(SpatialAttention())
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class DeconvPReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, output_padding=stride - 1
        )
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
