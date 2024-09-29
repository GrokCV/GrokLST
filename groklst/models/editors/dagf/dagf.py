# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   spn.py
@Time    :   2020/8/6 08:12
@Desc    :
"""
# import math

import torch

from mmengine.model import BaseModule
# import numpy as np
from torch import nn
from torch.nn.functional import interpolate, softmax
import sys
import os

# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ..dagf import common
from mmagic.registry import MODELS


class PyModel(nn.Module):
    def __init__(self, num_features, out_channels=1):
        super(PyModel, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.PReLU())

        self.layer2 = nn.Sequential(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1), nn.PReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.PReLU())
        self.res_scale = common.Scale(0)

    def forward(self, inputs, add_feature):
        out = self.layer1(inputs)
        if add_feature is not None:
            out = self.layer3(self.res_scale(interpolate(add_feature, scale_factor=2, mode="nearest")) + out)
        return out, self.layer2(out)


class FuseBlock(nn.Module):
    def __init__(self, num_feature, act, norm, kernel_size, num_res, scale=2):
        super(FuseBlock, self).__init__()

        self.scale = scale
        self.num = kernel_size * kernel_size

        self.aff_scale_const = nn.Parameter(0.5 * self.num * torch.ones(1))

        self.depth_kernel = nn.Sequential(
            common.ConvBNReLU2D(in_channels=num_feature, out_channels=num_feature, kernel_size=1, act=act, norm=norm),
            common.ConvBNReLU2D(in_channels=num_feature, out_channels=kernel_size**2, kernel_size=1),
        )

        self.guide_kernel = nn.Sequential(
            common.ConvBNReLU2D(in_channels=num_feature, out_channels=num_feature, kernel_size=1, act=act, norm=norm),
            common.ConvBNReLU2D(in_channels=num_feature, out_channels=kernel_size**2, kernel_size=1),
        )

        self.pix_shf = nn.PixelShuffle(upscale_factor=scale)

        self.weight_net = nn.Sequential(
            common.ConvBNReLU2D(
                in_channels=num_feature * 2,
                out_channels=num_feature,
                kernel_size=3,
                padding=1,
                act=act,
                norm="Adaptive",
            ),
            common.TUnet(num_features=num_feature, act=act, norm="Adaptive"),
            common.ConvBNReLU2D(
                in_channels=num_feature, out_channels=1, kernel_size=3, act=act, padding=1, norm="Adaptive"
            ),
        )
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=scale, padding=kernel_size // 2 * scale)
        self.inputs_conv = nn.Sequential(
            *[common.ResNet(num_features=num_feature, act=act, norm=norm) for _ in range(num_res)]
        )

    def forward(self, depth, guide, inputs, ret_kernel=False):
        b, c, h, w = inputs.size()
        h_, w_ = h * self.scale, w * self.scale
        weight_map = self.weight_net(torch.cat((depth, guide), 1))  # wu Softmax

        depth_kernel = self.depth_kernel(depth)
        guide_kernel = self.guide_kernel(guide)

        depth_kernel = softmax(depth_kernel, dim=1)
        guide_kernel = softmax(guide_kernel, dim=1)

        fuse_kernel = weight_map * depth_kernel + (1 - weight_map) * guide_kernel

        fuse_kernel = torch.tanh(fuse_kernel) / (self.aff_scale_const + 1e-8)

        abs_kernel = torch.abs(fuse_kernel)
        abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4

        abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0

        fuse_kernel = fuse_kernel / abs_kernel_sum

        inputs_up = interpolate(self.inputs_conv(inputs), scale_factor=self.scale, mode="nearest")
        unfold_inputs = self.unfold(inputs_up).view(b, c, -1, h_, w_)
        out = torch.einsum("bkhw, bckhw->bchw", [fuse_kernel, unfold_inputs])
        if ret_kernel:
            return out, fuse_kernel, weight_map
        return out


class InitLayer(nn.Module):
    def __init__(self, in_channels, num_features, flag=0):
        super(InitLayer, self).__init__()

        self.flag = flag

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_features, padding=1, kernel_size=3), nn.PReLU()
        )

        if flag == 0:
            self.layer2 = nn.Conv2d(in_channels=num_features, out_channels=num_features, padding=1, kernel_size=3)

        else:
            self.layer2 = nn.Conv2d(in_channels=2 * num_features, out_channels=num_features, padding=1, kernel_size=3)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_features, padding=1, kernel_size=3),
            nn.PReLU(),
            nn.Conv2d(in_channels=num_features, out_channels=num_features, padding=1, kernel_size=3),
        )

    def forward(self, inputs, x):
        out = self.layer1(inputs)
        if self.flag == 0:
            out = self.layer2(out)
        else:
            out = self.layer2(torch.cat((out, x), dim=1))
        return out



@MODELS.register_module()
class DAGF(BaseModule):
    """
    title: Deep Attentional Guided Image Filtering   
    paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10089494
    code: https://github.com/zhwzhong/DAGF
    """
    def __init__(
        self,
        scale=2,
        guide_channels=3,
        num_pyramid=3,
        num_features=32,
        act="PReLU",
        norm="None",
        filter_size=3,
        num_res=2,
        norm_flag=0, 
        norm_dict={'mean':None, 'std':None,'min':None, 'max':None}
    ):
        super(DAGF, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.num_pyramid = num_pyramid
        self.scale = scale
        self.img_upsampler = nn.Upsample(scale_factor=self.scale, mode="bicubic", align_corners=False)
        self.head = common.Head(num_features=num_features, expand_ratio=1, act=act, guide_channels=guide_channels)
        self.depth_pyramid = nn.ModuleList(
            [common.DownSample(num_features=num_features, act=act, norm=norm) for _ in range(self.num_pyramid - 1)]
        )

        self.guide_pyramid = nn.ModuleList(
            [common.DownSample(num_features=num_features, act=act, norm=norm) for _ in range(self.num_pyramid - 1)]
        )

        self.up_sample = nn.ModuleList(
            [
                FuseBlock(
                    num_feature=num_features,
                    act=act,
                    norm=norm,
                    kernel_size=filter_size,
                    num_res=num_res,
                    scale=2,
                )
                for _ in range(self.num_pyramid)
            ]
        )

        self.init_conv = nn.ModuleList(
            [InitLayer(1, num_features=num_features, flag=i) for i in range(self.num_pyramid)]
        )

        self.p_layers = nn.ModuleList([PyModel(num_features, out_channels=1) for _ in range(self.num_pyramid)])

        self.tail_conv = nn.Sequential(
            common.ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1, act=act),
            common.ConvBNReLU2D(in_channels=num_features, out_channels=1, kernel_size=3, padding=1, act=act),
        )

        self.res_scale = common.Scale(0)

    def forward(self, x, hr_guidance, **kwargs):
        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])

        lr_up = self.img_upsampler(x)
        lr_feature, guide_feature = self.head(lr_up, hr_guidance)
        depth_features, guide_features = [lr_feature], [guide_feature]
        for num_p in range(self.num_pyramid - 1):
            lr_feature = self.depth_pyramid[num_p](lr_feature)
            depth_features.append(lr_feature)
            guide_feature = self.guide_pyramid[num_p](guide_feature)
            guide_features.append(guide_feature)
        # 从小到大
        depth_features, guide_features = list(reversed(depth_features)), list(reversed(guide_features))
        lr_input = None
        ret_feature = []
        for i in range(self.num_pyramid):
            h, w = depth_features[i].size()[2:]

            lr_input = self.init_conv[i](interpolate(x, size=(h // 2, w // 2), mode="nearest"), lr_input)
            lr_input = self.up_sample[i](depth_features[i], guide_features[i], lr_input)

            ret_feature.append(lr_input)

        outs = []

        out1, out2 = None, None
        for i in range(self.num_pyramid):
            out1, out2 = self.p_layers[i](ret_feature[i], out1)
            out2 = interpolate(out2, size=hr_guidance.size()[2:], mode="bilinear", align_corners=False) + lr_up
            outs.append(out2)

        out = outs[-1]
        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out


if __name__ == "__main__":
    x = torch.randn(1, 1, 56, 56)

    # gui = torch.randn(1, 10, 112, 112)
    # gui = torch.randn(1, 10, 224, 224)
    gui = torch.randn(1, 10, 448, 448)
    net = DAGF(
        scale=8,
        guide_channels=10,
    )
    out = net(x, gui)
    print(out.shape)
