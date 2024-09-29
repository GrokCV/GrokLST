#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: OmniSR.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 23rd April 2023 3:06:36 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import torch
import torch.nn as nn
from .OSAG import OSAG
from .pixelshuffle import pixelshuffle_block
import torch.nn.functional as F
from mmagic.registry import MODELS

from mmengine.model import BaseModule

@MODELS.register_module()
class OmniSR(BaseModule):
    """
    title: Omni Aggregation Networks for Lightweight Image Super-Resolution
    paper: https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wang_Omni_Aggregation_Networks_CVPR_2023_supplemental.pdf
    
    code: https://github.com/Francis0625/Omni-SR
    """
    def __init__(self, 
                 up_scale=2, 
                 num_in_ch=1, 
                 num_out_ch=1, 
                 num_feat=64, 
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None},
                 **kwargs):
        super(OmniSR, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict

        # res_num = kwargs["res_num"]
        # up_scale = kwargs["upsampling"]
        # bias = kwargs["bias"]
        res_num = 5
        up_scale = up_scale
        bias = True

        residual_layer = []
        self.res_num = res_num

        for _ in range(res_num):
            temp_res = OSAG(channel_num=num_feat, **kwargs)
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)
        self.input = nn.Conv2d(
            in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.output = nn.Conv2d(
            in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.up = pixelshuffle_block(num_feat, num_out_ch, up_scale, bias=bias)

        # self.tail   = pixelshuffle_block(num_feat,num_out_ch,up_scale,bias=bias)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, sqrt(2. / n))

        self.window_size = 8
        self.up_scale = up_scale

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant", 0)
        return x

    def forward(self, x, **kwargs):
        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])

        H, W = x.shape[2:]
        x = self.check_image_size(x)

        residual = self.input(x)
        out = self.residual_layer(residual)

        # origin
        out = torch.add(self.output(out), residual)
        out = self.up(out)

        out = out[:, :, : H * self.up_scale, : W * self.up_scale]

        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out


if __name__ == "__main__":
    x = torch.randn((1, 1, 128, 128)).cuda()
    net = OmniSR(up_scale=2).cuda()
    out = net(x)
    print(out.shape)
