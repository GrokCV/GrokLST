# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   cunet.py
@Time    :   2022/7/26 20:32
@Desc    :
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale
from mmengine.model import BaseModule
from mmagic.registry import MODELS

class Prediction(nn.Module):
    def __init__(self, num_channels):
        super(Prediction, self).__init__()
        self.num_layers = 4
        self.in_channel = num_channels
        self.kernel_size = 9
        self.num_filters = 64

        self.layer_in = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,
                                  kernel_size=self.kernel_size, padding=4, stride=1, bias=False)
        nn.init.xavier_uniform_(self.layer_in.weight.data)
        self.lam_in = nn.parameter.Parameter(torch.FloatTensor([0.01]))

        self.lam_i = []
        self.layer_down = []
        self.layer_up = []
        for i in range(self.num_layers):
            down_conv = 'down_conv_{}'.format(i)
            up_conv = 'up_conv_{}'.format(i)
            lam_id = 'lam_{}'.format(i)
            layer_2 = nn.Conv2d(in_channels=self.num_filters, out_channels=self.in_channel,
                                kernel_size=self.kernel_size, padding=4, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_2.weight.data)
            setattr(self, down_conv, layer_2)
            self.layer_down.append(getattr(self, down_conv))
            layer_3 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_filters,
                                kernel_size=self.kernel_size, padding=4, stride=1, bias=False)
            nn.init.xavier_uniform_(layer_3.weight.data)
            setattr(self, up_conv, layer_3)
            self.layer_up.append(getattr(self, up_conv))

            lam_ = nn.parameter.Parameter(torch.FloatTensor([0.01]))
            setattr(self, lam_id, lam_)
            self.lam_i.append(getattr(self, lam_id))

    def forward(self, mod):
        p1 = self.layer_in(mod)
        tensor = torch.mul(torch.sign(p1), F.relu(torch.abs(p1) - self.lam_in))
        # print(tensor.device, p1.device, self.lam_in.device, 'tensor', mod.device)
        for i in range(self.num_layers):
            # print(next(self.layer_down[i].parameters()).device, 'para')
            p3 = self.layer_down[i](tensor)
            p4 = self.layer_up[i](p3)
            p5 = tensor - p4
            p6 = torch.add(p1, p5)
            tensor = torch.mul(torch.sign(p6), F.relu(
                torch.abs(p6) - self.lam_i[i]))
        return tensor


class decoder(nn.Module):
    def __init__(self, channel=1):
        super(decoder, self).__init__()
        self.channel = channel
        self.kernel_size = 9
        self.filters = 64
        self.conv_1 = nn.Conv2d(in_channels=self.filters, out_channels=self.channel, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        nn.init.xavier_uniform_(self.conv_1.weight.data)
        self.conv_2 = nn.Conv2d(in_channels=self.filters, out_channels=self.channel, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        nn.init.xavier_uniform_(self.conv_2.weight.data)

    def forward(self, u, z):
        rec_u = self.conv_1(u)
        rec_z = self.conv_2(z)
        z_rec = rec_u + rec_z
        return z_rec

@MODELS.register_module()
class CUNet(BaseModule):
    """ 
    title: Deep Convolutional Neural Network for Multi-modal Image Restoration and Fusion
    paper: https://arxiv.org/pdf/1910.04066.pdf
    code: https://github.com/cindydeng1991/TPAMI-CU-Net
    """
    def __init__(self,
                 dep_in_channels=1, 
                 guidance_in_channels=3, 
                 upscale_factor=2,
                 num_filters = 64,
                 kernel_size = 9,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super(CUNet, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict

        self.guidance_in_channels = guidance_in_channels
        self.dep_in_channels = dep_in_channels
        self.upscale_factor  = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)        
        
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.net_u = Prediction(num_channels=self.dep_in_channels)
        self.conv_u = nn.Conv2d(in_channels=self.num_filters, out_channels=self.dep_in_channels, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        nn.init.xavier_uniform_(self.conv_u.weight.data)
        self.net_v = Prediction(num_channels=self.guidance_in_channels)
        self.conv_v = nn.Conv2d(in_channels=self.num_filters, out_channels=self.guidance_in_channels, kernel_size=self.kernel_size,
                                stride=1, padding=4, bias=False)
        nn.init.xavier_uniform_(self.conv_v.weight.data)
        self.net_z = Prediction(num_channels=self.guidance_in_channels + self.dep_in_channels)
        self.decoder = decoder(dep_in_channels)

    def forward(self, x, hr_guidance, **kwargs):

        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])

        x  = self.img_upsampler(x)
        
        y = hr_guidance
        u = self.net_u(x)
        v = self.net_v(y)

        p_x = x - self.conv_u(u)
        p_y = y - self.conv_v(v) # 9-1 
        p_xy = torch.cat((p_x, p_y), dim=1)

        z = self.net_z(p_xy)
        out = self.decoder(u, z)
        
        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out


if __name__ == "__main__":
    lr_dep = torch.randn((2, 1, 112, 112)).cuda()
    rgb = [torch.randn(( 3, 224, 224)).cuda()] * 2
    # net = PixTransformNet(dep_in_channels=1, guidance_in_channels=3).cuda()
    
    net = CUNet().cuda()
    out = net(lr_dep, rgb)
    print(out.shape)