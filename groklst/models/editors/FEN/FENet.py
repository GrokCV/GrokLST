import torch.nn as nn
import torch
import torch.nn.functional as F
# import numpy as np
# import os
from .ops import *

from mmagic.registry import MODELS
from mmengine.model import BaseModule



@MODELS.register_module()
class FENet(BaseModule):
    """
    title: Frequency-Based Enhancement Network for Efficient Super-Resolution
    paper: https://ieeexplore.ieee.org/document/9778017
    
    code: https://github.com/pbehjatii/FENet-PyTorch
    """
    def __init__(self, 
                 scale=2, 
                 lst_channels=1, 
                 group=4,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super(FENet, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict

        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.n_blocks = 12
        self.scale = scale
        # scale = kwargs.get("scale")
        # group = kwargs.get("group", 4)

        # self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)

        self.entry_1 = wn(nn.Conv2d(lst_channels, 64, 3, 1, 1))

        body = [FEB(wn, 64, 64) for _ in range(self.n_blocks)]
        self.body = nn.Sequential(*body)
        self.reduction = BasicConv2d(wn, 64 * 13, 64, 1, 1, 0)

        self.upscample = UpsampleBlock(64, scale=scale, multi_scale=False, wn=wn, group=group)
        self.exit = wn(nn.Conv2d(64, lst_channels, 3, 1, 1))

        # self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)

    def forward(self, x, **kwargs):

        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])
        # x = self.sub_mean(x)
        res = x
        x = self.entry_1(x)

        c0 = x
        out_blocks = []

        out_blocks.append(c0)

        for i in range(self.n_blocks):

            x = self.body[i](x)
            out_blocks.append(x)

        output = self.reduction(torch.cat(out_blocks, 1))

        output = output + x

        output = self.upscample(output, scale=self.scale)
        output = self.exit(output)

        skip = F.interpolate(
            res, (x.size(-2) * self.scale, x.size(-1) * self.scale), mode="bicubic", align_corners=False
        )

        out = skip + output

        # output = self.add_mean(output)

        
        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out
    


if __name__ == "__main__":

    model = FENet(scale=8, group=4)
    x = torch.randn((1, 1, 56, 56))
    out = model(x)
    print(out.shape)
