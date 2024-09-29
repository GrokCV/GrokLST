# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.model import BaseModule
# import torch
from mmagic.registry import MODELS


@MODELS.register_module()
class SRCNN(BaseModule):
    """SRCNN network structure for image super resolution.

    SRCNN has three conv layers. For each layer, we can define the
    `in_channels`, `out_channels` and `kernel_size`.
    The input image will first be upsampled with a bicubic upsampler, and then
    super-resolved in the HR spatial size.

    Paper: Learning a Deep Convolutional Network for Image Super-Resolution.

    Args:
        channels (tuple[int]): A tuple of channel numbers for each layer
            including channels of input and output . Default: (3, 64, 32, 3).
        kernel_sizes (tuple[int]): A tuple of kernel sizes for each conv layer.
            Default: (9, 1, 5).
        upscale_factor (int): Upsampling factor. Default: 4.
    """

    def __init__(self, 
                 channels=(3, 64, 32, 3), 
                 kernel_sizes=(9, 1, 5), 
                 upscale_factor=4,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super().__init__()
        assert len(channels) == 4, "The length of channel tuple should be 4, " f"but got {len(channels)}"
        assert len(kernel_sizes) == 3, "The length of kernel tuple should be 3, " f"but got {len(kernel_sizes)}"
        self.upscale_factor = upscale_factor
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.img_upsampler = nn.Upsample(scale_factor=self.upscale_factor, mode="bicubic", align_corners=False)

        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2)
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2)
        self.conv3 = nn.Conv2d(channels[2], channels[3], kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2)

        self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])
        
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)

        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out
