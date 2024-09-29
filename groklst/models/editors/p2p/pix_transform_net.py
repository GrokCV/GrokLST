# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
# from mmengine.model import BaseModule
import torch
from mmengine.model import BaseModule
from mmagic.registry import MODELS


@MODELS.register_module()
class PixTransformNet(BaseModule):
    """
    title: Guided Super-Resolution as Pixel-to-Pixel Transformation
    paper: https://openaccess.thecvf.com/content_ICCV_2019/papers/de_Lutio_Guided_Super-Resolution_As_Pixel-to-Pixel_Transformation_ICCV_2019_paper.pdf
    code: https://github.com/prs-eth/PixTransform
    """

    def __init__(self, 
                 dep_in_channels=1, 
                 guidance_in_channels=3, 
                 upscale_factor=2,
                 kernel_size=1, 
                 weights_regularizer=None,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super(PixTransformNet, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict

        self.guidance_in_channels = guidance_in_channels
        self.dep_in_channels = dep_in_channels
        self.upscale_factor  = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

        self.spatial_net = nn.Sequential(nn.Conv2d(self.dep_in_channels, 32, (1, 1), padding=0),
                                         nn.ReLU(), nn.Conv2d(32, 2048, (kernel_size, kernel_size), padding=(kernel_size-1)//2))
        self.color_net = nn.Sequential(nn.Conv2d(guidance_in_channels, 32, (1, 1), padding=0),
                                       nn.ReLU(), nn.Conv2d(32, 2048, (kernel_size, kernel_size), padding=(kernel_size-1)//2))
        self.head_net = nn.Sequential(nn.ReLU(), nn.Conv2d(2048, 32, (kernel_size, kernel_size), padding=(kernel_size-1)//2),
                                      nn.ReLU(), nn.Conv2d(32, 1, (1, 1), padding=0))

        if weights_regularizer is None:
            reg_spatial = 0.0001
            reg_color = 0.001
            reg_head = 0.0001
        else:
            reg_spatial = weights_regularizer[0]
            reg_color = weights_regularizer[1]
            reg_head = weights_regularizer[2]

        self.params_with_regularizer = []
        self.params_with_regularizer += [
            {'params': self.spatial_net.parameters(), 'weight_decay': reg_spatial}]
        self.params_with_regularizer += [
            {'params': self.color_net.parameters(), 'weight_decay': reg_color}]
        self.params_with_regularizer += [
            {'params': self.head_net.parameters(), 'weight_decay': reg_head}]

    def forward(self, x, hr_guidance, **kwargs):
        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])

        hr_depth = self.img_upsampler(x)
        merged_features = self.spatial_net(
            hr_depth) + self.color_net(hr_guidance)

        out =  self.head_net(merged_features)
        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out



if __name__ == "__main__":

    lr_dep = torch.randn((2, 1, 112, 112)).cuda()
    rgb = [torch.randn(( 3, 224, 224)).cuda()] * 2
    net = PixTransformNet(dep_in_channels=1, guidance_in_channels=3).cuda()
    out = net(lr_dep, rgb)
    print(out.shape)