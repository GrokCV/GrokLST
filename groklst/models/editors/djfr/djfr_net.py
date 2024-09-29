import torch
from torch import nn
from .common import ConvBNReLU2D
from mmengine.model import BaseModule
from mmagic.registry import MODELS


@MODELS.register_module()
class DJFR(BaseModule):
    """
    title: Joint Image Filtering with Deep Convolutional Networks
    paper: https://ieeexplore.ieee.org/document/8598855
    code: https://github.com/Yijunmaverick/DeepJointFilter
    
    
    """
    def __init__(self, 
                 dep_in_channels=1, 
                 guidance_in_channels=3, 
                 upscale_factor = 2,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super(DJFR, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.guidance_in_channels = guidance_in_channels
        self.dep_in_channels = dep_in_channels
        self.upscale_factor  = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)
        
        self.depth_encoder = nn.Sequential(
            ConvBNReLU2D(in_channels=dep_in_channels, out_channels=96,
                         kernel_size=9, stride=1, padding=2, act='ReLU'),
            ConvBNReLU2D(in_channels=96, out_channels=48,
                         kernel_size=1, stride=1, padding=2, act='ReLU'),
            ConvBNReLU2D(in_channels=48, out_channels=3,
                         kernel_size=5, stride=1, padding=2)
        )

        self.rgb_encoder = nn.Sequential(
            ConvBNReLU2D(in_channels=guidance_in_channels, out_channels=96,
                         kernel_size=9, stride=1, padding=2, act='ReLU'),
            ConvBNReLU2D(in_channels=96, out_channels=48,
                         kernel_size=1, stride=1, padding=2, act='ReLU'),
            ConvBNReLU2D(in_channels=48, out_channels=3,
                         kernel_size=5, stride=1, padding=2)
        )

        self.decoder = nn.Sequential(
            ConvBNReLU2D(in_channels=6, out_channels=64,
                         kernel_size=9, stride=1, padding=4, act='ReLU'),
            ConvBNReLU2D(in_channels=64, out_channels=32,
                         kernel_size=1, stride=1, act='ReLU'),
            ConvBNReLU2D(in_channels=32, out_channels=dep_in_channels,
                         kernel_size=5, stride=1, padding=2)
        )

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
        hr_guidance_out = self.rgb_encoder(hr_guidance)
        dep_out = self.depth_encoder(hr_depth)
        out = self.decoder(torch.cat((hr_guidance_out, dep_out), dim=1))
        out =  out + hr_depth

        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out


# def make_model(args): return DJFR(args)

if __name__ == "__main__":
    arr = torch.randn(1, 1, 64, 64)
    b = torch.randn(1, 9, 128, 128)
    net = DJFR(1,9)
    out = net(arr, b)
    print(out.size())