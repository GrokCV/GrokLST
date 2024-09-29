# modified from https://github.com/ZQPei/deep_joint_filter

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmagic.registry import MODELS

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class CNN(nn.Module):
    def __init__(self, num_conv=3, c_in=1, channel=[96, 48, 1], kernel_size=[9, 1, 5], stride=[1, 1, 1], padding=[2, 2, 2]):
        super(CNN, self).__init__()

        layers = []
        for i in range(num_conv):
            layers += [nn.Conv2d(c_in if i == 0 else channel[i-1], channel[i],
                                 kernel_size[i], stride[i], padding[i], bias=True)]
            if i != num_conv-1:
                layers += [nn.ReLU(inplace=True)]

        self.feature = nn.Sequential(*layers)

        # self.init_weights()

    def forward(self, x):
        fmap = self.feature(x)
        return fmap


# called DJFR if residual = True
@MODELS.register_module()
class DJF(BaseModule):
    """
    title: Deep Joint Image Filtering
    paper: https://ieeexplore.ieee.org/document/8598855 or https://arxiv.org/abs/1710.04200
    code: https://github.com/Yijunmaverick/DeepJointFilter
    projects: http://vllab1.ucmerced.edu/~yli62/DJF_residual/
    """
    def __init__(self, 
                 dep_in_channels=1, 
                 guidance_in_channels=3, 
                 upscale_factor=2,
                 init_weights=False, 
                 residual=False,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}): # called DJFR if residual = True
        super().__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.residual = residual
        self.guidance_in_channels = guidance_in_channels
        self.dep_in_channels = dep_in_channels
        self.upscale_factor  = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

        self.cnn_t = CNN(c_in=dep_in_channels, channel=[96, 48, 1])
        self.cnn_g = CNN(c_in=guidance_in_channels, channel=[96, 48, 1])
        self.cnn_f = CNN(c_in=2, channel=[64, 32, 1])

        if init_weights:
            self.init_weights()

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
        fmap1 = self.cnn_t(hr_depth)
        fmap2 = self.cnn_g(hr_guidance)

        out = self.cnn_f(torch.cat([fmap1, fmap2], dim=1))

        if self.residual:
            out = out + hr_depth

        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out


if __name__ == "__main__":

    hr_dep = torch.randn((2, 1, 112, 112))
    rgb = torch.randn((2, 3, 224, 224))
    net = DJF()
    out = net(hr_dep, rgb)
    print(out.shape)