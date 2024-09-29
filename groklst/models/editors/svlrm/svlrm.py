import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mmengine.model import BaseModule
from mmagic.registry import MODELS


@MODELS.register_module()
class SVLRM(BaseModule):
    """
    title: Spatially Variant Linear Representation Models for Joint Filtering
    paper: https://openaccess.thecvf.com/content_CVPR_2019/papers/Pan_Spatially_Variant_Linear_Representation_Models_for_Joint_Filtering_CVPR_2019_paper.pdf or https://ieeexplore.ieee.org/document/8953422 
    code: https://github.com/curlyqian/SVLRM
    """
    def __init__(self, 
                 dep_channel=1, 
                 rgb_channels=10, 
                 scale=2,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super().__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.scale = scale
        in_channels = dep_channel + rgb_channels
        self.first_layer = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        feature_block = []
        for _ in range(1, 11):
            feature_block += [
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.1),
            ]
        self.feature_block = nn.Sequential(*feature_block)
        self.final_layer = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.data.normal_(0, math.sqrt(2.0 / n))
                nn.init.constant_(m.bias, 0)

    def forward(self, x, hr_guidance, **kwargs):

        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])

        hr_depth = F.interpolate(x, scale_factor=self.scale, mode="bicubic")
        input_tensor = torch.cat((hr_depth, hr_guidance), dim=1)
        param = F.leaky_relu(self.first_layer(input_tensor), 0.1)
        param = self.feature_block(param)
        param = self.final_layer(param)

        param_alpha, param_beta = param[:, :1, :, :], param[:, 1:, :, :]
        out = param_alpha * hr_depth + param_beta
        
        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out


if __name__ == "__main__":
    img_input = torch.randn(1, 1, 3, 3)
    img_guide = torch.ones(1, 1, 6, 6)

    network_s = SVLRM(scale=2)
    # network_s = network_s.apply(weights_init)
    img_out, param_alpha, param_beta = network_s(img_input, img_guide)
    print(network_s)
    print("theta {0} '\n' beta{1}".format(param_alpha, param_beta))
    print(param_alpha.size(), param_beta.size())

    # img_out = param_alpha * img_guide + param_beta

    print("==========>")
    print("img_input", img_input)
    print("img_guide", img_guide)
    print("img_out", img_out)

    diff = nn.L1Loss()
    diff = diff(img_out, img_guide)
    print(diff)
