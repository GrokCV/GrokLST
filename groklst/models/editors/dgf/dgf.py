import torch
from torch import nn
from .guided_filter import ConvGuidedFilter

from mmengine.model import BaseModule
from mmagic.registry import MODELS

@MODELS.register_module()
class DGF(BaseModule):
    """
    title: Fast end-to-end trainable guided filter     
    paper: https://ieeexplore.ieee.org/document/8578295 
    
    code: https://github.com/wuhuikai/DeepGuidedFilter/tree/master
    """
    def __init__(self, 
                 scale=2,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super(DGF, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.scale = scale
        self.img_upsampler = nn.Upsample(scale_factor=self.scale, mode="bicubic", align_corners=False)
        self.layers = ConvGuidedFilter(in_channels=1, num_features=64)

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

        hr_guidance = torch.mean(hr_guidance, dim=1, keepdim=True)

        out = self.layers(hr_guidance, lr_up)

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
    net = DGF(scale=8)
    out = net(x, gui)
    print(out.shape)
