# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   dmsg.py
@Time    :   2021/3/22 21:05
@Desc    :
"""
from .common import *

from mmengine.model import BaseModule
from mmagic.registry import MODELS

@MODELS.register_module()
class DMSG(BaseModule):
    """
    title: Depth Map Super-Resolution by Deep Multi-Scale Guidance
    paper: https://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2016_depth.pdf
    code: https://github.com/twhui/MSG-Net
    """
    def __init__(self, 
                 scale=2,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super(DMSG, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.scale = scale
        self.img_upsampler = nn.Upsample(scale_factor=self.scale, mode="bicubic", align_corners=False)
        m = int(np.log2(self.scale))
        j = np.arange(2, 2 * m - 1, 2)
        j_ = np.arange(3, 2 * m, 2)

        M = 3 * (m + 1)
        k = np.arange(1, 3 * m, 3)
        k_1 = k + 1
        k_2 = k + 2
        k_3 = np.arange(3 * m + 1, M - 1, 3)

        self.branchY = nn.ModuleList()
        self.branchMain = nn.ModuleList()
        self.gaussian = torch_gaussian(channels=1, kernel_size=15, sigma=5)

        self.branchY.append(ConvBNReLU2D(1, 49, kernel_size=7, stride=1, padding=3, act="PReLU"))
        self.branchY.append(ConvBNReLU2D(49, 32, kernel_size=5, stride=1, padding=2, act="PReLU"))

        for i in range(2, 2 * m):
            if i in j_:
                self.branchY.append(nn.MaxPool2d(3, 2, padding=1))

            if i in j:
                self.branchY.append(ConvBNReLU2D(32, 32, kernel_size=5, stride=1, padding=2, act="PReLU"))

        self.feature_extra = ConvBNReLU2D(1, 64, kernel_size=5, stride=1, padding=2, act="PReLU")

        in_channels, out_channels = 64, 32
        for i in range(1, M):
            if i in k:
                self.branchMain.append(DeConvReLU(in_channels, out_channels, kernel=5, stride=2, padding=2))
            if i in k_1:
                self.branchMain.append(DeConvReLU(in_channels * 2, out_channels, kernel=5, stride=1, padding=2))
            if (i in k_2) or (i in k_3):
                self.branchMain.append(ConvBNReLU2D(in_channels, out_channels, 5, stride=1, padding=2, act="PReLU"))
            in_channels, out_channels = 32, 32

        self.branchMain.append(ConvBNReLU2D(32, 1, 5, stride=1, padding=2, act="PReLU"))

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

        rgb_img = torch.mean(hr_guidance, dim=1, keepdim=True)
        h_Yh = rgb_img - self.gaussian(rgb_img)
        h_Yh = (h_Yh - torch_min(h_Yh)) / (torch_max(h_Yh) - torch_min(h_Yh))

        m = int(np.log2(self.scale))
        k = np.arange(0, 3 * m - 1, 3)

        outputs_Y = [h_Yh]

        for layer in self.branchY:
            outputs_Y.append(layer(outputs_Y[-1]))

        outputs_Main = [self.feature_extra(x - self.gaussian(x))]

        for i, layer in enumerate(self.branchMain):
            outputs_Main.append(layer(outputs_Main[-1]))
            if i in k:
                y_ind = 2 * (m - i // 3)
                outputs_Main.append(torch.cat((outputs_Y[y_ind], outputs_Main[-1]), dim=1))
        out = outputs_Main[-1] + self.gaussian(lr_up)

        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out
    

if __name__ == "__main__":
    x = torch.randn(1, 1, 56, 56)

    # gui = torch.randn(1, 3, 112, 112)
    # gui = torch.randn(1, 10, 224, 224)
    gui = torch.randn(1, 10, 448, 448)
    net = DMSG(
        scale=8,
        # guide_channels=10,
    )
    out = net(x, gui)
    print(out.shape)
