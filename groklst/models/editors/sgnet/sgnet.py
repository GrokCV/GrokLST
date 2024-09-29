from .common import *
import torch
import torch.nn as nn

from mmagic.registry import MODELS

from mmengine.model import BaseModule

@MODELS.register_module()
class SGNet(BaseModule):
    """
    title: SGNet: Structure Guided Network via Gradient-Frequency Awareness for Depth Map Super-Resolution
    paper: https://arxiv.org/pdf/2312.05799.pdf
    code: https://github.com/yanzq95/SGNet
    """
    def __init__(self, 
                 lst_channels=1, 
                 gui_channels=10, 
                 num_feats=40, 
                 kernel_size=3, 
                 scale=2,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super(SGNet, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.conv_rgb1 = nn.Conv2d(in_channels=gui_channels, out_channels=num_feats, kernel_size=kernel_size, padding=1)
        self.rgb_rb2 = ResBlock(
            default_conv,
            num_feats,
            kernel_size,
            bias=True,
            bn=False,
            act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            res_scale=1,
        )
        self.rgb_rb3 = ResBlock(
            default_conv,
            num_feats,
            kernel_size,
            bias=True,
            bn=False,
            act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            res_scale=1,
        )
        self.rgb_rb4 = ResBlock(
            default_conv,
            num_feats,
            kernel_size,
            bias=True,
            bn=False,
            act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
            res_scale=1,
        )

        self.conv_dp1 = nn.Conv2d(in_channels=lst_channels, out_channels=num_feats, kernel_size=kernel_size, padding=1)
        self.conv_dp2 = nn.Conv2d(in_channels=num_feats, out_channels=2 * num_feats, kernel_size=kernel_size, padding=1)
        self.dp_rg1 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=16, n_resblocks=6)
        self.dp_rg2 = ResidualGroup(default_conv, 2 * num_feats, kernel_size, reduction=16, n_resblocks=6)
        self.dp_rg3 = ResidualGroup(default_conv, 2 * num_feats, kernel_size, reduction=16, n_resblocks=6)
        self.dp_rg4 = ResidualGroup(default_conv, 3 * num_feats, kernel_size, reduction=16, n_resblocks=6)

        self.bridge1 = SDM(channels=num_feats, rgb_channels=num_feats, scale=scale)
        self.bridge2 = SDM(channels=2 * num_feats, rgb_channels=num_feats, scale=scale)
        self.bridge3 = SDM(channels=3 * num_feats, rgb_channels=num_feats, scale=scale)

        self.c_de = default_conv(4 * num_feats, 2 * num_feats, 1)

        my_tail = [
            ResidualGroup(default_conv, 3 * num_feats, kernel_size, reduction=16, n_resblocks=8),
            ResidualGroup(default_conv, 3 * num_feats, kernel_size, reduction=16, n_resblocks=8),
            ResidualGroup(default_conv, 3 * num_feats, kernel_size, reduction=16, n_resblocks=8),
        ]
        self.tail = nn.Sequential(*my_tail)

        self.upsampler = DenseProjection(3 * num_feats, 3 * num_feats, scale, up=True, bottleneck=False)
        last_conv = [
            default_conv(3 * num_feats, num_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, 1, kernel_size=3, bias=True),
        ]
        self.last_conv = nn.Sequential(*last_conv)
        self.bicubic = nn.Upsample(scale_factor=scale, mode="bicubic")

        self.c_rd = default_conv(8 * num_feats, 3 * num_feats, 1)
        self.c_grad = default_conv(2 * num_feats, num_feats, 1)
        self.c_grad2 = default_conv(3 * num_feats, 2 * num_feats, 1)
        self.c_grad3 = default_conv(3 * num_feats, 3 * num_feats, 1)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.gradNet = GCM(n_feats=num_feats, scale=scale)

    def forward(self, x, hr_guidance, **kwargs):
        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])

        out_re, grad_d4 = self.gradNet(x, hr_guidance)

        dp_in = self.act(self.conv_dp1(x))
        dp1 = self.dp_rg1(dp_in)

        cat10 = torch.cat([dp1, grad_d4], dim=1)
        dp1_ = self.c_grad(cat10)

        rgb1 = self.act(self.conv_rgb1(hr_guidance))
        rgb2 = self.rgb_rb2(rgb1)

        ca1_in, r1 = self.bridge1(dp1_, rgb2)
        dp2 = self.dp_rg2(torch.cat([dp1, ca1_in + dp_in], 1))

        cat11 = torch.cat([dp2, grad_d4], dim=1)
        dp2_ = self.c_grad2(cat11)

        rgb3 = self.rgb_rb3(r1)
        ca2_in, r2 = self.bridge2(dp2_, rgb3)

        ca2_in_ = ca2_in + self.conv_dp2(dp_in)

        cat1_0 = torch.cat([dp2, ca2_in_], 1)

        dp3 = self.dp_rg3(self.c_de(cat1_0))
        rgb4 = self.rgb_rb4(r2)

        cat12 = torch.cat([dp3, grad_d4], dim=1)
        dp3_ = self.c_grad3(cat12)

        ca3_in, r3 = self.bridge3(dp3_, rgb4)

        cat1 = torch.cat([dp1, dp2, dp3, ca3_in], 1)

        dp4 = self.dp_rg4(self.c_rd(cat1))

        tail_in = self.upsampler(dp4)
        out = self.last_conv(self.tail(tail_in))

        out = out + self.bicubic(x)

        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out



if __name__ == "__main__":
    x = torch.randn((1, 1, 112, 112)).cuda()
    hr_guidance = torch.randn((1, 10, 224, 224)).cuda()
    net = SGNet(num_feats=32, kernel_size=3, scale=2).cuda()
    out = net(x, hr_guidance)
    print(out.shape)
