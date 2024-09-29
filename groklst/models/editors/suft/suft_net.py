import torch
import os
import torch.nn as nn
from .common import *
from mmengine.model import BaseModule
from mmagic.registry import MODELS


@MODELS.register_module()
class SUFTNet(BaseModule):
    """
    title: Symmetric Uncertainty-Aware Feature Transmission for lst Super-Resolution
    paper: https://arxiv.org/abs/2306.00386
    code: https://github.com/ShiWuxuan/SUFT
    """

    def __init__(
        self,
        in_channels: int,
        num_feats: int = 32,
        kernel_size: int = 3,
        scale: int = 2,
        norm_flag=0,
        norm_dict={"mean": None, "std": None, "min": None, "max": None},
    ):
        super(SUFTNet, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.conv_rgb1 = nn.Conv2d(in_channels=in_channels, out_channels=num_feats, kernel_size=kernel_size, padding=1)
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

        self.conv_dp1 = nn.Conv2d(in_channels=1, out_channels=num_feats, kernel_size=kernel_size, padding=1)
        self.dp_rg1 = ResidualGroup(default_conv, num_feats, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg2 = ResidualGroup(default_conv, 64, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg3 = ResidualGroup(default_conv, 96, kernel_size, reduction=16, n_resblocks=4)
        self.dp_rg4 = ResidualGroup(default_conv, 128, kernel_size, reduction=16, n_resblocks=4)

        self.bridge1 = SUFT(dp_feats=32, add_feats=32, scale=scale)
        self.bridge2 = SUFT(dp_feats=64, add_feats=32, scale=scale)
        self.bridge3 = SUFT(dp_feats=96, add_feats=32, scale=scale)

        # self.downsample = default_conv(1, 128, kernel_size=kernel_size)

        my_tail = [
            ResidualGroup(default_conv, 128, kernel_size, reduction=16, n_resblocks=8),
            ResidualGroup(default_conv, 128, kernel_size, reduction=16, n_resblocks=8),
        ]
        self.tail = nn.Sequential(*my_tail)

        self.upsampler = DenseProjection(128, 128, scale, up=True, bottleneck=False)
        last_conv = [
            default_conv(128, num_feats, kernel_size=3, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, 1, kernel_size=3, bias=True),
        ]
        self.last_conv = nn.Sequential(*last_conv)
        self.bicubic = nn.Upsample(scale_factor=scale, mode="bicubic")

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # def forward(self, depth, image):
    def forward(self, x, hr_guidance, **kwargs):
        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1:  # z-score
            assert self.norm_dict["mean"] is not None and self.norm_dict["std"] is not None
            x = (x - self.norm_dict["mean"]) / self.norm_dict["std"]
        elif self.norm_flag == 2:  # min-max
            assert self.norm_dict["min"] is not None and self.norm_dict["max"] is not None
            x = (x - self.norm_dict["min"]) / (self.norm_dict["max"] - self.norm_dict["min"])

        lst = x
        dp_in = self.act(self.conv_dp1(x))
        dp1 = self.dp_rg1(dp_in)

        rgb1 = self.act(self.conv_rgb1(hr_guidance))
        rgb2 = self.rgb_rb2(rgb1)
        ca1_in = self.bridge1(dp1, rgb2)

        dp2 = self.dp_rg2(ca1_in)
        rgb3 = self.rgb_rb3(rgb2)
        ca2_in = self.bridge2(dp2, rgb3)

        dp3 = self.dp_rg3(ca2_in)
        rgb4 = self.rgb_rb4(rgb3)
        ca3_in = self.bridge3(dp3, rgb4)

        dp4 = self.dp_rg4(ca3_in)
        tail_in = self.upsampler(dp4)
        out = self.last_conv(self.tail(tail_in))

        out = out + self.bicubic(lst)
        
        if self.norm_flag == 1:
            out = out * self.norm_dict["std"] + self.norm_dict["mean"]
        elif self.norm_flag == 2:
            out = out * (self.norm_dict["max"] - self.norm_dict["min"]) + self.norm_dict["min"]

        return out


if __name__ == "__main__":
    hr_guidance = torch.randn((1, 3, 256, 256)).cuda()
    lst = torch.randn((1, 1, 128, 128)).cuda()
    net = lst(num_feats=32, kernel_size=3, scale=2).cuda()
    output = net(lst, hr_guidance)
    print(output.shape)
    pass
