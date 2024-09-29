import torch
import torch.nn.functional as F
import torch.nn as nn
from ..fdsr import octconv as oc
from mmagic.registry import MODELS

from mmengine.model import BaseModule
class MS_RB(nn.Module):
    def __init__(self, num_feats, kernel_size):
        super(MS_RB, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, padding=1, dilation=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, padding=2, dilation=2
        )
        self.conv4 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats, kernel_size=1, padding=0)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x))
        x3 = x1 + x2
        x4 = self.conv4(x3)
        out = x4 + x

        return out


def resample_data(input, s):
    """
    input: torch.floatTensor (N, C, H, W)
    s: int (resample factor)
    """

    assert not input.size(2) % s and not input.size(3) % s

    # if input.size(1) == 3:
    #     # bgr2gray (same as opencv conversion matrix)
    #     input = (0.299 * input[:, 2] + 0.587 * input[:, 1] + 0.114 * input[:, 0]).unsqueeze(1)

    out = torch.cat([input[:, :, i::s, j::s] for i in range(s) for j in range(s)], dim=1)

    """
        out: torch.floatTensor (N, c*s**2, H/s, W/s)
    """
    return out


@MODELS.register_module()
class FDSR(BaseModule):
    """
    title: Towards Fast and Accurate Real-World Depth Super-Resolution: Benchmark Dataset and Baseline
    paper: https://arxiv.org/abs/2104.06174
    code: https://github.com/lingzhi96/RGB-D-D-Dataset/tree/main
    """

    def __init__(
        self,
        scale=2,
        lst_channels=1,
        gui_channels=10,
        num_feats=32,
        kernel_size=3,
        norm_flag=0,
        norm_dict={"mean": None, "std": None, "min": None, "max": None},
    ):
        super(FDSR, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.scale = scale
        self.img_upsampler = nn.Upsample(scale_factor=scale, mode="bicubic", align_corners=False)
        self.conv_rgb1 = nn.Conv2d(
            in_channels=gui_channels * 16, out_channels=num_feats, kernel_size=kernel_size, padding=1
        )
        # self.conv_rgb2 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        # self.conv_rgb3 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        # self.conv_rgb4 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        # self.conv_rgb5 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)

        self.rgb_cbl2 = oc.Conv_BN_ACT(
            in_channels=num_feats,
            out_channels=num_feats,
            kernel_size=kernel_size,
            alpha_in=0,
            alpha_out=0.25,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.rgb_cbl3 = oc.Conv_BN_ACT(
            in_channels=num_feats,
            out_channels=num_feats,
            kernel_size=kernel_size,
            alpha_in=0.25,
            alpha_out=0.25,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.rgb_cbl4 = oc.Conv_BN_ACT(
            in_channels=num_feats,
            out_channels=num_feats,
            kernel_size=kernel_size,
            alpha_in=0.25,
            alpha_out=0.25,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        # self.rgb_cbl5 = oc.Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0.125, alpha_out=0.125,
        #                            stride=1, padding=1,dilation=1,groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv_dp1 = nn.Conv2d(
            in_channels=lst_channels * 16, out_channels=num_feats, kernel_size=kernel_size, padding=1
        )
        self.MSB1 = MS_RB(num_feats, kernel_size)
        self.MSB2 = MS_RB(56, kernel_size)
        self.MSB3 = MS_RB(80, kernel_size)
        self.MSB4 = MS_RB(104, kernel_size)

        self.conv_recon1 = nn.Conv2d(in_channels=104, out_channels=4 * num_feats, kernel_size=kernel_size, padding=1)
        self.ps1 = nn.PixelShuffle(2)
        self.conv_recon2 = nn.Conv2d(
            in_channels=num_feats, out_channels=4 * num_feats, kernel_size=kernel_size, padding=1
        )
        self.ps2 = nn.PixelShuffle(2)
        self.restore = nn.Conv2d(in_channels=num_feats, out_channels=1, kernel_size=kernel_size, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, hr_guidance, **kwargs):

        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1:  # z-score
            assert self.norm_dict["mean"] is not None and self.norm_dict["std"] is not None
            x = (x - self.norm_dict["mean"]) / self.norm_dict["std"]
        elif self.norm_flag == 2:  # min-max
            assert self.norm_dict["min"] is not None and self.norm_dict["max"] is not None
            x = (x - self.norm_dict["min"]) / (self.norm_dict["max"] - self.norm_dict["min"])

        x = self.img_upsampler(x)

        re_im = resample_data(hr_guidance, 4)
        re_dp = resample_data(x, 4)

        dp_in = self.act(self.conv_dp1(re_dp))
        dp1 = self.MSB1(dp_in)

        rgb1 = self.act(self.conv_rgb1(re_im))
        # rgb2 = self.act(self.conv_rgb2(rgb1))

        rgb2 = self.rgb_cbl2(rgb1)

        ca1_in = torch.cat([dp1, rgb2[0]], dim=1)
        dp2 = self.MSB2(ca1_in)
        # rgb3 = self.conv_rgb3(rgb2)
        rgb3 = self.rgb_cbl3(rgb2)
        # ca2_in = dp2 + rgb3
        ca2_in = torch.cat([dp2, rgb3[0]], dim=1)

        dp3 = self.MSB3(ca2_in)
        # rgb4 = self.conv_rgb4(rgb3)
        rgb4 = self.rgb_cbl4(rgb3)

        # ca3_in = rgb4 + dp3
        ca3_in = torch.cat([dp3, rgb4[0]], dim=1)

        dp4 = self.MSB4(ca3_in)
        up1 = self.ps1(self.conv_recon1(self.act(dp4)))
        up2 = self.ps2(self.conv_recon2(up1))
        out = self.restore(up2)
        out = x + out

        if self.norm_flag == 1:
            out = out * self.norm_dict["std"] + self.norm_dict["mean"]
        elif self.norm_flag == 2:
            out = out * (self.norm_dict["max"] - self.norm_dict["min"]) + self.norm_dict["min"]

        return out


if __name__ == "__main__":
    lr_dep = torch.randn((1, 1, 56, 56)).cuda()
    hr_guidance = torch.randn((1, 10, 112, 112)).cuda()
    net = FDSR(2).cuda()
    out = net(lr_dep, hr_guidance)
    print(out.shape)
