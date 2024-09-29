import torch
import torch.nn as nn
from .mocolsk import *
from mmengine.model import BaseModule
from mmagic.registry import MODELS


@MODELS.register_module()
class OpticalSARMoCoLSKNet(BaseModule):
    def __init__(
        self,
        in_channels: int = 1,
        gui_channels: int = 2,
        num_feats: int = 32,
        kernel_size: int = 3,
        mocolsk_kernel_size: int = 3,
        scale: int = 1,
        module: str = "DynamicLSKBlock",
        n_resblocks: int = 4,
        num_stages: int = 4,  # num of stages
        reduction: int = 16,
        mlp_type:str = "a",
        norm_flag=0,
        norm_dict={"mean": None, "std": None, "min": None, "max": None},
    ):
        super().__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.num_stages = num_stages

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.lst_stem = nn.Conv2d(in_channels, num_feats, kernel_size=kernel_size, padding=kernel_size // 2)
        self.gui_stem = nn.Conv2d(gui_channels, num_feats, kernel_size=kernel_size, padding=kernel_size // 2)

        # residual groups for lst branch
        self.lst_rgs = nn.ModuleList(
            [
                ResidualGroup(
                    default_conv, (i + 1) * num_feats, kernel_size, reduction=reduction, n_resblocks=n_resblocks
                )
                for i in range(self.num_stages)
            ]
        )
        # residual groups for guidance branch
        self.gui_rgs = nn.ModuleList(
            [
                ResidualGroup(default_conv, num_feats, kernel_size, reduction=reduction, n_resblocks=n_resblocks)
                for _ in range(self.num_stages)
            ]
        )
        # mocolsk modules
        self.bridges = nn.ModuleList(
            [
                eval(module)(
                    lst_dim=(i + 1) * num_feats,
                    gui_dim=num_feats,
                    scale=scale,
                    hidden=32,
                    num_layers=1,
                    mlp_type=mlp_type,
                    kernel_size=mocolsk_kernel_size,
                )
                for i in range(self.num_stages)
            ]
        )

        # reconstruction module (decoder)
        n = self.num_stages + 1
        decoder = [
            ResidualGroup(default_conv, n * num_feats, kernel_size, reduction=reduction, n_resblocks=n_resblocks),
            DenseProjection(n * num_feats, n * num_feats, scale, up=True, bottleneck=False),
            ResidualGroup(default_conv, n * num_feats, kernel_size, reduction=reduction, n_resblocks=2 * n_resblocks),
            ResidualGroup(default_conv, n * num_feats, kernel_size, reduction=reduction, n_resblocks=2 * n_resblocks),
        ]
        self.decoder = nn.Sequential(*decoder)

        # projection head in reconstruction module
        proj_head = [
            default_conv(n * num_feats, num_feats, kernel_size=kernel_size, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            default_conv(num_feats, 3, kernel_size=kernel_size, bias=True),
        ]
        self.proj_head = nn.Sequential(*proj_head)
        self.bicubic = nn.Upsample(scale_factor=scale, mode="bicubic")

    def forward(self, lst, hr_guidance, **kwargs):
        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1:  # z-score
            assert self.norm_dict["mean"] is not None and self.norm_dict["std"] is not None
            lst = (lst - self.norm_dict["mean"]) / self.norm_dict["std"]
        elif self.norm_flag == 2:  # min-max
            assert self.norm_dict["min"] is not None and self.norm_dict["max"] is not None
            lst = (lst - self.norm_dict["min"]) / (self.norm_dict["max"] - self.norm_dict["min"])

        B, C, H, W = lst.shape
        identity = lst
        # stem convolution
        lst_feats = self.act(self.lst_stem(lst))
        gui_feats = self.act(self.gui_stem(hr_guidance))
        # stages, i.e., lst branch and guidance branch
        for i in range(self.num_stages):
            _lst_feats = self.lst_rgs[i](lst_feats)
            gui_feats = self.gui_rgs[i](gui_feats)
            lst_feats = self.bridges[i](_lst_feats, gui_feats)

        # decoder
        decoder_feats = self.decoder(lst_feats)
        # projection head
        out = self.proj_head(decoder_feats)

        out = out + self.bicubic(identity)

        if self.norm_flag == 1:
            out = out * self.norm_dict["std"] + self.norm_dict["mean"]
        elif self.norm_flag == 2:
            out = out * (self.norm_dict["max"] - self.norm_dict["min"]) + self.norm_dict["min"]

        return out


if __name__ == "__main__":
    hr_guidance = torch.randn((1, 10, 256, 256)).cuda()
    lst = torch.randn((1, 1, 128, 128)).cuda()
    net = OpticalSARMoCoLSKNet(in_channels=10, num_feats=32, kernel_size=3, scale=2).cuda()
    output = net(lst, hr_guidance)
    print(output.shape)
    pass
