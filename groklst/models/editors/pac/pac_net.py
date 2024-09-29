"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import math
from collections import OrderedDict
from typing import Union
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from mmagic.registry import MODELS

from mmengine.model import BaseModule
from .pac import PacConvTranspose2d


def convert_to_single_channel(x):
    bs, ch, h, w = x.shape
    if ch != 1:
        x = x.reshape(bs * ch, 1, h, w)
    return x, ch


def recover_from_single_channel(x, ch):
    if ch != 1:
        bs_ch, _ch, h, w = x.shape
        assert _ch == 1
        assert bs_ch % ch == 0
        x = x.reshape(bs_ch // ch, ch, h, w)
    return x


def repeat_for_channel(x, ch):
    if ch != 1:
        bs, _ch, h, w = x.shape
        x = x.repeat(1, ch, 1, 1).reshape(bs * ch, _ch, h, w)
    return x


def th_rmse(pred, gt):
    return (pred - gt).pow(2).mean(dim=3).mean(dim=2).sum(dim=1).sqrt().mean()


def th_epe(pred, gt, small_flow=-1.0, unknown_flow_thresh=1e7):
    pred_u, pred_v = pred[:, 0].contiguous().view(-1), pred[:, 1].contiguous().view(-1)
    gt_u, gt_v = gt[:, 0].contiguous().view(-1), gt[:, 1].contiguous().view(-1)
    if gt_u.abs().max() > unknown_flow_thresh or gt_v.abs().max() > unknown_flow_thresh:
        idx_unknown = ((gt_u.abs() > unknown_flow_thresh) + (gt_v.abs() > unknown_flow_thresh)).nonzero()[:, 0]
        pred_u[idx_unknown] = 0
        pred_v[idx_unknown] = 0
        gt_u[idx_unknown] = 0
        gt_v[idx_unknown] = 0
    epe = ((pred_u - gt_u).pow(2) + (pred_v - gt_v).pow(2)).sqrt()
    if small_flow >= 0.0 and (gt_u.abs().min() <= small_flow or gt_v.abs().min() <= small_flow):
        idx_valid = ((gt_u.abs() > small_flow) + (gt_v.abs() > small_flow)).nonzero()[:, 0]
        epe = epe[idx_valid]
    return epe.mean()


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class JointBilateral(nn.Module):
    def __init__(self, scale_factor, channels, kernel_size, scale_space, scale_color):
        super(JointBilateral, self).__init__()
        self.channels = channels
        self.scale_space = float(scale_space)
        self.scale_color = float(scale_color)
        self.convt = PacConvTranspose2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=scale_factor,
            dilation=1,
            padding=1 + int((kernel_size - scale_factor - 1) // 2),
            output_padding=(kernel_size - scale_factor) % 2,
            normalize_kernel=True,
            bias=None,
        )
        self.convt.weight.data.fill_(0.0)
        for c in range(channels):
            self.convt.weight.data[c, c] = 1.0

    def forward(self, x, hr_guidance):
        x, C = convert_to_single_channel(x)
        bs, ch, h, w = hr_guidance.shape
        hh = th.arange(h, dtype=hr_guidance.dtype, device=hr_guidance.device)
        ww = th.arange(w, dtype=hr_guidance.dtype, device=hr_guidance.device)
        hr_guidance = th.cat(
            [
                hr_guidance * self.scale_color,
                hh.view(-1, 1).expand(bs, 1, -1, w) * self.scale_space,
                ww.expand(bs, 1, h, -1) * self.scale_space,
            ],
            dim=1,
        )
        hr_guidance = repeat_for_channel(hr_guidance, C)

        x = self.convt(x, hr_guidance)
        x = recover_from_single_channel(x, C)
        return x


class Bilinear(nn.Module):
    def __init__(self, scale_factor, channels=None, guide_channels=None):
        super(Bilinear, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x, hr_guidance):
        return F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)


class DJIF(nn.Module):
    def __init__(self, scale_factor, channels=1, guide_channels=3, fs=(9, 1, 5), ns_tg=(96, 48, 1), ns_f=(64, 32)):
        super(DJIF, self).__init__()
        assert fs[0] % 2 == 1 and fs[1] % 2 == 1 and fs[2] % 2 == 1
        paddings = tuple(f // 2 for f in fs)
        paddings_tg = sum(paddings) // 3, sum(paddings) // 3, sum(paddings) - 2 * (sum(paddings) // 3)
        self.scale_factor = scale_factor
        self.channels = channels
        self.guide_channels = guide_channels
        self.branch_t = nn.Sequential(
            nn.Conv2d(channels, ns_tg[0], kernel_size=fs[0], padding=paddings_tg[0]),
            nn.ReLU(),
            nn.Conv2d(ns_tg[0], ns_tg[1], kernel_size=fs[1], padding=paddings_tg[1]),
            nn.ReLU(),
            nn.Conv2d(ns_tg[1], ns_tg[2], kernel_size=fs[2], padding=paddings_tg[2]),
        )
        self.branch_g = nn.Sequential(
            nn.Conv2d(guide_channels, ns_tg[0], kernel_size=fs[0], padding=paddings_tg[0]),
            nn.ReLU(),
            nn.Conv2d(ns_tg[0], ns_tg[1], kernel_size=fs[1], padding=paddings_tg[1]),
            nn.ReLU(),
            nn.Conv2d(ns_tg[1], ns_tg[2], kernel_size=fs[2], padding=paddings_tg[2]),
        )
        self.branch_joint = nn.Sequential(
            nn.Conv2d(ns_tg[2] * 2, ns_f[0], kernel_size=fs[0], padding=paddings[0]),
            nn.ReLU(),
            nn.Conv2d(ns_f[0], ns_f[1], kernel_size=fs[1], padding=paddings[1]),
            nn.ReLU(),
            nn.Conv2d(ns_f[1], channels, kernel_size=fs[2], padding=paddings[2]),
        )

    def forward(self, x, hr_guidance):
        x, C = convert_to_single_channel(x)
        if x.shape[-1] < hr_guidance.shape[-1]:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        x = self.branch_t(x)
        hr_guidance = self.branch_g(hr_guidance)
        hr_guidance = repeat_for_channel(hr_guidance, C)
        x = self.branch_joint(th.cat((x, hr_guidance), dim=1))
        x = recover_from_single_channel(x, C)
        return x


class DJIFWide(DJIF):
    def __init__(self, scale_factor, channels=1, guide_channels=3):
        super(DJIFWide, self).__init__(scale_factor, channels, guide_channels, ns_tg=(256, 128, 1), ns_f=(256, 128))


# PAC
@MODELS.register_module()
class PAC(BaseModule):
    """
    title: Pixel-Adaptive Convolutional Neural Networks
    paper: https://arxiv.org/abs/1904.05373 or https://ieeexplore.ieee.org/document/8954063
    code: https://github.com/NVlabs/pacnet
    """
    def __init__(
        self,
        scale_factor=2,
        channels=1,
        guide_channels=3,
        n_t_layers=3,
        n_g_layers=3,
        n_f_layers=2,
        n_t_filters: Union[int, tuple] = 32,
        n_g_filters: Union[int, tuple] = 32,
        n_f_filters: Union[int, tuple] = 32,
        k_ch=16,
        f_sz_1=5,
        f_sz_2=5,
        t_bn=False,
        g_bn=False,
        u_bn=False,
        f_bn=False,
        norm_flag=0, 
        norm_dict={'mean':None, 'std':None,'min':None, 'max':None}
    ):
        super(PAC, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.channels = channels
        self.guide_channels = guide_channels
        self.scale_factor = scale_factor
        self.branch_t = None
        self.branch_g = None
        self.branch_f = None
        self.k_ch = k_ch

        assert n_g_layers >= 1, "Guidance branch should have at least one layer"
        assert n_f_layers >= 1, "Final prediction branch should have at least one layer"
        assert math.log2(scale_factor) % 1 == 0, "scale_factor needs to be a power of 2"
        assert f_sz_1 % 2 == 1, "filter size needs to be an odd number"
        num_ups = int(math.log2(scale_factor))  # number of 2x up-sampling operations
        pad = int(f_sz_1 // 2)

        if type(n_t_filters) == int:
            n_t_filters = (n_t_filters,) * n_t_layers
        else:
            assert len(n_t_filters) == n_t_layers

        if type(n_g_filters) == int:
            n_g_filters = (n_g_filters,) * (n_g_layers - 1)
        else:
            assert len(n_g_filters) == n_g_layers - 1

        if type(n_f_filters) == int:
            n_f_filters = (n_f_filters,) * (n_f_layers + num_ups - 1)
        else:
            assert len(n_f_filters) == n_f_layers + num_ups - 1

        # target branch
        t_layers = []
        n_t_channels = (channels,) + n_t_filters
        for l in range(n_t_layers):
            t_layers.append(
                (
                    "conv{}".format(l + 1),
                    nn.Conv2d(n_t_channels[l], n_t_channels[l + 1], kernel_size=f_sz_1, padding=pad),
                )
            )
            if t_bn:
                t_layers.append(("bn{}".format(l + 1), nn.BatchNorm2d(n_t_channels[l + 1])))
            if l < n_t_layers - 1:
                t_layers.append(("relu{}".format(l + 1), nn.ReLU()))
        self.branch_t = nn.Sequential(OrderedDict(t_layers))

        # guidance branch
        g_layers = []
        n_g_channels = (guide_channels,) + n_g_filters + (k_ch * num_ups,)
        for l in range(n_g_layers):
            g_layers.append(
                (
                    "conv{}".format(l + 1),
                    nn.Conv2d(n_g_channels[l], n_g_channels[l + 1], kernel_size=f_sz_1, padding=pad),
                )
            )
            if g_bn:
                g_layers.append(("bn{}".format(l + 1), nn.BatchNorm2d(n_g_channels[l + 1])))
            if l < n_g_layers - 1:
                g_layers.append(("relu{}".format(l + 1), nn.ReLU()))
        self.branch_g = nn.Sequential(OrderedDict(g_layers))

        # upsampling layers
        p, op = int((f_sz_2 - 1) // 2), (f_sz_2 % 2)
        self.up_convts = nn.ModuleList()
        self.up_bns = nn.ModuleList()
        n_f_channels = (n_t_channels[-1],) + n_f_filters + (channels,)
        for l in range(num_ups):
            self.up_convts.append(
                PacConvTranspose2d(
                    n_f_channels[l], n_f_channels[l + 1], kernel_size=f_sz_2, stride=2, padding=p, output_padding=op
                )
            )
            if u_bn:
                self.up_bns.append(nn.BatchNorm2d(n_f_channels[l + 1]))

        # final prediction branch
        f_layers = []
        for l in range(n_f_layers):
            f_layers.append(
                (
                    "conv{}".format(l + 1),
                    nn.Conv2d(
                        n_f_channels[l + num_ups], n_f_channels[l + num_ups + 1], kernel_size=f_sz_1, padding=pad
                    ),
                )
            )
            if f_bn:
                f_layers.append(("bn{}".format(l + 1), nn.BatchNorm2d(n_f_channels[l + num_ups + 1])))
            if l < n_f_layers - 1:
                f_layers.append(("relu{}".format(l + 1), nn.ReLU()))
        self.branch_f = nn.Sequential(OrderedDict(f_layers))

    def forward(self, x, hr_guidance, **kwargs):
        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])

        
        # x, C = convert_to_single_channel(x)
        B, C, H, W= x.shape

        x = self.branch_t(x)
        hr_guidance = self.branch_g(hr_guidance)
        for i in range(len(self.up_convts)):
            scale = math.pow(2, i + 1) / self.scale_factor
            guide_cur = hr_guidance[:, (i * self.k_ch) : ((i + 1) * self.k_ch)]
            if scale != 1:
                guide_cur = F.interpolate(guide_cur, scale_factor=scale, align_corners=False, mode="bilinear")
            guide_cur = repeat_for_channel(guide_cur, C)
            x = self.up_convts[i](x, guide_cur)
            if self.up_bns:
                x = self.up_bns[i](x)
            x = F.relu(x)
        out = self.branch_f(x)
        # x = recover_from_single_channel(x, C)
        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out


class PacJointUpsampleLite(PAC):
    def __init__(self, scale_factor, channels=1, guide_channels=3):
        if scale_factor == 4 or scale_factor == 2:
            args = dict(n_g_filters=(12, 22), n_t_filters=(12, 16, 22), n_f_filters=(12, 16, 22), k_ch=12)
        elif scale_factor == 8:
            args = dict(n_g_filters=(12, 16), n_t_filters=(12, 16, 16), n_f_filters=(12, 16, 16, 20), k_ch=12)
        elif scale_factor == 16:
            args = dict(n_g_filters=(8, 16), n_t_filters=(8, 16, 16), n_f_filters=(8, 16, 16, 16, 16), k_ch=10)
        else:
            raise ValueError("scale_factor can only be 4, 8, or 16.")
        super(PacJointUpsampleLite, self).__init__(scale_factor, channels, guide_channels, **args)


if __name__ == "__main__":
    lr_dep = torch.randn((1, 1, 56, 56))
    rgb = torch.randn((1, 10, 448, 448))
    # net = PacJointUpsampleLite(2)
    net = PAC(8, guide_channels=10)
    out = net(lr_dep, rgb)
    print(out.shape)
