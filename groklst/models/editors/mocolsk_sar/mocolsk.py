import torch.nn as nn
import torch
from .dynamic_mlp import *
import torch.nn.functional as F


class DynamicLSK(nn.Module):

    def __init__(self, dim, planes, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool_sizes = [1, 2, 3, 6]
        self.conv_lst = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial0 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.dynamic_mlp = FusionModule(
            inplanes=dim,
            planes=planes,
            hidden=hidden,
            num_layers=num_layers,
            mlp_type=mlp_type,
            kernel_size=kernel_size,
        )
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, lst, gui):
        B, C, H, W = lst.shape
        attn1 = self.conv_lst(lst)
        attn2 = self.conv_spatial0(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)

        # PPM
        lst_pools = []
        gui_pools = []
        for pool_size in self.pool_sizes:
            pool = F.adaptive_avg_pool2d(lst, (pool_size, pool_size))
            lst_pools.append(pool.view(B, C, -1))
            pool = F.adaptive_avg_pool2d(gui, (pool_size, pool_size))
            gui_pools.append(pool.view(B, C, -1))

        lst_pools = torch.cat(lst_pools, dim=2)
        lst_pools = lst_pools.permute(0, 2, 1)  # B, N, C

        gui_pools = torch.cat(gui_pools, dim=2)
        gui_pools = gui_pools.permute(0, 2, 1)  # B, N, C

        # Dynamic MLP
        weights = self.dynamic_mlp(lst_pools, gui_pools)  # B, N, C

        weights = torch.mean(weights, dim=1, keepdim=False).reshape(
            2, 2, self.kernel_size, self.kernel_size
        )  # in_chan=2, out_chan=2, kernel_size, kernel_size
        weights = nn.Parameter(data=weights, requires_grad=False)
        agg = F.conv2d(input=agg, weight=weights, stride=1, padding=self.kernel_size // 2, groups=1)
        sig = agg.sigmoid()
        attn1 = attn1 * sig[:, 0, :, :].unsqueeze(1)
        attn2 = attn2 * sig[:, 1, :, :].unsqueeze(1)
        out = self.conv(attn1 + attn2)

        return out

# MoCoLSK
class DynamicLSKBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="c", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = DynamicLSK(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats * gui], dim=1)
        out = self.down(feats)

        return out

#  DLSK2
class DynamicLSK2(nn.Module):

    def __init__(self, dim, planes, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        # lst
        self.conv5_lst = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv7_lst = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        # gui
        # self.conv5_gui = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # self.conv7_gui = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        self.avg_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.max_gap = nn.AdaptiveMaxPool2d((1, 1))
        self.dynamic_mlp = FusionModule(
            inplanes=dim,
            planes=planes,
            hidden=hidden,
            num_layers=num_layers,
            mlp_type=mlp_type,
            kernel_size=kernel_size,
        )
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, lst, gui):
        B, C, H, W = lst.shape
        attn1 = self.conv5_lst(lst)
        attn1 = self.conv7_lst(attn1)

        attn2 = self.conv5_lst(gui)
        attn2 = self.conv7_lst(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)

        lst_feats = self.avg_gap(lst).reshape(B, C)  # b,c
        gui_feats = self.avg_gap(gui).reshape(B, C)  # b,c

        # lst_feats = torch.cat([self.avg_gap(lst).reshape(B,C),self.max_gap(lst).reshape(B,C)],dim=-1)
        # gui_feats = torch.cat([self.avg_gap(gui).reshape(B,C),self.max_gap(gui).reshape(B,C)],dim=-1)
        weights = self.dynamic_mlp(lst_feats, gui_feats)
        weights = weights.reshape(
            2, 2, self.kernel_size, self.kernel_size
        )  # in_chan,out_chan, kernel_size[0], kernel_size[1]

        weights = nn.Parameter(data=weights, requires_grad=False)
        agg = F.conv2d(input=agg, weight=weights, stride=1, padding=1, groups=1)
        sig = agg.sigmoid()
        feats1 = attn1 * sig[:, 0, :, :].unsqueeze(1)
        feats2 = attn2 * sig[:, 1, :, :].unsqueeze(1)
        # out = self.conv(torch.cat([feats1,feats2], dim=1))
        out = self.conv(feats1 + feats2)

        return out * lst


class LSK(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)

        return x * attn

# LSKBlock
class LSKBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.lsk = LSK(2 * gui_dim)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        feats = self.lsk(torch.cat([lst_up, gui], dim=1))
        out = self.down(feats)

        return out


class SK(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, lst, gui):

        attn = torch.cat([lst, gui], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        feats = lst * sig[:, 0, :, :].unsqueeze(1) + gui * sig[:, 1, :, :].unsqueeze(1)
        out = self.conv(feats)

        return out

# SKBlock
class SKBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.sk = SK(dim=gui_dim)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        # SK
        feats = self.sk(lst_up, gui)
        out = self.down(torch.cat([feats, lst_up], dim=1))

        return out

# DSK
class DynamicSK(nn.Module):

    def __init__(self, dim, planes, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool_sizes = [1, 2, 3, 6]
        self.dynamic_mlp = FusionModule(
            inplanes=dim,
            planes=planes,
            hidden=hidden,
            num_layers=num_layers,
            mlp_type=mlp_type,
            kernel_size=kernel_size,
        )
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, lst, gui):
        B, C, H, W = lst.shape

        feats = torch.cat([lst, gui], dim=1)
        avg_feats = torch.mean(feats, dim=1, keepdim=True)
        max_feats, _ = torch.max(feats, dim=1, keepdim=True)
        agg = torch.cat([avg_feats, max_feats], dim=1)

        lst_pools = []
        gui_pools = []
        for pool_size in self.pool_sizes:
            pool = F.adaptive_avg_pool2d(lst, (pool_size, pool_size))
            lst_pools.append(pool.view(B, C, -1))
            pool = F.adaptive_avg_pool2d(gui, (pool_size, pool_size))
            gui_pools.append(pool.view(B, C, -1))

        lst_pools = torch.cat(lst_pools, dim=2)
        lst_pools = lst_pools.permute(0, 2, 1)  # B, N, C

        gui_pools = torch.cat(gui_pools, dim=2)
        gui_pools = gui_pools.permute(0, 2, 1)  # B, N, C

        weights = self.dynamic_mlp(lst_pools, gui_pools)  # B, N, C
        weights = torch.mean(weights, dim=1, keepdim=False).reshape(
            2, 2, self.kernel_size, self.kernel_size
        )  # in_chan,out_chan, kernel_size[0], kernel_size[1]

        weights = nn.Parameter(data=weights, requires_grad=False)
        agg = F.conv2d(input=agg, weight=weights, stride=1, padding=self.kernel_size // 2, groups=1)
        sig = agg.sigmoid()
        feats = lst * sig[:, 0, :, :].unsqueeze(1) + gui * sig[:, 1, :, :].unsqueeze(1)
        out = self.conv(feats)

        return out

# DSK2
class DynamicSK2(nn.Module):

    def __init__(self, dim, planes, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.max_gap = nn.AdaptiveMaxPool2d((1, 1))
        self.dynamic_mlp = FusionModule(
            inplanes=dim,
            planes=planes,
            hidden=hidden,
            num_layers=num_layers,
            mlp_type=mlp_type,
            kernel_size=kernel_size,
        )
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, lst, gui):
        B, C, H, W = lst.shape

        feats = torch.cat([lst, gui], dim=1)
        avg_attn = torch.mean(feats, dim=1, keepdim=True)
        max_attn, _ = torch.max(feats, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)

        lst_feats = self.avg_gap(lst).reshape(B, C)  # b,c
        gui_feats = self.avg_gap(gui).reshape(B, C)  # b,c

        weights = self.dynamic_mlp(lst_feats, gui_feats)
        weights = weights.reshape(
            2, 2, self.kernel_size, self.kernel_size
        )  # in_chan,out_chan, kernel_size[0], kernel_size[1]

        kernel = nn.Parameter(data=weights, requires_grad=False)
        agg = F.conv2d(input=agg, weight=kernel, stride=1, padding=1, groups=1)
        sig = agg.sigmoid()
        lst = lst * sig[:, 0, :, :].unsqueeze(1)
        gui = gui * sig[:, 1, :, :].unsqueeze(1)
        # out = self.conv(torch.cat([lst,gui], dim=1))
        out = self.conv(lst + gui)

        return out


class DynamicSKBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        # DynamicSK
        # self.dsk = DynamicSK(dim=gui_dim, planes=gui_dim, kernel_size=3)
        self.dsk = DynamicSK2(dim=gui_dim, planes=gui_dim)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        # DynamicSK
        feats = self.dsk(lst_up, gui)
        out = self.down(torch.cat([feats, lst_up], dim=1))

        return out

# SimpleBlock
class SimpleBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        out = self.down(torch.cat([lst_up, gui], dim=1))

        return out


# class CSExchange(nn.Module):
#     def __init__(self, dim, p=2, reduction=4) -> None:
#         super().__init__()
#         # lst
#         self.lst_conv = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#         )
#         # guidance
#         self.gui_conv = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#         )
#         self.p = p

#     def forward(self, lst, gui):
#         N, C, H, W = lst.shape
#         lst = self.lst_conv(lst)
#         gui = self.gui_conv(gui)

#         # channel exchange
#         channel_mask = torch.arange(C) % self.p == 0
#         channel_mask = channel_mask.unsqueeze(0).expand((N, -1))
#         out_lst = torch.zeros_like(lst)
#         out_gui = torch.zeros_like(gui)
#         out_lst[~channel_mask, ...] = lst[~channel_mask, ...]
#         out_gui[~channel_mask, ...] = gui[~channel_mask, ...]
#         out_lst[channel_mask, ...] = gui[channel_mask, ...]
#         out_gui[channel_mask, ...] = lst[channel_mask, ...]

#         # spatial exchange
#         spatial_mask = torch.arange(H) % self.p == 0
#         out_lst[..., ~spatial_mask] = lst[..., ~spatial_mask]
#         out_gui[..., ~spatial_mask] = gui[..., ~spatial_mask]
#         out_lst[..., spatial_mask] = gui[..., spatial_mask]
#         out_gui[..., spatial_mask] = lst[..., spatial_mask]

#         return out_lst, out_gui


# class CSExchangeBlock(nn.Module):
#     def __init__(self, lst_dim, gui_dim, scale):
#         super().__init__()
#         self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
#         self.cs_changer = CSExchange(gui_dim)
#         self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

#     def forward(self, depth, gui):
#         lst_up = self.lst_up(depth)
#         # DynamicSK
#         lst_feats, gui_feats = self.cs_changer(lst_up, gui)
#         out = self.down(torch.cat([lst_feats, gui_feats], dim=1))

#         return out


# class RandomCSExchange(nn.Module):
#     def __init__(self, dim, reduction=4) -> None:
#         super().__init__()
#         # lst
#         self.lst_conv = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#         )
#         # guidance
#         self.gui_conv = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#         )

#     def forward(self, lst, gui):
#         N, C, H, W = lst.shape

#         lst = self.lst_conv(lst)
#         gui = self.gui_conv(gui)

#         # random channel exchange
#         channel_mask = torch.randint(0, 2, size=(C,), dtype=torch.uint8)
#         channel_mask = channel_mask.type(torch.bool)
#         channel_mask = channel_mask.unsqueeze(0).expand((N, -1))
#         out_lst = torch.zeros_like(lst)
#         out_gui = torch.zeros_like(gui)
#         out_lst[~channel_mask, ...] = lst[~channel_mask, ...]
#         out_gui[~channel_mask, ...] = gui[~channel_mask, ...]
#         out_lst[channel_mask, ...] = gui[channel_mask, ...]
#         out_gui[channel_mask, ...] = lst[channel_mask, ...]

#         # random spatial exchange
#         spatial_mask = torch.randint(0, 2, size=(H,))
#         print(spatial_mask)
#         out_lst[..., ~spatial_mask] = lst[..., ~spatial_mask]
#         out_gui[..., ~spatial_mask] = gui[..., ~spatial_mask]
#         out_lst[..., spatial_mask] = gui[..., spatial_mask]
#         out_gui[..., spatial_mask] = lst[..., spatial_mask]

#         return out_lst, out_gui


# class ChannelExchange(nn.Module):
#     def __init__(self, dim, p=2, reduction=4) -> None:
#         super().__init__()
#         # lst
#         self.lst_conv = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#         )
#         # guidance
#         self.gui_conv = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#         )
#         self.p = p

#     def forward(self, lst, gui):
#         N, C, H, W = lst.shape
#         lst = self.lst_conv(lst)
#         gui = self.gui_conv(gui)

#         channel_mask = torch.arange(C) % self.p == 0
#         channel_mask = channel_mask.unsqueeze(0).expand((N, -1))
#         out_lst = torch.zeros_like(lst)
#         out_gui = torch.zeros_like(gui)
#         out_lst[~channel_mask, ...] = lst[~channel_mask, ...]
#         out_gui[~channel_mask, ...] = gui[~channel_mask, ...]
#         out_lst[channel_mask, ...] = gui[channel_mask, ...]
#         out_gui[channel_mask, ...] = lst[channel_mask, ...]

#         return out_lst, out_gui


# class SpatialExchange(nn.Module):
#     def __init__(self, dim, p=2, reduction=4) -> None:
#         super().__init__()
#         # lst
#         self.lst_conv = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#         )
#         # guidance
#         self.gui_conv = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#             nn.GELU(),
#             nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
#         )
#         self.p = p

#     def forward(self, lst, gui):
#         N, C, H, W = lst.shape
#         lst = self.lst_conv(lst)
#         gui = self.gui_conv(gui)

#         spatial_mask = torch.arange(H) % self.p == 0
#         out_lst = torch.zeros_like(lst)
#         out_gui = torch.zeros_like(gui)
#         out_lst[..., ~spatial_mask] = lst[..., ~spatial_mask]
#         out_gui[..., ~spatial_mask] = gui[..., ~spatial_mask]
#         out_lst[..., spatial_mask] = gui[..., spatial_mask]
#         out_gui[..., spatial_mask] = lst[..., spatial_mask]

#         return out_lst, out_gui


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {2: (6, 2, 2), 4: (8, 4, 2), 8: (12, 8, 2), 16: (20, 16, 2)}[scale]
    if up:
        conv_f = nn.ConvTranspose2d
    else:
        conv_f = nn.Conv2d

    return conv_f(in_channels, out_channels, kernel_size, stride=stride, padding=padding)


class DenseProjection(nn.Module):
    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        self.up = up
        if bottleneck:
            self.bottleneck = nn.Sequential(*[nn.Conv2d(in_channels, nr, 1), nn.PReLU(nr)])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels

        self.conv_1 = nn.Sequential(*[projection_conv(inter_channels, nr, scale, up), nn.PReLU(nr)])
        self.conv_2 = nn.Sequential(*[projection_conv(nr, inter_channels, scale, not up), nn.PReLU(inter_channels)])
        self.conv_3 = nn.Sequential(*[projection_conv(inter_channels, nr, scale, up), nn.PReLU(nr)])

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        a_0 = self.conv_1(x)
        b_0 = self.conv_2(a_0)
        e = b_0.sub(x)
        a_1 = self.conv_3(e)

        out = a_0.add(a_1)
        return out


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


# Residual Group (RG)


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                res_scale=1,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y