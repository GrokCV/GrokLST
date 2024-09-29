import torch.nn as nn
import torch
from .dynamic_mlp import *
import torch.nn.functional as F

__all__ = [ 'ResidualGroup', 'CALayer', 
            'RCAB', 'ResBlock', 'DenseProjection', 
            'projection_conv', 'default_conv',
            # -------------------------------------------------------------
            'BaselineWithCatBlock', 'BaselineWithoutCatBlock', 
            'CSExchangeWithCatBlock', 'CSExchangeWithoutCatBlock',
            'RandomCSExchangeWithCatBlock', 'RandomCSExchangeWithoutCatBlock',
            'ChannelExchangeWithCatBlock', 'ChannelExchangeWithoutCatBlock',
            'SpatialExchangeWithCatBlock', 'SpatialExchangeWithoutCatBlock',
            'RandomChannelExchangeWithCatBlock', 'RandomChannelExchangeWithoutCatBlock',
            'RandomSpatialExchangeWithCatBlock', 'RandomSpatialExchangeWithoutCatBlock',
            
            'ChannelExchangeWithConvCatBlock', "SpatialExchangeWithConvCatBlock"]

class CSExchange(nn.Module):
    def __init__(self, dim, p=2, reduction=4) -> None:
        super().__init__()
        self.p = p

    def forward(self, lst, gui):
        N, C, H, W = lst.shape

        # channel exchange
        channel_mask = torch.arange(C) % self.p == 0
        channel_mask = channel_mask.unsqueeze(0).expand((N, -1))
        out_lst = torch.zeros_like(lst)
        out_gui = torch.zeros_like(gui)
        out_lst[~channel_mask, ...] = lst[~channel_mask, ...]
        out_gui[~channel_mask, ...] = gui[~channel_mask, ...]
        out_lst[channel_mask, ...] = gui[channel_mask, ...]
        out_gui[channel_mask, ...] = lst[channel_mask, ...]

        # spatial exchange
        spatial_mask = torch.arange(H) % self.p == 0
        out_lst[..., ~spatial_mask] = lst[..., ~spatial_mask]
        out_gui[..., ~spatial_mask] = gui[..., ~spatial_mask]
        out_lst[..., spatial_mask] = gui[..., spatial_mask]
        out_gui[..., spatial_mask] = lst[..., spatial_mask]

        return out_lst, out_gui


class CSExchangeWithCatBlock(nn.Module):
    """changer

    Args:
        nn (_type_): _description_
    """
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = CSExchange(gui_dim)
        self.down = DenseProjection(2*gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        lst_feats, gui_feats = self.cs_changer(lst_up, gui)
        out = self.down(torch.cat([lst_feats, gui_feats], dim=1))

        return out
    


class CSExchangeWithoutCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = CSExchange(gui_dim)
        self.down = DenseProjection(gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        lst_feats, _ = self.cs_changer(lst_up, gui)
        out = self.down(lst_feats)

        return out

class BaselineWithCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.down = DenseProjection(2*gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        out = self.down(torch.cat([lst_up, gui], dim=1))

        return out


class BaselineWithoutCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.down = DenseProjection(gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        out = self.down(lst_up)

        return out



class RandomCSExchange(nn.Module):
    def __init__(self, dim, reduction=4) -> None:
        super().__init__()

    def forward(self, lst, gui):
        N, C, H, W = lst.shape

        # random channel exchange
        channel_mask = torch.randint(0, 2, size=(C,), dtype=torch.uint8)
        channel_mask = channel_mask.type(torch.bool)
        channel_mask = channel_mask.unsqueeze(0).expand((N, -1))
        out_lst = torch.zeros_like(lst)
        out_gui = torch.zeros_like(gui)
        out_lst[~channel_mask, ...] = lst[~channel_mask, ...]
        out_gui[~channel_mask, ...] = gui[~channel_mask, ...]
        out_lst[channel_mask, ...] = gui[channel_mask, ...]
        out_gui[channel_mask, ...] = lst[channel_mask, ...]

        # random spatial exchange
        spatial_mask = torch.randint(0, 2, size=(H,))
        # print(spatial_mask)
        out_lst[..., ~spatial_mask] = lst[..., ~spatial_mask]
        out_gui[..., ~spatial_mask] = gui[..., ~spatial_mask]
        out_lst[..., spatial_mask] = gui[..., spatial_mask]
        out_gui[..., spatial_mask] = lst[..., spatial_mask]

        return out_lst, out_gui



class RandomCSExchangeWithCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = RandomCSExchange(gui_dim)
        self.down = DenseProjection(2*gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        lst_feats, gui_feats = self.cs_changer(lst_up, gui)
        out = self.down(torch.cat([lst_feats, gui_feats], dim=1))

        return out

class RandomCSExchangeWithoutCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = RandomCSExchange(gui_dim)
        self.down = DenseProjection(gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        lst_feats, _ = self.cs_changer(lst_up, gui)
        out = self.down(lst_feats)

        return out

class ChannelExchange(nn.Module):
    def __init__(self, dim, p=2, reduction=4) -> None:
        super().__init__()
        self.p = p

    def forward(self, lst, gui):
        N, C, H, W = lst.shape

        channel_mask = torch.arange(C) % self.p == 0
        channel_mask = channel_mask.unsqueeze(0).expand((N, -1))
        out_lst = torch.zeros_like(lst)
        out_gui = torch.zeros_like(gui)
        out_lst[~channel_mask, ...] = lst[~channel_mask, ...]
        out_gui[~channel_mask, ...] = gui[~channel_mask, ...]
        out_lst[channel_mask, ...] = gui[channel_mask, ...]
        out_gui[channel_mask, ...] = lst[channel_mask, ...]

        return out_lst, out_gui


class ChannelExchangeWithCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = ChannelExchange(gui_dim)
        self.down = DenseProjection(2*gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        lst_feats, gui_feats = self.cs_changer(lst_up, gui)
        out = self.down(torch.cat([lst_feats, gui_feats], dim=1))

        return out

class ChannelExchangeWithConv(nn.Module):
    def __init__(self, dim, p=2, reduction=4) -> None:
        super().__init__()
        self.p = p
        self.conv1 = nn.Conv2d(dim//2 , dim//2, 1, 1)
        self.conv2 = nn.Conv2d(dim//2 , dim//2, 1, 1)

    def forward(self, lst, gui):
        N, C, H, W = lst.shape

        channel_mask = torch.arange(C) % self.p == 0
        channel_mask = channel_mask.unsqueeze(0).expand((N, -1))
        out_lst = torch.zeros_like(lst)
        out_gui = torch.zeros_like(gui)
        out_lst[~channel_mask, ...] = lst[~channel_mask, ...]
        out_gui[~channel_mask, ...] = gui[~channel_mask, ...]
        out_lst[channel_mask, ...] = self.conv2(gui[channel_mask, ...])
        out_gui[channel_mask, ...] = self.conv1(lst[channel_mask, ...])

        return out_lst, out_gui


class ChannelExchangeWithConvCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = ChannelExchangeWithConv(gui_dim)
        self.down = DenseProjection(2*gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        lst_feats, gui_feats = self.cs_changer(lst_up, gui)
        out = self.down(torch.cat([lst_feats, gui_feats], dim=1))

        return out

class ChannelExchangeWithoutCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = ChannelExchange(gui_dim)
        self.down = DenseProjection(gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        lst_feats, _ = self.cs_changer(lst_up, gui)
        out = self.down(lst_feats)

        return out

class SpatialExchange(nn.Module):
    def __init__(self, dim, p=2, reduction=4) -> None:
        super().__init__()
        self.p = p

    def forward(self, lst, gui):
        N, C, H, W = lst.shape

        spatial_mask = torch.arange(H) % self.p == 0
        out_lst = torch.zeros_like(lst)
        out_gui = torch.zeros_like(gui)
        out_lst[..., ~spatial_mask] = lst[..., ~spatial_mask]
        out_gui[..., ~spatial_mask] = gui[..., ~spatial_mask]
        out_lst[..., spatial_mask] = gui[..., spatial_mask]
        out_gui[..., spatial_mask] = lst[..., spatial_mask]

        return out_lst, out_gui



class SpatialExchangeWithCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = SpatialExchange(gui_dim)
        self.down = DenseProjection(2*gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        lst_feats, gui_feats = self.cs_changer(lst_up, gui)
        out = self.down(torch.cat([lst_feats, gui_feats], dim=1))

        return out


class SpatialExchangeWithConv(nn.Module):
    def __init__(self, dim, p=2, reduction=4) -> None:
        super().__init__()
        self.p = p
        self.conv1 = nn.Conv2d(dim , dim, 1, 1)
        self.conv2 = nn.Conv2d(dim , dim, 1, 1)

    def forward(self, lst, gui):
        N, C, H, W = lst.shape

        spatial_mask = torch.arange(H) % self.p == 0
        out_lst = torch.zeros_like(lst)
        out_gui = torch.zeros_like(gui)
        out_lst[..., ~spatial_mask] = lst[..., ~spatial_mask]
        out_gui[..., ~spatial_mask] = gui[..., ~spatial_mask]
        out_lst[..., spatial_mask] = self.conv2(gui[..., spatial_mask])
        out_gui[..., spatial_mask] = self.conv1(lst[..., spatial_mask])

        return out_lst, out_gui



class SpatialExchangeWithConvCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = SpatialExchangeWithConv(gui_dim)
        self.down = DenseProjection(2*gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        lst_feats, gui_feats = self.cs_changer(lst_up, gui)
        out = self.down(torch.cat([lst_feats, gui_feats], dim=1))

        return out

class SpatialExchangeWithoutCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = SpatialExchange(gui_dim)
        self.down = DenseProjection(gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        lst_feats, _ = self.cs_changer(lst_up, gui)
        out = self.down(lst_feats)

        return out


class RandomChannelExchange(nn.Module):
    def __init__(self, dim, reduction=4) -> None:
        super().__init__()

    def forward(self, lst, gui):
        N, C, H, W = lst.shape
        out_lst = torch.zeros_like(lst)
        out_gui = torch.zeros_like(gui)

        # random channel exchange
        channel_mask = torch.randint(0, 2, size=(C,), dtype=torch.uint8)
        channel_mask = channel_mask.type(torch.bool)
        channel_mask = channel_mask.unsqueeze(0).expand((N, -1))
        out_lst[~channel_mask, ...] = lst[~channel_mask, ...]
        out_gui[~channel_mask, ...] = gui[~channel_mask, ...]
        out_lst[channel_mask, ...] = gui[channel_mask, ...]
        out_gui[channel_mask, ...] = lst[channel_mask, ...]

        return out_lst, out_gui


class RandomChannelExchangeWithCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = RandomChannelExchange(gui_dim)
        self.down = DenseProjection(2*gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        lst_feats, gui_feats = self.cs_changer(lst_up, gui)
        out = self.down(torch.cat([lst_feats, gui_feats], dim=1))

        return out
    

class RandomChannelExchangeWithoutCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = RandomChannelExchange(gui_dim)
        self.down = DenseProjection(gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        lst_feats, _ = self.cs_changer(lst_up, gui)
        out = self.down(lst_feats)

        return out

class RandomSpatialExchange(nn.Module):
    def __init__(self, dim, reduction=4) -> None:
        super().__init__()

    def forward(self, lst, gui):
        N, C, H, W = lst.shape
        out_lst = torch.zeros_like(lst)
        out_gui = torch.zeros_like(gui)

        # random spatial exchange
        spatial_mask = torch.randint(0, 2, size=(H,))
        # print(spatial_mask)
        out_lst[..., ~spatial_mask] = lst[..., ~spatial_mask]
        out_gui[..., ~spatial_mask] = gui[..., ~spatial_mask]
        out_lst[..., spatial_mask] = gui[..., spatial_mask]
        out_gui[..., spatial_mask] = lst[..., spatial_mask]

        return out_lst, out_gui


class RandomSpatialExchangeWithCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = RandomSpatialExchange(gui_dim)
        self.down = DenseProjection(2*gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        lst_feats, gui_feats = self.cs_changer(lst_up, gui)
        out = self.down(torch.cat([lst_feats, gui_feats], dim=1))

        return out
    

class RandomSpatialExchangeWithoutCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = RandomSpatialExchange(gui_dim)
        self.down = DenseProjection(gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, weights=None):
        lst_up = self.lst_up(depth)
        lst_feats, _ = self.cs_changer(lst_up, gui)
        out = self.down(lst_feats)

        return out


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
