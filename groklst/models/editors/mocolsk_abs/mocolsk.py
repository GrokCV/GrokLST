import torch.nn as nn
import torch
from .dynamic_mlp import *
import torch.nn.functional as F

# BaselineBlock
class BaselineBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        out = self.down(torch.cat([lst_up, gui], dim=1))

        return out


class MoCoLSKConv(nn.Module):
    """LST -> LSK Pathway & return attn * gui

    """
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

        return out*gui


# MoCoLSK Module
class MoCoLSKBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoLSKConv(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out


class MoCoLSKConvWithout_MF(nn.Module):
    """LST -> LSK Pathway & return attn * gui

    """
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


# MoCoLSK Module
class MoCoLSKBlockWithout_MF(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoLSKConvWithout_MF(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out



class MoCoLSKConv_squeeze(nn.Module):
    """LST -> LSK Pathway & return attn * gui

    """
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
        lst_pools = torch.mean(lst_pools.permute(0, 2, 1), dim=1)  # B, N, C

        gui_pools = torch.cat(gui_pools, dim=2)
        gui_pools = torch.mean(gui_pools.permute(0, 2, 1), dim=1)  # B, N, C

        # Dynamic MLP
        weights = self.dynamic_mlp(lst_pools, gui_pools)  # B, N, C

        weights = weights.reshape(
            2, 2, self.kernel_size, self.kernel_size
        )  # in_chan=2, out_chan=2, kernel_size, kernel_size
        weights = nn.Parameter(data=weights, requires_grad=False)
        agg = F.conv2d(input=agg, weight=weights, stride=1, padding=self.kernel_size // 2, groups=1)
        sig = agg.sigmoid()
        attn1 = attn1 * sig[:, 0, :, :].unsqueeze(1)
        attn2 = attn2 * sig[:, 1, :, :].unsqueeze(1)
        out = self.conv(attn1 + attn2)

        return out*gui


# MoCoLSK Module
class MoCoLSKBlock_squeeze(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoLSKConv_squeeze(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out


class MoCoLSKPythwayConv(nn.Module):
    """LST -> LSK Pathway & return attn * gui & (only use LSK pythway)
        静态卷积
    """
    def __init__(self, dim, planes, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool_sizes = [1, 2, 3, 6]
        self.conv_lst = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial0 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
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
        sig = self.conv_squeeze(agg).sigmoid()

        attn1 = attn1 * sig[:, 0, :, :].unsqueeze(1)
        attn2 = attn2 * sig[:, 1, :, :].unsqueeze(1)
        out = self.conv(attn1 + attn2)

        return out*gui


class MoCoLSKPathwayBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoLSKPythwayConv(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out



class MoCoLSKPythwayConv_Guidance(nn.Module):
    """guidance -> LSK Pathway & return attn * lst & (only use LSK pythway)

    """
    def __init__(self, dim, planes, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool_sizes = [1, 2, 3, 6]
        self.conv_lst = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial0 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, lst, gui):
        B, C, H, W = lst.shape
        attn1 = self.conv_lst(gui)
        attn2 = self.conv_spatial0(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()

        attn1 = attn1 * sig[:, 0, :, :].unsqueeze(1)
        attn2 = attn2 * sig[:, 1, :, :].unsqueeze(1)
        out = self.conv(attn1 + attn2)

        return out*lst


class MoCoLSKPathwayBlock_Guidance(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoLSKPythwayConv_Guidance(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out



class MoCoOneLKConv(nn.Module):
    """RF=23, one large kernel convolution.

    """
    def __init__(self, dim, planes, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        # self.kernel_size = kernel_size
        # self.pool_sizes = [1, 2, 3, 6]
        self.lkconv = nn.Conv2d(dim, dim, 23, padding=11, groups=dim)

        # self.dynamic_mlp = FusionModule12(
        #     inplanes=dim,
        #     planes=planes,
        #     hidden=hidden,
        #     num_layers=num_layers,
        #     mlp_type=mlp_type,
        #     kernel_size=kernel_size,
        # )
        # self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, lst, gui):
        B, C, H, W = lst.shape
        out = self.lkconv(lst)

        # avg_attn = torch.mean(attn, dim=1, keepdim=True)
        # max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        # agg = torch.cat([avg_attn, max_attn], dim=1)

        # # PPM
        # lst_pools = []
        # gui_pools = []
        # for pool_size in self.pool_sizes:
        #     pool = F.adaptive_avg_pool2d(lst, (pool_size, pool_size))
        #     lst_pools.append(pool.view(B, C, -1))
        #     pool = F.adaptive_avg_pool2d(gui, (pool_size, pool_size))
        #     gui_pools.append(pool.view(B, C, -1))

        # lst_pools = torch.cat(lst_pools, dim=2)
        # lst_pools = lst_pools.permute(0, 2, 1)  # B, N, C

        # gui_pools = torch.cat(gui_pools, dim=2)
        # gui_pools = gui_pools.permute(0, 2, 1)  # B, N, C

        # # Dynamic MLP
        # weights = self.dynamic_mlp(lst_pools, gui_pools)  # B, N, C

        # weights = torch.mean(weights, dim=1, keepdim=False).reshape(
        #     1, 2, self.kernel_size, self.kernel_size
        # )  # in_chan=2, out_chan=2, kernel_size, kernel_size
        # weights = nn.Parameter(data=weights, requires_grad=False)
        # agg = F.conv2d(input=agg, weight=weights, stride=1, padding=self.kernel_size // 2, groups=1)
        # sig = agg.sigmoid()
        # attn = attn * sig
        # out = self.conv(attn)

        return out*gui


class MoCoOneLKBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoOneLKConv(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out


class MoCoLSKConv_M(nn.Module):
    """ LST -> LSK Pathway & return attn * lst
        不进行模态融合
    """
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

        return out * lst


class MoCoLSK_MBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoLSKConv_M(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out

# todo
class MoCoLSKCSConv(nn.Module):
    """ MoCoLSKCSConv: MoCoLSK with Channel Selection. 
    """
    def __init__(self, dim, planes, hidden=32, num_layers=1, mlp_type="a", 
                 weights_dim=None, M=2, r=2, L=32):
        """Constructor
        Args:
            dim: input channel dimensionality.
            M: the number of branchs.
            r: the radio for compute d, the length of z.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(MoCoLSKCSConv, self).__init__()
        d = max(int(dim / r), L)
        weights_dim = d
        self.M = M
        self.dim = dim
        dim = dim
        self.lkconv1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.lkconv2 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

        self.fc = nn.Linear(dim, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, dim))
        self.softmax = nn.Softmax(dim=1)

        self.pool_sizes = [1, 2, 3, 6]
        self.dynamic_mlp = CSFusionModule(
            inplanes=dim,
            planes=planes,
            hidden=hidden,
            num_layers=num_layers,
            mlp_type=mlp_type,
            weights_dim=weights_dim,
        )

    def forward(self, lst, gui):
        B, C, H, W = lst.shape
        
        lk_fea1 = self.lkconv1(lst)
        lk_fea2 = self.lkconv2(lk_fea1)
        fea1 = self.conv1(lk_fea1).unsqueeze(dim=1)
        fea2 = self.conv2(lk_fea2).unsqueeze(dim=1)

        feas = torch.cat([fea1, fea2], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # GAP for LSK pathway
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s) # B C

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
        weights = torch.mean(weights, dim=1, keepdim=False) # B C
        
        fea_z = fea_z + weights

        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        attn = (feas * attention_vectors).sum(dim=1)

        out = attn * gui

        return out


class MoCoLSKCSBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoLSKCSConv(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out


class MoCoLSKConvWithoutPPM_Avg(nn.Module):
    """
    LST -> LSK Pathway & return attn * gui & (use avg, not PPM)
    """
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

        # average pooling, not PPM
        lst_pools = F.adaptive_avg_pool2d(lst, (1, 1)).reshape(B, C)  # b,c
        gui_pools = F.adaptive_avg_pool2d(gui, (1, 1)).reshape(B, C)  # b,c

        # Dynamic MLP
        weights = self.dynamic_mlp(lst_pools, gui_pools)  # B, C
        # in_chan=2, out_chan=2, kernel_size, kernel_size
        weights = weights.reshape(2, 2, self.kernel_size, self.kernel_size)  
        weights = nn.Parameter(data=weights, requires_grad=False)
        agg = F.conv2d(input=agg, weight=weights, stride=1, padding=self.kernel_size // 2, groups=1)
        sig = agg.sigmoid()
        attn1 = attn1 * sig[:, 0, :, :].unsqueeze(1)
        attn2 = attn2 * sig[:, 1, :, :].unsqueeze(1)
        out = self.conv(attn1 + attn2)

        return out*gui


# MoCoLSK Module
class MoCoLSKWithoutPPMBlock_Avg(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoLSKConvWithoutPPM_Avg(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out


class MoCoLSKConvWithoutPPM_Max(nn.Module):
    """
    LST -> LSK Pathway & return attn * gui & (use avg, not PPM)
    """
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

        # max pooling, not PPM
        lst_pools = F.adaptive_max_pool2d(lst, (1, 1)).reshape(B, C)  # b,c
        gui_pools = F.adaptive_max_pool2d(gui, (1, 1)).reshape(B, C)  # b,c

        # Dynamic MLP
        weights = self.dynamic_mlp(lst_pools, gui_pools)  # B, C
        # in_chan=2, out_chan=2, kernel_size, kernel_size
        weights = weights.reshape(2, 2, self.kernel_size, self.kernel_size)  
        weights = nn.Parameter(data=weights, requires_grad=False)
        agg = F.conv2d(input=agg, weight=weights, stride=1, padding=self.kernel_size // 2, groups=1)
        sig = agg.sigmoid()
        attn1 = attn1 * sig[:, 0, :, :].unsqueeze(1)
        attn2 = attn2 * sig[:, 1, :, :].unsqueeze(1)
        out = self.conv(attn1 + attn2)

        return out*gui


# MoCoLSK Module
class MoCoLSKWithoutPPMBlock_Max(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoLSKConvWithoutPPM_Max(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out



class MoCoLSKConvWithoutPPM_AvgMax(nn.Module):
    """
    LST -> LSK Pathway & return attn * gui & (use avg, not PPM)
    """
    def __init__(self, dim, planes, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool_sizes = [1, 2, 3, 6]
        self.conv_lst = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial0 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.dynamic_mlp = FusionModule(
            inplanes=2*dim,
            planes=2*planes,
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

        # average and max pooling, not PPM
        lst_pools = torch.cat([F.adaptive_avg_pool2d(lst, (1, 1)).reshape(B, C),
                               F.adaptive_max_pool2d(lst, (1, 1)).reshape(B, C)], dim=-1)  # b,2c
        gui_pools = torch.cat([F.adaptive_avg_pool2d(gui, (1, 1)).reshape(B, C),
                               F.adaptive_max_pool2d(gui, (1, 1)).reshape(B, C)], dim=-1)  # b,2c

        # Dynamic MLP
        weights = self.dynamic_mlp(lst_pools, gui_pools)  # B, C
        # in_chan=2, out_chan=2, kernel_size, kernel_size
        weights = weights.reshape(2, 2, self.kernel_size, self.kernel_size)  
        weights = nn.Parameter(data=weights, requires_grad=False)
        agg = F.conv2d(input=agg, weight=weights, stride=1, padding=self.kernel_size // 2, groups=1)
        sig = agg.sigmoid()
        attn1 = attn1 * sig[:, 0, :, :].unsqueeze(1)
        attn2 = attn2 * sig[:, 1, :, :].unsqueeze(1)
        out = self.conv(attn1 + attn2)

        return out*gui


# MoCoLSK Module
class MoCoLSKWithoutPPMBlock_AvgMax(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoLSKConvWithoutPPM_AvgMax(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out


# Guidance -> LSK Pathway
class InvMoCoLSKConv(nn.Module):
    """
    gui -> LSK Pathway & return attn * lst
    """

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
        # todo self.conv_lst(lst) -> self.conv_lst(gui)
        attn1 = self.conv_lst(gui)
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

        return out * lst


# MoCoLSK Module: Guidance -> LSK Pathway
class InvMoCoLSKBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = InvMoCoLSKConv(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out


# Guidance -> LSK Pathway
class InvMoCoLSKConv_M(nn.Module):
    """
    gui -> LSK Pathway & return attn * gui
    """

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
        # todo self.conv_lst(lst) -> self.conv_lst(gui)
        attn1 = self.conv_lst(gui)
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

        return out * gui


# MoCoLSK Module: Guidance -> LSK Pathway
class InvMoCoLSK_MBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = InvMoCoLSKConv_M(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out


class MoCoLSKConvWithSmallKernel_k3d1p1_k3d2p2(nn.Module):
    """
    gui -> LSK Pathway & return attn * lst & (small kernel, same as SKNet, not large kernel)
    """

    def __init__(self, dim, planes, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool_sizes = [1, 2, 3, 6]
        # small kernel, same as SKNet.
        self.conv_lst = nn.Conv2d(dim, dim, 3, stride=1, padding=1, dilation=1, groups=dim)
        self.conv_spatial0 = nn.Conv2d(dim, dim, 3, stride=1, padding=2, dilation=2, groups=dim)

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

        return out*gui


class MoCoLSKWithSmallKernelBlock_k3d1p1_k3d2p2(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoLSKConvWithSmallKernel_k3d1p1_k3d2p2(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out


class MoCoLSKConvWithSmallKernel_k3d1p1_k5d1p2(nn.Module):
    """
    gui -> LSK Pathway & return attn * lst & (small kernel, same as SKNet, not large kernel)
    """

    def __init__(self, dim, planes, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool_sizes = [1, 2, 3, 6]
        # small kernel, same as SKNet.
        self.conv_lst = nn.Conv2d(dim, dim, 3, stride=1, padding=1, dilation=1, groups=dim)
        self.conv_spatial0 = nn.Conv2d(dim, dim, 5, stride=1, padding=2, dilation=1, groups=dim)

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

        return out*gui



class MoCoLSKWithSmallKernelBlock_k3d1p1_k5d1p2(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoLSKConvWithSmallKernel_k3d1p1_k5d1p2(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out



class MoCoLSKConvWithSmallKernel_k3d1p1_k5d2p4(nn.Module):
    """
    gui -> LSK Pathway & return attn * lst & (small kernel, same as SKNet, not large kernel)
    """

    def __init__(self, dim, planes, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool_sizes = [1, 2, 3, 6]
        # small kernel, same as SKNet.
        self.conv_lst = nn.Conv2d(dim, dim, 3, stride=1, padding=1, dilation=1, groups=dim)
        self.conv_spatial0 = nn.Conv2d(dim, dim, 5, stride=1, padding=4, dilation=2, groups=dim)

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

        return out*gui



class MoCoLSKWithSmallKernelBlock_k3d1p1_k5d2p4(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoLSKConvWithSmallKernel_k3d1p1_k5d2p4(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out


class MoCoLSKConvWithSmallKernel_k7d1p3_k9d4p16(nn.Module):
    """
    gui -> LSK Pathway & return attn * lst & (small kernel, same as SKNet, not large kernel)
    """

    def __init__(self, dim, planes, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool_sizes = [1, 2, 3, 6]
        # small kernel, same as SKNet.
        self.conv_lst = nn.Conv2d(dim, dim, 7, stride=1, padding=3, dilation=1, groups=dim)
        self.conv_spatial0 = nn.Conv2d(dim, dim, 9, stride=1, padding=16, dilation=4, groups=dim)

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

        return out*gui



class MoCoLSKWithSmallKernelBlock_k7d1p3_k9d4p16(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoLSKConvWithSmallKernel_k7d1p3_k9d4p16(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out


class MoCoLSKConvWithSmallKernel_k9d1p4_k11d5p25(nn.Module):
    """
    gui -> LSK Pathway & return attn * lst & (small kernel, same as SKNet, not large kernel)
    """

    def __init__(self, dim, planes, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool_sizes = [1, 2, 3, 6]
        # small kernel, same as SKNet.
        self.conv_lst = nn.Conv2d(dim, dim, 9, stride=1, padding=4, dilation=1, groups=dim)
        self.conv_spatial0 = nn.Conv2d(dim, dim, 11, stride=1, padding=25, dilation=5, groups=dim)

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

        return out*gui



class MoCoLSKWithSmallKernelBlock_k9d1p4_k11d5p25(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.dlsk = MoCoLSKConvWithSmallKernel_k9d1p4_k11d5p25(gui_dim, gui_dim, hidden=hidden, num_layers=num_layers, mlp_type=mlp_type, kernel_size=kernel_size)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        dlsk_feats = self.dlsk(lst_up, gui)
        feats = torch.cat([lst_up, dlsk_feats], dim=1)
        out = self.down(feats)

        return out


class LSKConv_M(nn.Module):

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


class LSKMBlock(nn.Module):
    """ LSKBlock with Multimodality
    """
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.lsk_m = LSKConv_M(2 * gui_dim)
        self.conv = nn.Conv2d(2*gui_dim, gui_dim, 1)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        feats = self.lsk_m(torch.cat([lst_up, gui], dim=1))

        feats = self.conv(feats)
        out = self.down(torch.cat([feats, lst_up], dim=1))

        return out


class LSKCSConv_M(nn.Module):
    """ LSK-CS: LSK with Channel Selection. 
        LSKCSConv_M: LSK-CS with Multimodality.
        same as SKNET: https://github.com/pppLang/SKNet/tree/master
    """
    def __init__(self, features, M=2, r=2, L=32):
        """Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            r: the radio for compute d, the length of z.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(LSKCSConv_M, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        dim = features
        self.lkconv1 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.lkconv2 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        lk_fea1 = self.lkconv1(x)
        lk_fea2 = self.lkconv2(lk_fea1)
        fea1 = self.conv1(lk_fea1).unsqueeze(dim=1)
        fea2 = self.conv2(lk_fea2).unsqueeze(dim=1)

        feas = torch.cat([fea1, fea2], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        attn = (feas * attention_vectors).sum(dim=1)

        out = attn * x

        return out

# todo
class LSKCSMBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.lsk_m = LSKCSConv_M(2 * gui_dim)
        self.conv = nn.Conv2d(2*gui_dim, gui_dim, 1)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        feats = self.lsk_m(torch.cat([lst_up, gui], dim=1))

        feats = self.conv(feats)
        out = self.down(torch.cat([feats, lst_up], dim=1))

        return out


class SKConv_M(nn.Module):
    """# SKNET: https://github.com/pppLang/SKNet/tree/master and 
    https://github.com/developer0hye/SKNet-PyTorch

    """
    def __init__(self, features, M=2, r=2, G=None, stride=1, L=32):
        """Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv_M, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        if G == None:
            G = features
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        features,
                        features,
                        kernel_size=3,
                        stride=stride,
                        padding=1 + i,
                        dilation=1 + i,
                        groups=G, # default: depth-wise conv
                    ),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False),
                )
            )
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)

        return fea_v


class SKMBlock(nn.Module):
    """ SKBlock with Multimodality

    """
    def __init__(self, lst_dim, gui_dim, scale, hidden=32, num_layers=1, mlp_type="a", kernel_size=3):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.sk_m = SKConv_M(features=2*gui_dim)
        self.conv = nn.Conv2d(2*gui_dim, gui_dim, 1)
        self.down = DenseProjection(2 * gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui):
        lst_up = self.lst_up(depth)
        # SK
        feats = self.sk_m(torch.cat((lst_up, gui), dim=1))

        feats = self.conv(feats)
        out = self.down(torch.cat([feats, lst_up], dim=1))

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
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1):
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
        res = res + x

        return res


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(False), res_scale=1):
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
        res = res + x
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
                act=nn.LeakyReLU(negative_slope=0.2, inplace=False),
                res_scale=1,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
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
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
