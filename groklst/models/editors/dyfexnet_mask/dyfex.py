import torch.nn as nn
import torch
from .dynamic_mlp import *
import torch.nn.functional as F
from .channel_attentions import SELayer, eca_layer


__all__ = [ 'ResidualGroup', 'CALayer', 
            'RCAB', 'ResBlock', 'DenseProjection', 
            'projection_conv', 'default_conv',
            # -------------------------------------------------------------
            'DynamicChannelExchangeWithCatBlock', 'DynamicChannelExchangeWithoutCatBlock',
            'DynamicCSExchangeWithCatBlock', 'DynamicCSExchangeWithoutCatBlock', 
            "DynamicChannelExchangeWithConvCatBlock",
            "DynamicChannelExchangeWithSECatBlock", 
            "DynamicChannelExchangeWithECACatBlock", "ChannelAttExchangeBlock"]


from einops import rearrange

INF = 1e8


class conv_bn_relu(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(conv_bn_relu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class ChannelAttExchange(nn.Module):
    """
    Channel exchange with dynamic mask prediction using attention.

    Args:
        p (float, optional): Fraction of the features to be exchanged. Defaults to 1/2.
        mlp_channels (int, optional): Number of channels for the MLP. Defaults to 256.
    """

    def __init__(self, p=1 / 2, mlp_channels=256):
        super().__init__()
        assert 0 <= p <= 1, "p should be between 0 and 1"
        self.p = p
        self.num_exchange_channels = int(mlp_channels * self.p)
        self.mlp = MLP(
            in_channels=self.num_exchange_channels,
            hidden_channels=64,
            out_channels=self.num_exchange_channels,
        )
        self.attention_mask = LSKLayer(mlp_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        N, C, H, W = x1.shape

        x1_mask_logits = self.attention_mask(x1)  # B, C
        x1_mask_logits = self.sigmoid(x1_mask_logits)  # B, C
        x2_mask_logits = self.attention_mask(x2)  # B, C
        x2_mask_logits = self.sigmoid(x2_mask_logits)  # B, C

        x1_topk_values, x1_topk_indices = x1_mask_logits.topk(
            self.num_exchange_channels, dim=1, largest=True, sorted=False
        )
        x2_topk_values, x2_topk_indices = x2_mask_logits.topk(
            self.num_exchange_channels, dim=1, largest=True, sorted=False
        )

        x1_topk_indices_sorted = torch.sort(x1_topk_indices, dim=1).values
        x2_topk_indices_sorted = torch.sort(x2_topk_indices, dim=1).values

        x1_extracted = x1[
            torch.arange(N).unsqueeze(-1), x1_topk_indices_sorted
        ]  # B, k, H, W
        x2_extracted = x2[
            torch.arange(N).unsqueeze(-1), x2_topk_indices_sorted
        ]  # B, k, H, W

        x1_extracted = rearrange(x1_extracted, "b c h w -> b (h w) c")
        x2_extracted = rearrange(x2_extracted, "b c h w -> b (h w) c")

        x1_extracted = self.mlp(x1_extracted)
        x2_extracted = self.mlp(x2_extracted)

        x1_extracted = rearrange(x1_extracted, "b (h w) c -> b c h w", h=H, w=W)
        x2_extracted = rearrange(x2_extracted, "b (h w) c -> b c h w", h=H, w=W)

        # out_x1 = x2_extracted
        # out_x2 = x1_extracted

        out_x1 = x1.clone()
        out_x2 = x2.clone()

        for i in range(N):
            out_x1[i, x1_topk_indices_sorted[i]] = x2_extracted[i]
            out_x2[i, x2_topk_indices_sorted[i]] = x1_extracted[i]

        return out_x1, out_x2


class ChannelAttExchangeBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, mask_indim=None):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = ChannelAttExchange(p=1/2, mlp_channels=gui_dim)
        self.down = DenseProjection(2*gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, mask=None):
        lst_up = self.lst_up(depth)
        lst_feats, gui_feats = self.cs_changer(lst_up, gui)
        out = self.down(torch.cat([lst_feats, gui_feats], dim=1))

        return out, mask
    

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class LSKLayer(nn.Module):
    def __init__(self, feat_channels):
        super(LSKLayer, self).__init__()
        self.conv0 = nn.Conv2d(
            feat_channels, feat_channels, 5, padding=2, groups=feat_channels
        )
        self.conv_spatial = nn.Conv2d(
            feat_channels,
            feat_channels,
            7,
            stride=1,
            padding=9,
            groups=feat_channels,
            dilation=3,
        )
        self.conv1 = nn.Conv2d(feat_channels, feat_channels // 2, 1)
        self.conv2 = nn.Conv2d(feat_channels, feat_channels // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(feat_channels // 2, feat_channels, 1)

    def forward(self, x):
        attn1 = self.conv0(x)  # B, C, H, W
        attn2 = self.conv_spatial(attn1)  # B, C, H, W
        attn1 = self.conv1(attn1)  # B, C//2, H, W
        attn2 = self.conv2(attn2)  # B, C//2, H, W

        attn = torch.cat([attn1, attn2], dim=1)  # B, C, H, W
        avg_attn = torch.mean(attn, dim=1, keepdim=True)  # B, 1, H, W
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)  # B, 1, H, W
        agg = torch.cat([avg_attn, max_attn], dim=1)  # B, 2, H, W
        sig = self.conv_squeeze(agg).sigmoid()  # B, 2, H, W
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(
            1
        )  # B, C//2, H, W
        attn = self.conv(attn)  # B, C, H, W

        y = x * attn  # B, C, H, W
        y = y.mean(dim=[2, 3])  # B, C
        return y


class DynamicChannelExchange(nn.Module):
    def __init__(self, dim, mask_indim, p=2, reduction=4, dynamic=True) -> None:
        super().__init__()
        self.p = p
        self.dynamic = dynamic
        self.mask_encoder = FCNet(mask_indim, dim, num_filts=dim)

    def forward(self, lst, gui, mask=None):
        N, C, H, W = lst.shape

        # dynamic channel exchange
        if mask is not None:
            mask = self.mask_encoder(mask).sigmoid()
            if self.dynamic:
                # 计算每个样本的中位数索引
                B, C = mask.shape
                k = C // 2
                # 获取每个样本的第 k 大值
                kth_values, _ = torch.kthvalue(mask, k, dim=1)
                # 扩展 kth_values 以便与 channel_mask 进行比较
                kth_values = kth_values.unsqueeze(1).expand(-1, C)
                # 使用 torch.where 将大于等于 kth_value 的值设为 True，其余为 False
                channel_mask = torch.where(mask > kth_values, torch.tensor(True), torch.tensor(False))
            else: # use threshod=0.5
                channel_mask = mask > 0.5
        else:
            channel_mask = torch.arange(C) % self.p == 0
            channel_mask = channel_mask.unsqueeze(0).expand((N, -1))

        out_lst = torch.zeros_like(lst)
        out_gui = torch.zeros_like(gui)
        out_lst[~channel_mask, ...] = lst[~channel_mask, ...]
        out_gui[~channel_mask, ...] = gui[~channel_mask, ...]
        out_lst[channel_mask, ...] = gui[channel_mask, ...]
        out_gui[channel_mask, ...] = lst[channel_mask, ...]

        return out_lst, out_gui, mask


class DynamicChannelExchangeWithCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, mask_indim=None):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = DynamicChannelExchange(gui_dim, mask_indim)
        self.down = DenseProjection(2*gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, mask=None):
        lst_up = self.lst_up(depth)
        lst_feats, gui_feats, mask = self.cs_changer(lst_up, gui, mask)
        out = self.down(torch.cat([lst_feats, gui_feats], dim=1))

        return out, mask


class DynamicChannelExchangeWithoutCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, mask_indim=None):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = DynamicChannelExchange(gui_dim, mask_indim)
        self.down = DenseProjection(gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, mask=None):
        lst_up = self.lst_up(depth)
        lst_feats, _, mask = self.cs_changer(lst_up, gui, mask)
        out = self.down(lst_feats)

        return out, mask


class DynamicCSExchange(nn.Module):
    def __init__(self, dim, mask_indim, p=2, reduction=4, dynamic=True) -> None:
        super().__init__()
        self.p = p
        self.dynamic = dynamic
        self.mask_encoder = FCNet(mask_indim, dim, num_filts=dim)
        self.fc = nn.Linear(dim, 512)

    def forward(self, lst, gui, mask=None):
        N, C, H, W = lst.shape

        # dynamic channel exchange
        if mask is not None:
            mask = self.mask_encoder(mask).sigmoid()
            # spatial mask
            spatial_mask = self.fc(mask).sigmoid()
            if self.dynamic:
                # channel
                B, C = mask.shape
                k = C // 2
                kth_values, _ = torch.kthvalue(mask, k, dim=1)
                kth_values = kth_values.unsqueeze(1).expand(-1, C)
                channel_mask = torch.where(mask > kth_values, torch.tensor(True), torch.tensor(False))
                # spatial
                B, _C = spatial_mask.shape
                _k = _C // 2
                _kth_values, _ = torch.kthvalue(mask, _k, dim=1)
                _kth_values = _kth_values.unsqueeze(1).expand(-1, _C)
                channel_mask = torch.where(spatial_mask > _kth_values, torch.tensor(True), torch.tensor(False))
            else: # use threshod=0.5
                channel_mask = mask > 0.5
        else:
            channel_mask = torch.arange(C) % self.p == 0
            channel_mask = channel_mask.unsqueeze(0).expand((N, -1))

        out_lst = torch.zeros_like(lst)
        out_gui = torch.zeros_like(gui)
        # channel
        out_lst[~channel_mask, ...] = lst[~channel_mask, ...]
        out_gui[~channel_mask, ...] = gui[~channel_mask, ...]
        out_lst[channel_mask, ...] = gui[channel_mask, ...]
        out_gui[channel_mask, ...] = lst[channel_mask, ...]

        # spatial
        out_lst[~spatial_mask, ...] = lst[~spatial_mask, ...]
        out_gui[~spatial_mask, ...] = gui[~spatial_mask, ...]
        out_lst[spatial_mask, ...] = gui[spatial_mask, ...]
        out_gui[spatial_mask, ...] = lst[spatial_mask, ...]

        return out_lst, out_gui, mask


class DynamicCSExchangeWithCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, mask_indim=None):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = DynamicCSExchange(gui_dim, mask_indim)
        self.down = DenseProjection(2*gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, mask=None):
        lst_up = self.lst_up(depth)
        lst_feats, gui_feats, mask = self.cs_changer(lst_up, gui, mask)
        out = self.down(torch.cat([lst_feats, gui_feats], dim=1))

        return out, mask



class DynamicCSExchangeWithoutCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, mask_indim=None):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = DynamicCSExchange(gui_dim, mask_indim)
        self.down = DenseProjection(gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, mask=None):
        lst_up = self.lst_up(depth)
        lst_feats, _, mask = self.cs_changer(lst_up, gui, mask)
        out = self.down(lst_feats)

        return out, mask


class DynamicChannelExchangeWithoutCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, mask_indim=None):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = DynamicChannelExchange(gui_dim, mask_indim)
        self.down = DenseProjection(gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, mask=None):
        lst_up = self.lst_up(depth)
        lst_feats, _, mask = self.cs_changer(lst_up, gui, mask)
        out = self.down(lst_feats)

        return out, mask


class DynamicChannelExchangeWithConv(nn.Module):
    def __init__(self, dim, mask_indim, p=2, reduction=4, dynamic=True) -> None:
        super().__init__()
        self.p = p
        self.dynamic = dynamic
        self.mask_encoder = FCNet(mask_indim, dim, num_filts=dim)
        self.conv1 = nn.Conv2d(dim//2 , dim//2, 1, 1)
        self.conv2 = nn.Conv2d(dim//2 , dim//2, 1, 1)

    def forward(self, lst, gui, mask=None):
        N, C, H, W = lst.shape

        # dynamic channel exchange
        if mask is not None:
            mask = self.mask_encoder(mask).sigmoid()
            if self.dynamic:
                # 计算每个样本的中位数索引
                B, C = mask.shape
                k = C // 2
                # 获取每个样本的第 k 大值
                kth_values, _ = torch.kthvalue(mask, k, dim=1)
                # 扩展 kth_values 以便与 mask 进行比较
                kth_values = kth_values.unsqueeze(1).expand(-1, C)
                # 使用 torch.where 将大于等于 kth_value 的值设为 True，其余为 False
                channel_mask = torch.where(mask > kth_values, torch.tensor(True), torch.tensor(False))
            else: # use threshod=0.5
                channel_mask = mask > 0.5
        else:
            channel_mask = torch.arange(C) % self.p == 0
            channel_mask = channel_mask.unsqueeze(0).expand((N, -1))

        out_lst = torch.zeros_like(lst)
        out_gui = torch.zeros_like(gui)
        out_lst[~channel_mask, ...] = lst[~channel_mask, ...]
        out_gui[~channel_mask, ...] = gui[~channel_mask, ...]
        out_lst[channel_mask, ...] = self.conv2(gui[channel_mask, ...])
        out_gui[channel_mask, ...] = self.conv1(lst[channel_mask, ...])

        return out_lst, out_gui, mask


class DynamicChannelExchangeWithConvCatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, mask_indim=None):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = DynamicChannelExchangeWithConv(gui_dim, mask_indim)
        self.down = DenseProjection(2*gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, mask=None):
        lst_up = self.lst_up(depth)
        lst_feats, gui_feats, mask = self.cs_changer(lst_up, gui, mask)
        out = self.down(torch.cat([lst_feats, gui_feats], dim=1))

        return out, mask


class DynamicChannelExchangeWithSE(nn.Module):
    def __init__(self, dim, mask_indim, p=2, reduction=4, dynamic=True) -> None:
        super().__init__()
        self.p = p
        self.dynamic = dynamic
        self.mask_encoder = FCNet(mask_indim, dim, num_filts=dim)
        self.conv1 = nn.Conv2d(dim//2 , dim//2, 1, 1)
        self.conv2 = nn.Conv2d(dim//2 , dim//2, 1, 1)
        self.channel_attention = SELayer(channel=2*dim, reduction=16)

    def forward(self, lst, gui, mask=None):
        N, C, H, W = lst.shape

        # 1. use mask
        # dynamic channel exchange
        mask1 = self.mask_encoder(mask).sigmoid()

        # 2. use lst and gui
        mask2 = self.channel_attention(torch.cat([lst, gui], dim=1))
        
        # 3. calculate mask
        # 计算每个样本的中位数索引
        mask = mask1 * mask2
        B, C = mask.shape
        k = C // 2
        # # 获取每个样本的第 k 大值 (从小到大)
        # kth_values, _ = torch.kthvalue(mask, k, dim=1)
        # # 扩展 kth_values 以便与 mask 进行比较
        # kth_values = kth_values.unsqueeze(1).expand(-1, C)
        # # 使用 torch.where 将大于等于 kth_value 的值设为 True，其余为 False
        # channel_mask = torch.where(mask > kth_values, torch.tensor(True), torch.tensor(False))

        # 对 mask 进行 topk 操作，获取前 k 大的值和对应的索引
        topk_values, topk_indices = torch.topk(mask, k, dim=1, largest=True, sorted=False)

        # 创建 channel_mask，并将前 k 个值的位置设为 True
        channel_mask = torch.zeros_like(mask, dtype=torch.bool)
        channel_mask.scatter_(1, topk_indices, True)

        out_lst = torch.zeros_like(lst)
        out_gui = torch.zeros_like(gui)
        out_lst[~channel_mask, ...] = lst[~channel_mask, ...]
        out_gui[~channel_mask, ...] = gui[~channel_mask, ...]
        out_lst[channel_mask, ...] = self.conv2(gui[channel_mask, ...])
        out_gui[channel_mask, ...] = self.conv1(lst[channel_mask, ...])

        return out_lst, out_gui, mask


class DynamicChannelExchangeWithSECatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, mask_indim=None):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = DynamicChannelExchangeWithSE(gui_dim, mask_indim)
        self.down = DenseProjection(2*gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, mask=None):
        lst_up = self.lst_up(depth)
        lst_feats, gui_feats, mask = self.cs_changer(lst_up, gui, mask)
        out = self.down(torch.cat([lst_feats, gui_feats], dim=1))

        return out, mask




class DynamicChannelExchangeWithECA(nn.Module):
    def __init__(self, dim, mask_indim, p=2, reduction=4, dynamic=True) -> None:
        super().__init__()
        self.p = p
        self.dynamic = dynamic
        self.mask_encoder = FCNet(mask_indim, dim, num_filts=dim)
        self.conv1 = nn.Conv2d(dim//2 , dim//2, 1, 1)
        self.conv2 = nn.Conv2d(dim//2 , dim//2, 1, 1)
        self.channel_attention = eca_layer(channel=2*dim)

    def forward(self, lst, gui, mask=None):
        N, C, H, W = lst.shape

        # 1. use mask
        # dynamic channel exchange
        mask1 = self.mask_encoder(mask).sigmoid()

        # 2. use lst and gui
        mask2 = self.channel_attention(torch.cat([lst, gui], dim=1))
        
        # 3. calculate mask
        mask = mask1 * mask2
        B, C = mask.shape
        k = C // 2
        # 对 mask 进行 topk 操作，获取前 k 大的值和对应的索引
        topk_values, topk_indices = torch.topk(mask, k, dim=1, largest=True, sorted=False)

        # 创建 channel_mask，并将前 k 个值的位置设为 True
        channel_mask = torch.zeros_like(mask, dtype=torch.bool)
        channel_mask.scatter_(1, topk_indices, True)


        out_lst = torch.zeros_like(lst)
        out_gui = torch.zeros_like(gui)
        out_lst[~channel_mask, ...] = lst[~channel_mask, ...]
        out_gui[~channel_mask, ...] = gui[~channel_mask, ...]
        out_lst[channel_mask, ...] = self.conv2(gui[channel_mask, ...])
        out_gui[channel_mask, ...] = self.conv1(lst[channel_mask, ...])

        return out_lst, out_gui, mask


class DynamicChannelExchangeWithECACatBlock(nn.Module):
    def __init__(self, lst_dim, gui_dim, scale, mask_indim=None):
        super().__init__()
        self.lst_up = DenseProjection(lst_dim, gui_dim, scale, up=True, bottleneck=False)
        self.cs_changer = DynamicChannelExchangeWithECA(gui_dim, mask_indim)
        self.down = DenseProjection(2*gui_dim, lst_dim + gui_dim, scale, up=False, bottleneck=False)

    def forward(self, depth, gui, mask=None):
        lst_up = self.lst_up(depth)
        lst_feats, gui_feats, mask = self.cs_changer(lst_up, gui, mask)
        out = self.down(torch.cat([lst_feats, gui_feats], dim=1))

        return out, mask





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
