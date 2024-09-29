import math
import torch
import torch.nn as nn

import torch.nn.functional as F

from mmengine.model import BaseModule
from mmagic.registry import MODELS


@MODELS.register_module()
class CTNET(BaseModule):
    """
    title: Contextual Transformation Network for Lightweight Remote-Sensing Image Super-Resolution

    paper: https://ieeexplore.ieee.org/document/9632567/
    
    code: https://github.com/BITszwang/CTNet
    
    """
    def __init__(self, 
                 in_nc=1, 
                 out_nc=1, 
                 nf=64, 
                 unf=24, 
                 nb=16,
                 scale=2,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super(CTNET, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict

        self.scale = scale
        ### first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        self.ctb1 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa1 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.cem2 = CEM(nf)
        self.ctb2 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa2 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.cem3 = CEM(nf)
        self.ctb3 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa3 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.cem4 = CEM(nf)
        self.ctb4 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa4 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.cem5 = CEM(nf)
        self.ctb5 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa5 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.cem6 = CEM(nf)
        self.ctb6 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa6 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.cem7 = CEM(nf)
        self.ctb7 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa7 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.cem8 = CEM(nf)
        self.ctb8 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa8 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.cem9 = CEM(nf)
        self.ctb9 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa9 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.cem10 = CEM(nf)
        self.ctb10 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa10 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.cem11 = CEM(nf)
        self.ctb11 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa11 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.cem12 = CEM(nf)
        self.ctb12 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa12 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.cem13 = CEM(nf)
        self.ctb13 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa13 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.cem14 = CEM(nf)
        self.ctb14 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa14 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.cem15 = CEM(nf)
        self.ctb15 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa15 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        self.cem16 = CEM(nf)
        self.ctb16 = CTB(in_chs=nf, mid_chs=nf // 2, out_chs=nf)
        self.cfa16 = nn.Conv2d(nf, nf // nb, 3, 1, 3 // 2, groups=nf // nb, bias=False)

        #### upsampling
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.c = nn.Conv2d(nf, nf, 1, 1, 0, bias=True)

    def forward(self, x, **kwargs):

        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])


        fea = self.conv_first(x)

        out_b1 = self.ctb1(fea)
        skip1 = self.cfa1(out_b1)

        out_b1 = self.cem2(out_b1)
        out_b2 = self.ctb2(out_b1)
        skip2 = self.cfa2(out_b2)

        out_b2 = self.cem3(out_b2)
        out_b3 = self.ctb3(out_b2)
        skip3 = self.cfa3(out_b3)

        out_b3 = self.cem4(out_b3)
        out_b4 = self.ctb4(out_b3)
        skip4 = self.cfa4(out_b4)

        out_b4 = self.cem5(out_b4)
        out_b5 = self.ctb5(out_b4)
        skip5 = self.cfa5(out_b5)

        out_b5 = self.cem6(out_b5)
        out_b6 = self.ctb6(out_b5)
        skip6 = self.cfa6(out_b6)

        out_b6 = self.cem7(out_b6)
        out_b7 = self.ctb7(out_b6)
        skip7 = self.cfa7(out_b7)

        out_b7 = self.cem8(out_b7)
        out_b8 = self.ctb8(out_b7)
        skip8 = self.cfa8(out_b8)

        out_b8 = self.cem9(out_b8)
        out_b9 = self.ctb9(out_b8)
        skip9 = self.cfa9(out_b9)

        out_b9 = self.cem10(out_b9)
        out_b10 = self.ctb10(out_b9)
        skip10 = self.cfa10(out_b10)

        out_b10 = self.cem11(out_b10)
        out_b11 = self.ctb11(out_b10)
        skip11 = self.cfa11(out_b11)

        out_b11 = self.cem12(out_b11)
        out_b12 = self.ctb12(out_b11)
        skip12 = self.cfa12(out_b12)

        out_b12 = self.cem13(out_b12)
        out_b13 = self.ctb13(out_b12)
        skip13 = self.cfa13(out_b13)

        out_b13 = self.cem14(out_b13)
        out_b14 = self.ctb14(out_b13)
        skip14 = self.cfa14(out_b14)

        out_b14 = self.cem15(out_b14)
        out_b15 = self.ctb15(out_b14)
        skip15 = self.cfa15(out_b15)

        out_b15 = self.cem16(out_b15)
        out_b16 = self.ctb16(out_b15)
        skip16 = self.cfa16(out_b16)

        trunk = self.c(
            torch.cat(
                (
                    skip1,
                    skip2,
                    skip3,
                    skip4,
                    skip5,
                    skip6,
                    skip7,
                    skip8,
                    skip9,
                    skip10,
                    skip11,
                    skip12,
                    skip13,
                    skip14,
                    skip15,
                    skip16,
                ),
                dim=1,
            )
        )

        fea = fea + trunk

        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode="nearest"))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest"))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))

        out = self.conv_last(fea)

        ILR = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        out = out + ILR

        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out
    
    
class CEM(nn.Module):
    def __init__(self, n_feats):
        super(CEM, self).__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class CFE(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r):
        super(CFE, self).__init__()

        self.k1 = nn.Sequential(
            nn.Conv2d(
                inplanes, planes, kernel_size=1, stride=1, padding=0, dilation=dilation, groups=groups, bias=False
            )
        )

        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=2),
            nn.Conv2d(
                inplanes, planes, kernel_size=1, stride=1, padding=0, dilation=dilation, groups=groups, bias=False
            ),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(
                inplanes, planes, kernel_size=1, stride=1, padding=0, dilation=dilation, groups=groups, bias=False
            )
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(
                planes, planes, kernel_size=1, stride=stride, padding=0, dilation=dilation, groups=groups, bias=False
            )
        )

    def forward(self, x):
        identity = self.k1(x)

        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:]))
        )  # sigmoid(identity + k2)
        out = torch.mul(identity, out)

        return out


class PA(nn.Module):
    """PA is pixel attention"""

    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


def mean_channels(F):
    assert F.dim() == 4
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert F.dim() == 4
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(channel, channel, 1)

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        # y = self.conv(y)
        y = self.sigmoid(y)
        return x * y


class DElayer(nn.Module):

    def __init__(self, nf):

        super(DElayer, self).__init__()
        self.k1 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.cca = CCALayer(nf)

    def forward(self, x):

        y = self.cca(x)
        y = self.k1(y)
        out = x + y

        return out


####


class SqueezeExcite(nn.Module):
    def __init__(
        self, in_chs, se_ratio=0.25, reduced_base_chs=None, act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class CTL(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(CTL, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.cfe = nn.Sequential(
            CFE(inplanes=inp, planes=init_channels, stride=stride, padding=1, dilation=1, groups=1, pooling_r=2),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cft = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.feat_enhanced = nn.Sequential(
            nn.Conv2d(inp, init_channels + new_channels, 1, stride, 0, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.cfe(x)
        x2 = self.cft(x1)
        x_enhanced = self.feat_enhanced(x)
        out = torch.cat([x1, x2], dim=1)
        out = out + x_enhanced

        return out[:, : self.oup, :, :]


class CTB(nn.Module):
    """Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, se_ratio=0.0):
        super(CTB, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        self.ctl1 = CTL(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ctl2 = CTL(mid_chs, out_chs, relu=False)

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                # nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                # nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x = self.ctl1(x)

        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        if self.se is not None:
            x = self.se(x)

        x = self.ctl2(x)
        x += self.shortcut(residual)
        return x




if __name__ == "__main__":
    x = torch.randn((1, 1, 64, 64)).cuda()
    net = CTNET(scale=2).cuda()
    out = net(x)
    print(out.shape)
