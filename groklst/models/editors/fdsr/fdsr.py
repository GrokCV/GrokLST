import math
import torch
import torch.nn as nn
from torch.nn.functional import pad
from mmengine.model import BaseModule
from mmagic.registry import MODELS

def tensor_pad(img, scale=32):
    h, w = img.size()[2:]

    pad_h = scale - h % scale
    pad_w = scale - w % scale

    padding = [pad_w, 0, pad_h, 0]
    img = pad(img, padding, "reflect")

    return img, pad_h, pad_w


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.is_dw = groups == in_channels
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."
        self.alpha_in, self.alpha_out = alpha_in, alpha_out
        self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
            nn.Conv2d(int(alpha_in * in_channels), int(alpha_out * out_channels),
                      kernel_size, 1, padding, dilation, math.ceil(alpha_in * groups), bias)
        self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 or self.is_dw else \
            nn.Conv2d(int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                      kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 or self.is_dw else \
            nn.Conv2d(in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels),
                      kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
            nn.Conv2d(in_channels - int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                      kernel_size, 1, padding, dilation, math.ceil(groups - alpha_in * groups), bias)

    def forward(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)

        x_h = self.downsample(x_h) if self.stride == 2 else x_h
        x_h2h = self.conv_h2h(x_h)
        x_h2l = self.conv_h2l(self.downsample(
            x_h)) if self.alpha_out > 0 and not self.is_dw else None
        if x_l is not None:
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None
            if self.is_dw:
                return x_h2h, x_l2l
            else:
                x_l2h = self.conv_l2h(x_l)
                x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
                x_h = x_l2h + x_h2h
                x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
                return x_h, x_l
        else:
            return x_h2h, x_h2l


class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(Conv_BN, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(
            int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(
            int(out_channels * alpha_out))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_l


class Conv_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(Conv_BN_ACT, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        # self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        # self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))
        # self.act = activation_layer(inplace=True)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act(x_h)
        x_l = self.act(x_l) if x_l is not None else None
        return x_h, x_l


class MS_RB(nn.Module):
    def __init__(self, num_feats, kernel_size):
        super(MS_RB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                               kernel_size=kernel_size, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                               kernel_size=kernel_size, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                               kernel_size=1, padding=0)
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

    assert (not input.size(2) % s and not input.size(3) % s)

    if input.size(1) == 3:
        # bgr2gray (same as opencv conversion matrix)
        input = (0.299 * input[:, 2] + 0.587 *
                 input[:, 1] + 0.114 * input[:, 0]).unsqueeze(1)

    out = torch.cat([input[:, :, i::s, j::s] for i in range(s)
                    for j in range(s)], dim=1)

    """
        out: torch.floatTensor (N, s**2, H/s, W/s)
    """
    return out

@MODELS.register_module()
class FDSR(BaseModule):
    """
    paper: https://arxiv.org/abs/2104.06174 or https://openaccess.thecvf.com/content/CVPR2021/html/He_Towards_Fast_and_Accurate_Real-World_Depth_Super-Resolution_Benchmark_Dataset_and_CVPR_2021_paper.html
    code: https://github.com/lingzhi96/RGB-D-D-Dataset/tree/main
    
    """
    def __init__(self, 
                 dep_in_channels=1, 
                 guidance_in_channels=3,
                 upscale_factor = 2, 
                 num_feats=32,
                 kernel_size=3,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super(FDSR, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        # todo
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)
        
        downscale_factor = 4
        self.dep_pixelunshuffle = nn.PixelUnshuffle(
            downscale_factor=downscale_factor)
        self.rgb_pixelunshuffle = nn.PixelUnshuffle(
            downscale_factor=downscale_factor)
        
        guidance_channels = guidance_in_channels * downscale_factor**2
        self.conv_rgb1 = nn.Conv2d(in_channels=guidance_channels, out_channels=num_feats,
                                   kernel_size=kernel_size, padding=1)
        # self.conv_rgb2 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        # self.conv_rgb3 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        # self.conv_rgb4 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)
        # self.conv_rgb5 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
        #                           kernel_size=kernel_size, padding=1)

        self.rgb_cbl2 = Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size,
                                    alpha_in=0, alpha_out=0.25,
                                    stride=1, padding=1, dilation=1, groups=1, bias=False, norm_layer=nn.BatchNorm2d,
                                    activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.rgb_cbl3 = Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size,
                                    alpha_in=0.25, alpha_out=0.25,
                                    stride=1, padding=1, dilation=1, groups=1, bias=False, norm_layer=nn.BatchNorm2d,
                                    activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.rgb_cbl4 = Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size,
                                    alpha_in=0.25, alpha_out=0.25,
                                    stride=1, padding=1, dilation=1, groups=1, bias=False, norm_layer=nn.BatchNorm2d,
                                    activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # self.rgb_cbl5 = oc.Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0.125, alpha_out=0.125,
        #                            stride=1, padding=1,dilation=1,groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(negative_slope=0.2, inplace=True))
        dep_channels = dep_in_channels * downscale_factor**2
        self.conv_dp1 = nn.Conv2d(in_channels=dep_channels, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)
        self.MSB1 = MS_RB(num_feats, kernel_size)
        self.MSB2 = MS_RB(56, kernel_size)
        self.MSB3 = MS_RB(80, kernel_size)
        self.MSB4 = MS_RB(104, kernel_size)

        self.conv_recon1 = nn.Conv2d(
            in_channels=104, out_channels=4 * num_feats, kernel_size=kernel_size, padding=1)
        self.ps1 = nn.PixelShuffle(2)
        self.conv_recon2 = nn.Conv2d(in_channels=num_feats, out_channels=4 * num_feats, kernel_size=kernel_size,
                                     padding=1)
        self.ps2 = nn.PixelShuffle(2)
        self.restore = nn.Conv2d(
            in_channels=num_feats, out_channels=dep_in_channels, kernel_size=kernel_size, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, hr_guidance):
        hr_guidance = torch.stack(hr_guidance, dim=0)
        hr_depth = self.img_upsampler(x)
        
        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])

        # hr_guidance, _, _ = tensor_pad(hr_guidance, 32)
        # depth, h, w = tensor_pad(depth, 32)
        h = w = 0
        # re_im = resample_data(hr_guidance, 4)
        # re_dp = resample_data(depth, 4)
        re_im = self.rgb_pixelunshuffle(hr_guidance)  # todo
        re_dp = self.dep_pixelunshuffle(hr_depth)  # todo

        dp_in = self.act(self.conv_dp1(re_dp))
        dp1 = self.MSB1(dp_in)

        rgb1 = self.act(self.conv_rgb1(re_im))
        # rgb2 = self.act(self.conv_rgb2(rgb1))

        rgb2 = self.rgb_cbl2(rgb1)
        # print(dp1.size(), rgb2[0].size())
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
        out = hr_depth + out

        out =  out[:, :, h:, w:]
        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out



if __name__ == "__main__":
    lr_dep = torch.randn((2, 1, 112, 112)).cuda()
    hr_guidance = torch.randn((2, 9, 224, 224)).cuda()
    net = FDSR(dep_in_channels=1, guidance_in_channels=9).cuda()
    out = net(lr_dep,hr_guidance )
    print(out.shape)