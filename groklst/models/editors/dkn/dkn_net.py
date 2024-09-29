import torch
import torch.nn.functional as F
import torch.nn as nn

from mmengine.model import BaseModule
from mmagic.registry import MODELS

def grid_generator(k, r, n):
    """grid_generator
    Parameters
    ---------
    f : filter_size, int
    k: kernel_size, int
    n: number of grid, int
    Returns
    -------
    torch.Tensor. shape = (n, 2, k, k)
    """
    grid_x, grid_y = torch.meshgrid(
        [torch.linspace(k // 2, k // 2 + r - 1, steps=r), torch.linspace(k // 2, k // 2 + r - 1, steps=r)]
    )
    grid = torch.stack([grid_x, grid_y], 2).view(r, r, 2)

    return grid.unsqueeze(0).repeat(n, 1, 1, 1).cuda()


class Kernel_DKN(nn.Module):
    def __init__(self, input_channel, kernel_size):
        super(Kernel_DKN, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, 7)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 2, stride=(2, 2))
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 2, stride=(2, 2))
        self.conv5 = nn.Conv2d(64, 128, 5)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.conv7 = nn.Conv2d(128, 128, 3)

        self.conv_weight = nn.Conv2d(128, kernel_size**2, 1)
        self.conv_offset = nn.Conv2d(128, 2 * kernel_size**2, 1)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        offset = self.conv_offset(x)
        weight = torch.sigmoid(self.conv_weight(x))

        return weight, offset


@MODELS.register_module()
class DKN(BaseModule):
    """
    title: Deformable Kernel Networks for Joint Image Filtering
    paper: https://arxiv.org/pdf/1910.08373.pdf
    code: https://github.com/cvlab-yonsei/dkn
    
    
    """
    def __init__(self, 
                 lst_channels=1, 
                 gui_channels=10, 
                 scale=4, 
                 kernel_size=3, 
                 filter_size=15, 
                 residual=True,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super(DKN, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.img_upsampler = nn.Upsample(scale_factor=scale, mode="bicubic", align_corners=False)
        self.ImageKernel = Kernel_DKN(input_channel=gui_channels, kernel_size=kernel_size)
        self.DepthKernel = Kernel_DKN(input_channel=lst_channels, kernel_size=kernel_size)
        self.residual = residual
        self.kernel_size = kernel_size
        self.filter_size = filter_size

    def forward(self, x, hr_guidance, **kwargs):
        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])

        x = self.img_upsampler(x)
        weight, offset = self._shift_and_stitch(hr_guidance, x)

        h, w = hr_guidance.size(2), hr_guidance.size(3)
        b = hr_guidance.size(0)
        k = self.filter_size
        r = self.kernel_size
        hw = h * w

        # weighted average
        # (b, 2*r**2, h, w) -> (b*hw, r, r, 2)
        offset = offset.permute(0, 2, 3, 1).contiguous().view(b * hw, r, r, 2)
        # (b, r**2, h, w) -> (b*hw, r**2, 1)
        weight = weight.permute(0, 2, 3, 1).contiguous().view(b * hw, r * r, 1)

        # (b*hw, r, r, 2)
        grid = grid_generator(k, r, b * hw)

        coord = grid + offset
        coord = (coord / k * 2) - 1

        # (b, k**2, hw) -> (b*hw, 1, k, k)
        depth_col = F.unfold(x, k, padding=k // 2).permute(0, 2, 1).contiguous().view(b * hw, 1, k, k)

        # (b*hw, 1, k, k), (b*hw, r, r, 2) => (b*hw, 1, r^2)
        depth_sampled = F.grid_sample(depth_col, coord).view(b * hw, 1, -1)

        # (b*w*h, 1, r^2) x (b*w*h, r^2, 1) => (b, 1, h,w)
        out = torch.bmm(depth_sampled, weight).view(b, 1, h, w)

        if self.residual:
            out = out + x
        
        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out

    def _infer(self, hr_guidance, x):
        imkernel, imoffset = self.ImageKernel(hr_guidance)
        depthkernel, depthoffset = self.DepthKernel(x)

        weight = imkernel * depthkernel
        offset = imoffset * depthoffset

        if self.residual:
            weight = weight - torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight = weight / torch.sum(weight, 1).unsqueeze(1).expand_as(weight)

        return weight, offset

    def _shift_and_stitch(self, hr_guidance, x):

        offset = torch.zeros(
            (hr_guidance.size(0), 2 * self.kernel_size**2, hr_guidance.size(2), hr_guidance.size(3)),
            dtype=hr_guidance.dtype,
            layout=hr_guidance.layout,
            device=hr_guidance.device,
        )
        weight = torch.zeros(
            (hr_guidance.size(0), self.kernel_size**2, hr_guidance.size(2), hr_guidance.size(3)),
            dtype=hr_guidance.dtype,
            layout=hr_guidance.layout,
            device=hr_guidance.device,
        )

        for i in range(4):
            for j in range(4):
                m = nn.ZeroPad2d((25 - j, 22 + j, 25 - i, 22 + i))
                img_shift = m(hr_guidance)
                depth_shift = m(x)
                w, o = self._infer(img_shift, depth_shift)

                weight[:, :, i::4, j::4] = w
                offset[:, :, i::4, j::4] = o

        return weight, offset


class Kernel_FDKN(nn.Module):
    def __init__(self, input_channel, kernel_size, factor=4):
        super(Kernel_FDKN, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv_weight = nn.Conv2d(128, kernel_size**2 * (factor) ** 2, 1)
        self.conv_offset = nn.Conv2d(128, 2 * kernel_size**2 * (factor) ** 2, 1)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6(x))

        offset = self.conv_offset(x)
        weight = torch.sigmoid(self.conv_weight(x))

        return weight, offset


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
class FDKN(nn.Module):

    """
    title: Deformable Kernel Networks for Joint Image Filtering
    paper: https://arxiv.org/pdf/1910.08373.pdf
    code: https://github.com/cvlab-yonsei/dkn
    
    
    """
    def __init__(self, 
                 lst_channels=1, 
                 gui_channels=10, 
                 scale=4, 
                 kernel_size=3, 
                 filter_size=15, 
                 residual=True,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super(FDKN, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.factor = scale  # resample facto
        self.img_upsampler = nn.Upsample(scale_factor=scale, mode="bicubic", align_corners=False)
        self.ImageKernel = Kernel_FDKN(
            input_channel=gui_channels * scale * scale, kernel_size=kernel_size, factor=self.factor
        )
        self.DepthKernel = Kernel_FDKN(
            input_channel=lst_channels * scale * scale, kernel_size=kernel_size, factor=self.factor
        )
        self.residual = residual
        self.kernel_size = kernel_size
        self.filter_size = filter_size

    def forward(self, x, hr_guidance, **kwargs):

        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])

        x = self.img_upsampler(x)
        re_im = resample_data(hr_guidance, self.factor)
        re_dp = resample_data(x, self.factor)

        imkernel, imoffset = self.ImageKernel(re_im)
        depthkernel, depthoffset = self.DepthKernel(re_dp)

        weight = imkernel * depthkernel
        offset = imoffset * depthoffset

        ps = nn.PixelShuffle(self.factor)
        weight = ps(weight)
        offset = ps(offset)

        if self.residual:
            weight = weight - torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight = weight / torch.sum(weight, 1).unsqueeze(1).expand_as(weight)

        b, h, w = hr_guidance.size(0), hr_guidance.size(2), hr_guidance.size(3)
        k = self.filter_size
        r = self.kernel_size
        hw = h * w

        # weighted average
        # (b, 2*r**2, h, w) -> (b*hw, r, r, 2)
        offset = offset.permute(0, 2, 3, 1).contiguous().view(b * hw, r, r, 2)
        # (b, r**2, h, w) -> (b*hw, r**2, 1)
        weight = weight.permute(0, 2, 3, 1).contiguous().view(b * hw, r * r, 1)

        # (b*hw, r, r, 2)
        grid = grid_generator(k, r, b * hw)
        coord = grid + offset
        coord = (coord / k * 2) - 1

        # (b, k**2, hw) -> (b*hw, 1, k, k)
        depth_col = F.unfold(x, k, padding=k // 2).permute(0, 2, 1).contiguous().view(b * hw, 1, k, k)

        # (b*hw, 1, k, k), (b*hw, r, r, 2) => (b*hw, 1, r^2)
        depth_sampled = F.grid_sample(depth_col, coord).view(b * hw, 1, -1)

        # (b*w*h, 1, r^2) x (b*w*h, r^2, 1) => (b, 1, h, w)
        out = torch.bmm(depth_sampled, weight).view(b, 1, h, w)

        if self.residual:
            out = out + x

        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out


if __name__ == "__main__":
    lst = torch.randn((1, 1, 28, 28)).cuda()
    hr_guidance = torch.randn((1, 10, 224, 224)).cuda()
    net = DKN(scale=8).cuda()
    # net = FDKN(scale=8).cuda()
    out = net(lst, hr_guidance)
    print(out.shape)
