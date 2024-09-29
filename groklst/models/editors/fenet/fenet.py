import torch
import torch.nn as nn

import torch.nn.functional as F
from mmagic.registry import MODELS
from mmengine.model import BaseModule



@MODELS.register_module()
class FeNet(BaseModule):
    """
    title: FeNet: Feature Enhancement Network for Lightweight Remote-Sensing Image Super-Resolution 
    paper: https://ieeexplore.ieee.org/document/9759417
    
    code: https://github.com/wangzheyuan-666/FeNet
    """
    def __init__(self, 
                 in_channels=1, 
                 upscale_factor=2, 
                 num_fea=48, 
                 out_channels=1, 
                 num_LBs=4,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super(FeNet, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.num_LBs = num_LBs
        self.upscale_factor = upscale_factor
        # feature extraction
        self.fea_conv = nn.Sequential(nn.Conv2d(in_channels, num_fea, 3, 1, 1), nn.Conv2d(num_fea, num_fea, 3, 1, 1))

        # LBlocks
        LBs = []
        for i in range(num_LBs):
            LBs.append(FEB(num_fea))
        self.LBs = nn.ModuleList(LBs)

        # BFModule
        self.BFM = BFModule(num_fea)

        # Reconstruction
        self.upsample = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),
            nn.Conv2d(num_fea, out_channels * (upscale_factor**2), 3, 1, 1),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x, **kwargs):

        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])

        bi = F.interpolate(x, scale_factor=self.upscale_factor, mode="bicubic", align_corners=False)
        # feature extraction
        fea = self.fea_conv(x)

        # LBlocks
        outs = []
        temp = fea
        for i in range(self.num_LBs):
            temp = self.LBs[i](temp)
            outs.append(temp)

        # BFM
        H = self.BFM(outs)

        # reconstruct
        out = self.upsample(H + fea)

        out =  out + bi
    
        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out
    

def mean_channels(x):
    assert x.dim() == 4
    spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (x.shape[2] * x.shape[3])


class MeanShift(nn.Conv2d):
    def __init__(self, mean=[0.4488, 0.4371, 0.4040], std=[1.0, 1.0, 1.0], sign=-1):
        super(MeanShift, self).__init__(3, 3, 1)
        std = torch.Tensor(std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * torch.Tensor(mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False


class CALayer(nn.Module):
    def __init__(self, num_fea):
        super(CALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_fea, num_fea // 8, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_fea // 8, num_fea, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, fea):

        return self.conv_du(fea)


# lightweight lattice block
class LLBlock(nn.Module):
    def __init__(self, num_fea):
        super(LLBlock, self).__init__()
        self.channel1 = num_fea // 2
        self.channel2 = num_fea - self.channel1
        self.convblock = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),
        )

        self.A_att_conv = CALayer(self.channel1)
        self.B_att_conv = CALayer(self.channel2)

        self.fuse1 = nn.Conv2d(num_fea, self.channel1, 1, 1, 0)
        self.fuse2 = nn.Conv2d(num_fea, self.channel2, 1, 1, 0)
        self.fuse = nn.Conv2d(num_fea, num_fea, 1, 1, 0)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.channel1, self.channel2], dim=1)

        x1 = self.convblock(x1)

        A = self.A_att_conv(x1)
        P = torch.cat((x2, A * x1), dim=1)

        B = self.B_att_conv(x2)
        Q = torch.cat((x1, B * x2), dim=1)

        c = torch.cat((self.fuse1(P), self.fuse2(Q)), dim=1)
        out = self.fuse(c)
        return out


# attention fuse
class AF(nn.Module):
    def __init__(self, num_fea):
        super(AF, self).__init__()
        self.CA1 = CALayer(num_fea)
        self.CA2 = CALayer(num_fea)
        self.fuse = nn.Conv2d(num_fea * 2, num_fea, 1)

    def forward(self, x1, x2):
        x1 = self.CA1(x1) * x1
        x2 = self.CA2(x2) * x2
        return self.fuse(torch.cat((x1, x2), dim=1))


# Feature enhancement block
class FEB(nn.Module):
    def __init__(self, num_fea):
        super(FEB, self).__init__()
        self.CB1 = LLBlock(num_fea)
        self.CB2 = LLBlock(num_fea)
        self.CB3 = LLBlock(num_fea)
        self.AF1 = AF(num_fea)
        self.AF2 = AF(num_fea)

    def forward(self, x):
        x1 = self.CB1(x)
        x2 = self.CB2(x1)
        x3 = self.CB3(x2)
        f1 = self.AF1(x3, x2)
        f2 = self.AF2(f1, x1)
        return x + f2


class RB(nn.Module):
    def __init__(self, num_fea):
        super(RB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_fea, num_fea * 2, 3, 1, 1),
            nn.LeakyReLU(0.05),
            nn.Conv2d(num_fea * 2, num_fea, 3, 1, 1),
        )

    def forward(self, x):
        return self.conv(x) + x


class BFModule(nn.Module):
    def __init__(self, num_fea):
        super(BFModule, self).__init__()
        self.conv4 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)
        self.conv3 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)
        self.fuse43 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)
        self.fuse32 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)
        self.conv1 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x_list):
        H4 = self.act(self.conv4(x_list[3]))
        H3_half = self.act(self.conv3(x_list[2]))
        H3 = self.fuse43(torch.cat([H4, H3_half], dim=1))
        H2_half = self.act(self.conv2(x_list[1]))
        H2 = self.fuse32(torch.cat([H3, H2_half], dim=1))
        H1_half = self.act(self.conv1(x_list[0]))
        H1 = torch.cat([H2, H1_half], dim=1)

        return H1





def count_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameters:" + str(k))


if __name__ == "__main__":
    x = torch.randn((1, 1, 64, 64)).cuda()
    net = FeNet(upscale_factor=2).cuda()
    out = net(x)
    print(out.shape)
