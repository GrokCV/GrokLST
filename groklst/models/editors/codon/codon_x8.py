from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
from math import sqrt
from .ResCBAM import ChannelGate
from .CAC_module import CAC_channel as CHANNEL
from .CAC_module import CAC_spatial as SPATIAL


from mmengine.model import BaseModule
from mmagic.registry import MODELS


@MODELS.register_module()
class CODONNet8(BaseModule):
    """
    title: CODON: On Orchestrating Cross-Domain Attentions for Depth Super-Resolution
    paper: https://link.springer.com/article/10.1007/s11263-021-01545-w
    code: https://github.com/619862306/CODON/tree/master
    """

    def __init__(self, 
                 lst_channels=1, 
                 gui_channels=10,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        """
        no non_local
        """
        super(CODONNet8, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.upscale_factor = 8
        self.img_upsampler = nn.Upsample(scale_factor=self.upscale_factor, mode="bicubic", align_corners=False)
        self.input = nn.Conv2d(
            in_channels=lst_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv_input = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        # --------------------------------------------------------------------------------------------------------------------------#
        self.input_c = nn.Conv2d(
            in_channels=gui_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv_input_c = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_c = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        # --------------------------------------------------------------------------------------------------------------------------#
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(in_channels=64 * 2, out_channels=64 * 2, kernel_size=5, stride=1, padding=2, bias=False)
        self.confuse_fuse = nn.Conv2d(
            in_channels=64 * 2, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.conv11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # --------------------------------------------------------------------------------------------------------------------------#
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))
        self.attention_c0 = CHANNEL(128)
        self.attention_c1 = CHANNEL(128)
        self.attention_c2 = CHANNEL(128)
        self.attention_c3 = CHANNEL(128)
        self.attention_c4 = CHANNEL(128)
        self.attention_s0 = SPATIAL()
        self.attention_s1 = SPATIAL()
        self.attention_s2 = SPATIAL()
        self.attention_s3 = SPATIAL()
        self.attention_s4 = SPATIAL()
        self.attention_c5 = ChannelGate(64)
        self.attention_s5 = SPATIAL()

    def forward(self, x, hr_guidance, **kwargs):  # x深度图 y彩色图
        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])
            
        x = self.img_upsampler(x)
        residual = x
        inputs = self.relu(self.input(x))
        inputs = self.relu(self.conv_input(inputs))
        out = inputs
        inputs_c = self.relu(self.input_c(hr_guidance))
        inputs_c = self.relu(self.conv_input_c(inputs_c))
        out_c = inputs_c
        for _ in range(5):  # 网络一共五层MC
            out_MC_R1 = self.relu(self.conv1(out))
            out_MC_P1_c = self.relu(self.conv5(out_c))
            out_MC_P1 = self.relu(self.conv2(out))
            out_MC_R1_c = self.relu(self.conv4(out_c))
            out_MC_stage = torch.cat((out_MC_R1, out_MC_P1), 1)
            out_MC_stage_c = torch.cat((out_MC_R1_c, out_MC_P1_c), 1)
            out_MC_R2 = self.relu(self.conv3(out_MC_stage))
            out_MC_R2_c = self.relu(self.conv6(out_MC_stage_c))
            out_c = self.confuse_c(out_MC_R2_c)
            out = self.confuse(out_MC_R2)
            CAC_cat = torch.cat((out_c, out), 1)  # Fcat
            if _ == 0:
                CAC_channel = self.attention_c0(CAC_cat)
                CAC_spatial = self.attention_s0(CAC_cat)
                ad_CAC = CAC_channel * CAC_spatial
                out = out * ad_CAC
                out_c = out_c * ad_CAC
            if _ == 1:
                CAC_channel = self.attention_c1(CAC_cat)
                CAC_spatial = self.attention_s1(CAC_cat)
                ad_CAC = CAC_channel * CAC_spatial
                out = out * ad_CAC
                out_c = out_c * ad_CAC
            if _ == 2:
                CAC_channel = self.attention_c2(CAC_cat)
                CAC_spatial = self.attention_s2(CAC_cat)
                ad_CAC = CAC_channel * CAC_spatial
                out = out * ad_CAC
                out_c = out_c * ad_CAC
            if _ == 3:
                CAC_channel = self.attention_c3(CAC_cat)
                CAC_spatial = self.attention_s3(CAC_cat)
                ad_CAC = CAC_channel * CAC_spatial
                out = out * ad_CAC
                out_c = out_c * ad_CAC
            if _ == 4:
                CAC_channel = self.attention_c4(CAC_cat)
                CAC_spatial = self.attention_s4(CAC_cat)
                ad_CAC = CAC_channel * CAC_spatial
                out = out * ad_CAC
                out_c = out_c * ad_CAC

            out_c = torch.add(out_c, inputs_c)
            out = torch.add(out, inputs)
        fuse = torch.cat((out, out_c), 1)
        fuse = self.relu(self.conv7(fuse))
        out_fuse = fuse
        for _ in range(3):  # MC是一次循环
            out_fuse_MC_R1 = self.relu(self.conv8(out_fuse))
            out_fuse_MC_P1 = self.relu(self.conv9(out_fuse))
            out_fuse_MC_stage = torch.cat((out_fuse_MC_R1, out_fuse_MC_P1), 1)
            out_fuse_MC_R2 = self.relu(self.conv10(out_fuse_MC_stage))
            out_fuse = self.confuse_fuse(out_fuse_MC_R2)
            out_fuse = torch.add(out_fuse, fuse)  # 每次MC后add residual
        out = self.relu(self.conv11(out_fuse))
        out_fuse_final = self.output(out)
        out = torch.add(out_fuse_final, residual)
        
        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out


if __name__ == "__main__":
    x = torch.randn((1, 1, 28, 28)).cuda()
    hr_guidance = torch.randn((1, 10, 224, 224)).cuda()
    net = CODONNet8().cuda()
    out = net(x, hr_guidance)
    print(out.shape)
