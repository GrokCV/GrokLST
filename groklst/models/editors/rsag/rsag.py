import torch
import torch.nn as nn
# import os
from ..rsag import common

# import common
import numpy as np

from mmengine.model import BaseModule
from mmagic.registry import MODELS



class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        if key.startswith("__"):
            raise AttributeError
        return self.get(key, None)

    def __setattr__(self, key, value):
        if key.startswith("__"):
            raise AttributeError("Cannot set magic attribute '{}'".format(key))
        self[key] = value


OPTS = AttrDict()
OPTS.seed = 123
OPTS.epoch = 1000
OPTS.output_model = "results"
OPTS.pre_train = True

OPTS.n_blocks = 30
OPTS.n_colors = 1
OPTS.n_feats = 16
OPTS.lr = 0.00002
# OPTS.scale_factor = 4
OPTS.negval = 0.2

OPTS.min_rmse = 1000.0
# OPTS.scale_lst = [pow(2, s + 1) for s in range(int(np.log2(OPTS.scale_factor)))]
# OPTS.gui_channels = 10

@MODELS.register_module()
class RSAG(BaseModule):
    """
    title: Recurrent Structure Attention Guidance for Depth Super-Resolution
    paper: https://dl.acm.org/doi/10.1609/aaai.v37i3.25440
    
    code: https://github.com/Yuanjiayii/DSR_RSAG
    """
    
    def __init__(self, 
                 scale_factor=4, 
                 gui_channels=10, 
                 opts=OPTS, 
                 conv=common.default_conv,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super(RSAG, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        # self.opts = opts
        assert scale_factor in [4, 8], f"scale_factor must be one of [4, 8]!"
        self.scale_lst = [pow(2, s + 1) for s in range(int(np.log2(scale_factor)))]
        self.phase = len(self.scale_lst)
        n_blocks = opts.n_blocks
        n_feats = opts.n_feats
        n_colors = opts.n_colors  # x channels
        kernel_size = 3
        act = nn.ReLU(True)

        self.att = [common.SAttention(n_feats * pow(2, p - 1)) for p in range(self.phase, 0, -1)]
        self.att = nn.ModuleList(self.att)
        self.upsample = nn.Upsample(scale_factor=max(self.scale_lst), mode="bicubic", align_corners=False)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode="bicubic", align_corners=False)
        self.head = conv(n_colors, n_feats, kernel_size)
        self.rgb_head = conv(gui_channels, n_feats, kernel_size)
        self.up = []
        for p in range(self.phase, 0, -1):
            self.up.append(conv(n_colors, n_feats * pow(2, p - 1), kernel_size))

        self.up = nn.ModuleList(self.up)

        self.down = [
            common.DownBlock(opts, 2, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1))
            for p in range(self.phase)
        ]
        self.down = nn.ModuleList(self.down)
        self.down_lf = [common.DownBlock(opts, 2, n_feats, n_feats, n_feats * 2)]
        self.down_lf = nn.ModuleList(self.down_lf)

        up_body_blocks = [
            [common.CBAM(conv, 4 * n_feats * pow(2, (p - 1)), kernel_size, act=act) for _ in range(n_blocks)]
            for p in range(self.phase, 1, -1)
        ]
        up_body_blocks.insert(
            0, [common.CBAM(conv, n_feats * pow(2, self.phase), kernel_size, act=act) for _ in range(n_blocks)]
        )
        up = [
            [
                common.Upsampler(conv, 2, n_feats * pow(2, self.phase), act=False),
                conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1),
            ]
        ]

        for p in range(self.phase - 1, 0, -1):  # 2,1
            up.append(
                [
                    common.Upsampler(conv, 2, 4 * n_feats * pow(2, p), act=False),
                    conv(4 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1),
                ]
            )

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(nn.Sequential(*up_body_blocks[idx], *up[idx]))

        self.up_lf = common.DeconvPReLu(n_feats * 4, n_feats, 5, stride=2, padding=2)
        self.tail = conv(4 * n_feats, n_colors, kernel_size)
        self.tail_lf = conv(n_feats * 2, n_colors, kernel_size)
        self.conv35 = common.DCN()

    def forward(self, lst, hr_guidance, **kwargs):
        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            lst = (lst - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            lst = (lst - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])
        
        low_up = self.upsample(lst)
        hr_0, lf_0 = self.conv35(low_up)

        x = self.head(hr_0)
        hr_guidance = self.rgb_head(hr_guidance)
        lf_res0 = self.head(lf_0)

        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)

        copie_rgb = []
        for idx in range(self.phase):
            copie_rgb.append(hr_guidance)
            hr_guidance = self.down[idx](hr_guidance)

        copies_lst = []
        for idx in range(self.phase):
            lst = self.upsample_2(lst)
            x_up = self.up[idx](lst)
            copies_lst.append(x_up)

        copies_res = []
        x1 = x
        for idx in range(self.phase):
            x = self.up_blocks[idx](x)
            copies_res.append(x)
            rgb_att = self.att[idx](copies_lst[idx], copie_rgb[self.phase - idx - 1])
            x = torch.cat((x, copies[self.phase - idx - 1], rgb_att, copies_lst[idx]), 1)

        sr = self.tail(x)
        lf_res = self.down_lf[0](lf_res0)
        lf_res = torch.cat((lf_res, copies_res[-2]), 1)
        lf_res = self.up_lf(lf_res)
        lf_res = torch.cat((lf_res, copies_res[-1]), 1)
        lf_res = self.tail_lf(lf_res)
        sr0 = sr + lf_0 + lf_res

        sr1 = self.head(sr0)
        copies_lst = []
        for idx in range(self.phase):
            copies_lst.append(sr1)
            sr1 = self.down[idx](sr1)

        for idx in range(self.phase):
            x1 = self.up_blocks[idx](x1)
            copies_res.append(x1)
            rgb_att = self.att[idx](copies_lst[self.phase - idx - 1], copie_rgb[self.phase - idx - 1])
            x1 = torch.cat((x1, copies[self.phase - idx - 1], rgb_att, copies_lst[self.phase - idx - 1]), 1)

        sr = self.tail(x1)
        lf_res = self.down_lf[0](lf_res0)
        lf_res = torch.cat((lf_res, copies_res[-2]), 1)
        lf_res = self.up_lf(lf_res)
        lf_res = torch.cat((lf_res, copies_res[-1]), 1)
        lf_res = self.tail_lf(lf_res)
        out = sr + lf_0 + lf_res

        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out


if __name__ == "__main__":
    x = torch.randn((1, 1, 28, 28)).cuda()

    hr_guidance = torch.randn((1, 10, 112, 112)).cuda()
    # hr_guidance = torch.randn((1, 10, 224, 224)).cuda()
    net = RSAG(4).cuda()
    out = net(x, hr_guidance)
    print(out.shape)
