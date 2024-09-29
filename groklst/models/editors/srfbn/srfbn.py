import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, MeanShift
from mmagic.registry import MODELS

from mmengine.model import BaseModule


class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor, act_type, norm_type):
        super(FeedbackBlock, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        self.num_groups = num_groups

        self.compress_in = ConvBlock(
            2 * num_features, num_features, kernel_size=1, act_type=act_type, norm_type=norm_type
        )

        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        for idx in range(self.num_groups):
            self.upBlocks.append(
                DeconvBlock(
                    num_features,
                    num_features,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    act_type=act_type,
                    norm_type=norm_type,
                )
            )
            self.downBlocks.append(
                ConvBlock(
                    num_features,
                    num_features,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    act_type=act_type,
                    norm_type=norm_type,
                    valid_padding=False,
                )
            )
            if idx > 0:
                self.uptranBlocks.append(
                    ConvBlock(
                        num_features * (idx + 1),
                        num_features,
                        kernel_size=1,
                        stride=1,
                        act_type=act_type,
                        norm_type=norm_type,
                    )
                )
                self.downtranBlocks.append(
                    ConvBlock(
                        num_features * (idx + 1),
                        num_features,
                        kernel_size=1,
                        stride=1,
                        act_type=act_type,
                        norm_type=norm_type,
                    )
                )

        self.compress_out = ConvBlock(
            num_groups * num_features, num_features, kernel_size=1, act_type=act_type, norm_type=norm_type
        )

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)

        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)  # when idx == 0, lr_features == [x]
            if idx > 0:
                LD_L = self.uptranBlocks[idx - 1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx - 1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)  # leave out input x, i.e. lr_features[0]
        output = self.compress_out(output)

        self.last_hidden = output

        return output

    def reset_state(self):
        self.should_reset = True

@MODELS.register_module()
class SRFBN(BaseModule):
    """
    title: Feedback Network for Image Super-Resolution
    paper: https://ieeexplore.ieee.org/document/8953436
    code: https://github.com/Paper99/SRFBN_CVPR19
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        upscale_factor=2,
        num_features=32,
        num_steps=4,
        num_groups=3,
        act_type="prelu",
        norm_type=None,
        norm_flag=0, 
        norm_dict={'mean':None, 'std':None,'min':None, 'max':None}
    ):
        super(SRFBN, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict

        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        self.num_steps = num_steps
        self.num_features = num_features
        self.upscale_factor = upscale_factor

        # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = MeanShift(rgb_mean, rgb_std)

        # LR feature extraction block
        self.conv_in = ConvBlock(in_channels, 4 * num_features, kernel_size=3, act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4 * num_features, num_features, kernel_size=1, act_type=act_type, norm_type=norm_type)

        # basic block
        self.block = FeedbackBlock(num_features, num_groups, upscale_factor, act_type, norm_type)

        # reconstruction block
        # uncomment for pytorch 0.4.0
        # self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')

        self.out = DeconvBlock(
            num_features,
            num_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            act_type="prelu",
            norm_type=norm_type,
        )
        self.conv_out = ConvBlock(num_features, out_channels, kernel_size=3, act_type=None, norm_type=norm_type)

        # self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, x, **kwargs):
        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])

        self._reset_state()

        # x = self.sub_mean(x)
        # uncomment for pytorch 0.4.0
        # inter_res = self.upsample(x)

        # comment for pytorch 0.4.0
        inter_res = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode="bilinear", align_corners=False)

        x = self.conv_in(x)
        x = self.feat_in(x)

        outs = []
        for _ in range(self.num_steps):
            h = self.block(x)

            h = torch.add(inter_res, self.conv_out(self.out(h)))
            # h = self.add_mean(h)
            outs.append(h)

        out =  outs[-1]  # return output of every timesteps
    
        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out

    def _reset_state(self):
        self.block.reset_state()


if __name__ == "__main__":
    x = torch.randn((1, 1, 112, 112)).cuda()
    net = SRFBN(in_channels=1, out_channels=1, upscale_factor=2).cuda()
    out = net(x)
    print(out.shape)
