import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.functional import max_pool2d
from mmengine.model import BaseModule
from mmagic.registry import MODELS


def tensor_pad(img, scale=32):
    h, w = img.size()[2:]

    pad_h = scale - h % scale
    pad_w = scale - w % scale

    padding = [pad_w, 0, pad_h, 0]
    img = pad(img, padding, "constant", 0)

    return img, pad_h, pad_w


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ConvReLU(nn.Module):
    def __init__(self, dep_in_channels, out_channels, kernel_size, stride, padding):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=dep_in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        xavier_init(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class DeConvReLU(nn.Module):
    def __init__(self, dep_in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(DeConvReLU, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels=dep_in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                       padding=padding, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        xavier_init(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

@MODELS.register_module()
class DSRN(BaseModule):
    """
    title: Depth Super-Resolution based on Deep Edge-Aware Learning
    paper: https://www.sciencedirect.com/science/article/abs/pii/S0031320320300790
    
    Note that there is a little bit different between the original Tensorflow code provided by the author.
    We add BN to DSRN to avoid gradient vanishing problem.
    """
    def __init__(self,
                 dep_in_channels=1, 
                 guidance_in_channels=3, 
                 upscale_factor=2,
                 norm_flag=0, 
                 norm_dict={'mean':None, 'std':None,'min':None, 'max':None}):
        super(DSRN, self).__init__()
        self.norm_flag = norm_flag
        self.norm_dict = norm_dict
        self.guidance_in_channels = guidance_in_channels
        self.dep_in_channels = dep_in_channels
        self.upscale_factor  = upscale_factor
        self.img_upsampler = nn.Upsample(
            scale_factor=self.upscale_factor,
            mode='bicubic',
            align_corners=False)

        self.rgb_1 = ConvReLU(guidance_in_channels, 64, 3, 1, 1)
        self.rgb_2 = ConvReLU(64, 128, 3, 1, 1)
        self.rgb_3 = ConvReLU(128, 256, 3, 1, 1)
        self.rgb_4 = ConvReLU(256, 512, 3, 1, 1)

        self.encoder_1 = nn.Sequential(
            ConvReLU(1, 64, 3, 1, 1),
            ConvReLU(64, 64, 3, 1, 1)
        )

        self.encoder_2 = nn.Sequential(
            ConvReLU(128, 128, 3, 1, 1),
            ConvReLU(128, 128, 3, 1, 1)
        )

        self.encoder_3 = nn.Sequential(
            ConvReLU(256, 256, 3, 1, 1),
            ConvReLU(256, 256, 3, 1, 1)
        )

        self.encoder_4 = nn.Sequential(
            ConvReLU(512, 512, 3, 1, 1),
            ConvReLU(512, 512, 3, 1, 1)
        )

        self.input_1 = ConvReLU(dep_in_channels, 64, 3, 1, 1)
        self.input_2 = ConvReLU(dep_in_channels, 128, 3, 1, 1)
        self.input_3 = ConvReLU(dep_in_channels, 256, 3, 1, 1)
        self.input_4 = ConvReLU(dep_in_channels, 512, 3, 1, 1)

        self.decoder_1 = nn.Sequential(
            ConvReLU(1024, 1024, 3, 1, 1),
            ConvReLU(1024, 1024, 3, 1, 1),
            DeConvReLU(1024, 512, 3, 2, 1, 1)
        )

        self.decoder_2 = nn.Sequential(
            ConvReLU(512 * 3, 512, 3, 1, 1),
            ConvReLU(512, 512, 3, 1, 1),
            DeConvReLU(512, 256, 3, 2, 1, 1)
        )

        self.decoder_3 = nn.Sequential(
            ConvReLU(256 * 3, 256, 3, 1, 1),
            ConvReLU(256, 256, 3, 1, 1),
            DeConvReLU(256, 128, 3, 2, 1, 1)
        )

        self.decoder_4 = nn.Sequential(
            ConvReLU(128 * 3, 128, 3, 1, 1),
            ConvReLU(128, 128, 3, 1, 1),
            DeConvReLU(128, 64, 3, 2, 1, 1)
        )

        self.tail = nn.Sequential(
            ConvReLU(64 * 3, 64, 3, 1, 1),
            ConvReLU(64, 64, 3, 1, 1),
            ConvReLU(64, dep_in_channels, 3, 1, 1),
        )

    def forward(self,  x, hr_guidance, **kwargs):

        # self.norm_flag == 0 means do not normalization.
        assert self.norm_flag in [0, 1, 2]
        if self.norm_flag == 1: # z-score
            assert self.norm_dict['mean'] is not None and self.norm_dict['std'] is not None
            x = (x - self.norm_dict['mean']) / self.norm_dict['std']
        elif self.norm_flag == 2: # min-max
            assert self.norm_dict['min'] is not None and self.norm_dict['max'] is not None
            x = (x - self.norm_dict['min']) / (self.norm_dict['max'] - self.norm_dict['min'])
            
        hr_depth = self.img_upsampler(x)
        # hr_depth, _, _ = tensor_pad(hr_depth, 4 * self.args.scale)
        # hr_guidance, h, w = tensor_pad(hr_guidance, 4 * self.args.scale)
        h = w = 0
        rgb_feature1 = self.rgb_1(hr_guidance)
        rgb_feature2 = self.rgb_2(max_pool2d(rgb_feature1, kernel_size=2))
        rgb_feature3 = self.rgb_3(max_pool2d(rgb_feature2, kernel_size=2))
        rgb_feature4 = self.rgb_4(max_pool2d(rgb_feature3, kernel_size=2))
        # print(self.rgb_1.layers[0].weight[:1, :,:,:])

        input_d1 = max_pool2d(hr_depth, kernel_size=2)
        input_feature1 = self.input_1(input_d1)

        input_d2 = max_pool2d(input_d1, kernel_size=2)
        input_feature2 = self.input_2(input_d2)

        input_d3 = max_pool2d(input_d2, kernel_size=2)
        input_feature3 = self.input_3(input_d3)

        input_d4 = max_pool2d(input_d3, kernel_size=2)
        input_feature4 = self.input_4(input_d4)

        enc_1 = self.encoder_1(hr_depth)

        enc_2 = max_pool2d(enc_1, kernel_size=2)

        enc_3 = self.encoder_2(torch.cat((enc_2, input_feature1), 1))

        enc_4 = max_pool2d(enc_3, kernel_size=2)

        enc_5 = self.encoder_3(torch.cat((enc_4, input_feature2), 1))

        enc_6 = max_pool2d(enc_5, kernel_size=2)

        enc_7 = self.encoder_4(torch.cat((enc_6, input_feature3), 1))

        enc_8 = max_pool2d(enc_7, kernel_size=2)

        dec_1 = self.decoder_1(torch.cat((enc_8, input_feature4), 1))

        dec_2 = self.decoder_2(torch.cat((dec_1, rgb_feature4, enc_7), 1))

        dec_3 = self.decoder_3(torch.cat((dec_2, rgb_feature3, enc_5), 1))
        dec_4 = self.decoder_4(torch.cat((dec_3, rgb_feature2, enc_3), 1))

        dec_5 = self.tail(torch.cat((dec_4, rgb_feature1, enc_1), 1)) + hr_depth

        out =  dec_5[:, :, h:, w:]
    
        if self.norm_flag == 1:
            out = out * self.norm_dict['std'] + self.norm_dict['mean']
        elif self.norm_flag == 2:
            out = out * (self.norm_dict['max'] - self.norm_dict['min']) + self.norm_dict['min'] 

        return out


if __name__ == "__main__":

    x = torch.randn((2, 1, 112, 112))
    hr_guidance = [torch.randn((9, 224, 224))]*2
    net = DSRN(1,9)
    out = net(x, hr_guidance)
    print(out.shape)