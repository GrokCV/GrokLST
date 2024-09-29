import torch
import math


class DenseBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation="relu", norm="batch"):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm == "batch":
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == "instance":
            self.bn = torch.nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == "relu":
            self.act = torch.nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = torch.nn.PReLU()
        elif self.activation == "lrelu":
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = torch.nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(torch.nn.Module):
    def __init__(
        self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation="prelu", norm=None
    ):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == "batch":
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == "instance":
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == "relu":
            self.act = torch.nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = torch.nn.PReLU()
        elif self.activation == "lrelu":
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = torch.nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


# class AvgpoolingBlock(torch.nn.Module):
#     def __init__(self, kernel_size=3, stride=1, padding=1, dilation, return_indices, bias=True, ceil_mode, activation='prelu', norm=None):
#         super(AvgpoolingBlock, self).__init__()
#         self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, dilation=1, return_indices=False, ceil_mode=False)
#         #self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

#         self.norm = norm
#         if self.norm =='batch':
#             self.bn = torch.nn.BatchNorm2d(output_size)
#         elif self.norm == 'instance':
#             self.bn = torch.nn.InstanceNorm2d(output_size)

#         self.activation = activation
#         if self.activation == 'relu':
#             self.act = torch.nn.ReLU(True)
#         elif self.activation == 'prelu':
#             self.act = torch.nn.PReLU()
#         elif self.activation == 'lrelu':
#             self.act = torch.nn.LeakyReLU(0.2, True)
#         elif self.activation == 'tanh':
#             self.act = torch.nn.Tanh()
#         elif self.activation == 'sigmoid':
#             self.act = torch.nn.Sigmoid()

#     def forward(self, x):
#         if self.norm is not None:
#             out = self.bn(self.avgpool(x))
#         else:
#             out = self.avgpool(x)

#         if self.activation is not None:
#             return self.act(out)
#         else:
#             return out

# class MaxpoolingBlock(torch.nn.Module):
#     def __init__(self, kernel_size=4, stride=4, padding=0, bias=True, activation='prelu', norm=None):
#         super(MaxpoolingBlock, self).__init__()
#         self.avgpool = torch.nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, return_indices=False, ceil_mode=False)
#         #self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

#         self.norm = norm
#         if self.norm =='batch':
#             self.bn = torch.nn.BatchNorm2d(output_size)
#         elif self.norm == 'instance':
#             self.bn = torch.nn.InstanceNorm2d(output_size)

#         self.activation = activation
#         if self.activation == 'relu':
#             self.act = torch.nn.ReLU(True)
#         elif self.activation == 'prelu':
#             self.act = torch.nn.PReLU()
#         elif self.activation == 'lrelu':
#             self.act = torch.nn.LeakyReLU(0.2, True)
#         elif self.activation == 'tanh':
#             self.act = torch.nn.Tanh()
#         elif self.activation == 'sigmoid':
#             self.act = torch.nn.Sigmoid()

#     def forward(self, x):
#         if self.norm is not None:
#             out = self.bn(self.avgpool(x))
#         else:
#             out = self.avgpool(x)

#         if self.activation is not None:
#             return self.act(out)
#         else:
#             return out


class DilaConvBlock(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=2,
        bias=True,
        activation="prelu",
        norm=None,
    ):
        super(DilaConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, dilation, bias=bias)

        self.norm = norm
        if self.norm == "batch":
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == "instance":
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == "relu":
            self.act = torch.nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = torch.nn.PReLU()
        elif self.activation == "lrelu":
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = torch.nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(
        self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation="prelu", norm=None
    ):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == "batch":
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == "instance":
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == "relu":
            self.act = torch.nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = torch.nn.PReLU()
        elif self.activation == "lrelu":
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = torch.nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation="prelu", norm="batch"):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == "batch":
            self.bn = torch.nn.BatchNorm2d(num_filter)
        elif norm == "instance":
            self.bn = torch.nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == "relu":
            self.act = torch.nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = torch.nn.PReLU()
        elif self.activation == "lrelu":
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = torch.nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        return out


class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation="prelu", norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class UpBlockPix(torch.nn.Module):
    def __init__(
        self, num_filter, kernel_size=8, stride=4, padding=2, scale=4, bias=True, activation="prelu", norm=None
    ):
        super(UpBlockPix, self).__init__()
        self.up_conv1 = Upsampler(scale, num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = Upsampler(scale, num_filter)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlock(torch.nn.Module):
    def __init__(
        self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation="prelu", norm=None
    ):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlockPix(torch.nn.Module):
    def __init__(
        self,
        num_filter,
        kernel_size=8,
        stride=4,
        padding=2,
        num_stages=1,
        scale=4,
        bias=True,
        activation="prelu",
        norm=None,
    ):
        super(D_UpBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = Upsampler(scale, num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = Upsampler(scale, num_filter)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation="prelu", norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class DownBlockPix(torch.nn.Module):
    def __init__(
        self, num_filter, kernel_size=8, stride=4, padding=2, scale=4, bias=True, activation="prelu", norm=None
    ):
        super(DownBlockPix, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = Upsampler(scale, num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlock(torch.nn.Module):
    def __init__(
        self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation="prelu", norm=None
    ):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlockPix(torch.nn.Module):
    def __init__(
        self,
        num_filter,
        kernel_size=8,
        stride=4,
        padding=2,
        num_stages=1,
        scale=4,
        bias=True,
        activation="prelu",
        norm=None,
    ):
        super(D_DownBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = Upsampler(scale, num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class PSBlock(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        scale_factor,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        activation="prelu",
        norm="batch",
    ):
        super(PSBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size * scale_factor**2, kernel_size, stride, padding, bias=bias)
        self.ps = torch.nn.PixelShuffle(scale_factor)

        self.norm = norm
        if self.norm == "batch":
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif norm == "instance":
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == "relu":
            self.act = torch.nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = torch.nn.PReLU()
        elif self.activation == "lrelu":
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = torch.nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))

        if self.activation is not None:
            out = self.act(out)
        return out


class Upsampler(torch.nn.Module):
    def __init__(self, scale, n_feat, bn=False, act="prelu", bias=True):
        super(Upsampler, self).__init__()
        modules = []
        for _ in range(int(math.log(scale, 2))):
            modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
            modules.append(torch.nn.PixelShuffle(2))
            if bn:
                modules.append(torch.nn.BatchNorm2d(n_feat))
            # modules.append(torch.nn.PReLU())
        self.up = torch.nn.Sequential(*modules)

        self.activation = act
        if self.activation == "relu":
            self.act = torch.nn.ReLU(True)
        elif self.activation == "prelu":
            self.act = torch.nn.PReLU()
        elif self.activation == "lrelu":
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == "tanh":
            self.act = torch.nn.Tanh()
        elif self.activation == "sigmoid":
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out


class Upsample2xBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, upsample="deconv", activation="relu", norm="batch"):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2
        # 1. Deconvolution (Transposed convolution)
        if upsample == "deconv":
            self.upsample = DeconvBlock(
                input_size, output_size, kernel_size=4, stride=2, padding=1, bias=bias, activation=activation, norm=norm
            )

        # 2. Sub-pixel convolution (Pixel shuffler)
        elif upsample == "ps":
            self.upsample = PSBlock(
                input_size, output_size, scale_factor=scale_factor, bias=bias, activation=activation, norm=norm
            )

        # 3. Resize and Convolution
        elif upsample == "rnc":
            self.upsample = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=scale_factor, mode="nearest"),
                ConvBlock(
                    input_size,
                    output_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    activation=activation,
                    norm=norm,
                ),
            )

    def forward(self, x):
        out = self.upsample(x)
        return out


class UpBlock_x8(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, bias=True, activation="prelu", norm=None):
        super(UpBlock_x8, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv4 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)
        self.up_conv5 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv6 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv7 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        h0 = self.up_conv1(x)
        h1 = self.up_conv2(h0)
        h2 = self.up_conv3(h1)
        l0 = self.up_conv4(h2)
        h3 = self.up_conv5(l0 - x)
        h4 = self.up_conv6(h3)
        h5 = self.up_conv7(h4)
        return h2 + h5


class DownBlock_x8(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, bias=True, activation="prelu", norm=None):
        super(DownBlock_x8, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv4 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv5 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h1 = self.down_conv2(l0)
        h2 = self.down_conv3(h1)
        h3 = self.down_conv4(h2)
        l1 = self.down_conv5(h3 - x)
        return l0 + l1


class D_DownBlock_x8(torch.nn.Module):
    def __init__(
        self, num_filter, kernel_size=6, stride=2, padding=2, num_stages=1, bias=True, activation="prelu", norm=None
    ):
        super(D_DownBlock_x8, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv4 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv5 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)

    def forward(self, x):
        x1 = self.conv(x)
        l0 = self.down_conv1(x1)
        h0 = self.down_conv2(l0)
        h1 = self.down_conv3(h0)
        h2 = self.down_conv4(h1)
        l1 = self.down_conv5(h2 - x1)
        return l1 + l0


class D_UpBlock_x8(torch.nn.Module):
    def __init__(
        self, num_filter, kernel_size=6, stride=2, padding=2, num_stages=1, bias=True, activation="prelu", norm=None
    ):
        super(D_UpBlock_x8, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

        self.up_conv4 = ConvBlock(num_filter, num_filter, 12, 8, 2, activation, norm=None)
        self.up_conv5 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv6 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv7 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        h1 = self.up_conv2(h0)
        h2 = self.up_conv3(h1)
        l0 = self.up_conv4(h2)
        h3 = self.up_conv5(l0 - x)
        h4 = self.up_conv6(h3)
        h5 = self.up_conv7(h4)
        return h2 + h5


class PriorBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation="prelu", norm=None):
        super(PriorBlock, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.dilaconv2 = DilaConvBlock(2 * 64, 64, 3, 1, 2, dilation=2, activation="prelu", norm=None)
        self.dilaconv3 = DilaConvBlock(3 * 64, 64, 3, 1, 3, dilation=3, activation="prelu", norm=None)
        self.dilaconv4 = DilaConvBlock(4 * 64, 64, 3, 1, 4, dilation=4, activation="prelu", norm=None)
        self.direct_up1 = DeconvBlock(4 * 64, 64, kernel_size, stride, padding, activation="prelu", norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((x_prior1, x_prior2, x_prior3, x_prior4), 1)

        h_prior1 = self.direct_up1(concat_p1)
        # out = self.output_conv1(h_prior1)

        return h_prior1


class PriorBlock2(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation="prelu", norm=None):
        super(PriorBlock2, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.dilaconv2 = DilaConvBlock(3 * 64, 64, 3, 1, 2, dilation=2, activation="prelu", norm=None)
        self.dilaconv3 = DilaConvBlock(4 * 64, 64, 3, 1, 3, dilation=3, activation="prelu", norm=None)
        self.dilaconv4 = DilaConvBlock(5 * 64, 64, 3, 1, 4, dilation=4, activation="prelu", norm=None)
        self.direct_up1 = DeconvBlock(4 * 64, 64, kernel_size, stride, padding, activation="prelu", norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((x_prior1, x_prior2, x_prior3, x_prior4), 1)

        h_prior1 = self.direct_up1(concat_p1)
        # out = self.output_conv1(h_prior1)

        return h_prior1


class PriorBlock3(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation="prelu", norm=None):
        super(PriorBlock3, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.dilaconv2 = DilaConvBlock(4 * 64, 64, 3, 1, 2, dilation=2, activation="prelu", norm=None)
        self.dilaconv3 = DilaConvBlock(5 * 64, 64, 3, 1, 3, dilation=3, activation="prelu", norm=None)
        self.dilaconv4 = DilaConvBlock(6 * 64, 64, 3, 1, 4, dilation=4, activation="prelu", norm=None)
        self.direct_up1 = DeconvBlock(4 * 64, 64, kernel_size, stride, padding, activation="prelu", norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((x_prior1, x_prior2, x_prior3, x_prior4), 1)

        h_prior1 = self.direct_up1(concat_p1)
        # out = self.output_conv1(h_prior1)

        return h_prior1


class PriorBlock4(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation="prelu", norm=None):
        super(PriorBlock4, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.dilaconv2 = DilaConvBlock(5 * 64, 64, 3, 1, 2, dilation=2, activation="prelu", norm=None)
        self.dilaconv3 = DilaConvBlock(6 * 64, 64, 3, 1, 3, dilation=3, activation="prelu", norm=None)
        self.dilaconv4 = DilaConvBlock(7 * 64, 64, 3, 1, 4, dilation=4, activation="prelu", norm=None)
        self.direct_up1 = DeconvBlock(4 * 64, 64, kernel_size, stride, padding, activation="prelu", norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((x_prior1, x_prior2, x_prior3, x_prior4), 1)

        h_prior1 = self.direct_up1(concat_p1)
        # out = self.output_conv1(h_prior1)

        return h_prior1


class PriorBlock5(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation="prelu", norm=None):
        super(PriorBlock5, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.dilaconv2 = DilaConvBlock(6 * 64, 64, 3, 1, 2, dilation=2, activation="prelu", norm=None)
        self.dilaconv3 = DilaConvBlock(7 * 64, 64, 3, 1, 3, dilation=3, activation="prelu", norm=None)
        self.dilaconv4 = DilaConvBlock(8 * 64, 64, 3, 1, 4, dilation=4, activation="prelu", norm=None)
        self.direct_up1 = DeconvBlock(4 * 64, 64, kernel_size, stride, padding, activation="prelu", norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((x_prior1, x_prior2, x_prior3, x_prior4), 1)

        h_prior1 = self.direct_up1(concat_p1)
        # out = self.output_conv1(h_prior1)

        return h_prior1


class PriorBlock5(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation="prelu", norm=None):
        super(PriorBlock5, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.dilaconv2 = DilaConvBlock(6 * 64, 64, 3, 1, 2, dilation=2, activation="prelu", norm=None)
        self.dilaconv3 = DilaConvBlock(7 * 64, 64, 3, 1, 3, dilation=3, activation="prelu", norm=None)
        self.dilaconv4 = DilaConvBlock(8 * 64, 64, 3, 1, 4, dilation=4, activation="prelu", norm=None)
        self.direct_up1 = DeconvBlock(4 * 64, 64, kernel_size, stride, padding, activation="prelu", norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((x_prior1, x_prior2, x_prior3, x_prior4), 1)

        h_prior1 = self.direct_up1(concat_p1)
        # out = self.output_conv1(h_prior1)
        return h_prior1


class PriorBlock6(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation="prelu", norm=None):
        super(PriorBlock6, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.dilaconv2 = DilaConvBlock(7 * 64, 64, 3, 1, 2, dilation=2, activation="prelu", norm=None)
        self.dilaconv3 = DilaConvBlock(8 * 64, 64, 3, 1, 3, dilation=3, activation="prelu", norm=None)
        self.dilaconv4 = DilaConvBlock(9 * 64, 64, 3, 1, 4, dilation=4, activation="prelu", norm=None)
        self.direct_up1 = DeconvBlock(4 * 64, 64, kernel_size, stride, padding, activation="prelu", norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((x_prior1, x_prior2, x_prior3, x_prior4), 1)

        h_prior1 = self.direct_up1(concat_p1)
        # out = self.output_conv1(h_prior1)
        return h_prior1


class LA_attentionBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation="prelu", norm=None):
        super(LA_attentionBlock, self).__init__()

        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)

        self.up_1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation="prelu", norm=None)

        self.act_1 = torch.nn.ReLU(True)

    def forward(self, x):

        p1 = self.avgpool_1(x)
        # print(p1.shape, '111111111111')
        l0 = self.up_1(p1)
        # print(l0.shape, 'l0l0l0l0l0l0l0l0l0l0l0l0')
        act1 = self.act_1(x - l0)
        # print(act1.shape, 'acacacacacaacacaacac')
        out_la = x + 0.2 * (act1 * x)
        # print(out_la.shape, 'oooooooooooooooooooooooooooo')

        return out_la


class LA_attentionBlock2(torch.nn.Module):
    def __init__(
        self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation="prelu", norm=None
    ):
        super(LA_attentionBlock2, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation="prelu", norm=None)

        self.avgpool_1 = torch.nn.AvgPool2d(8, 4, 2)

        self.up_1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation="prelu", norm=None)

        self.act_1 = torch.nn.ReLU(True)

    def forward(self, x):

        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        # print(p1.shape, '111111111111')
        l0 = self.up_1(p1)
        # print(l0.shape, 'l0l0l0l0l0l0l0l0l0l0l0l0')
        act1 = self.act_1(x - l0)
        # print(act1.shape, 'acacacacacaacacaacac')
        out_la = x + 0.2 * (act1 * x)
        # print(out_la.shape, 'oooooooooooooooooooooooooooo')

        return out_la


class GA_attentionBlock(torch.nn.Module):
    def __init__(self, num_filter):
        super(GA_attentionBlock, self).__init__()

        self.g_aver_pooling1 = torch.nn.AdaptiveAvgPool2d(1)

        self.fc1 = torch.nn.Linear(in_features=num_filter, out_features=round(num_filter / 16))

        self.act_1 = torch.nn.ReLU(True)

        self.fc2 = torch.nn.Linear(in_features=round(num_filter / 16), out_features=num_filter)

        self.act_2 = torch.nn.Sigmoid()

        # self.avgpool_1 = torch.nn.AvgPool2d(8, 4, 2)

        # self.up_1 = DeconvBlock(num_filter, num_filter , kernel_size, stride, padding, activation='prelu', norm=None)

        # self.act_1 = torch.nn.ReLU(True)

    def forward(self, x):

        x1 = self.g_aver_pooling1(x)
        x1 = x1.view(x1.size(0), -1)
        c1 = self.fc1(x1)
        act1 = self.act_1(c1)
        c2 = self.fc2(act1)
        act2 = self.act_2(c2)
        act2 = act2.view(act2.size(0), act2.size(1), 1, 1)

        y = x * act2

        return y


class GA_res_attentionBlock(torch.nn.Module):
    def __init__(self, num_filter):
        super(GA_res_attentionBlock, self).__init__()

        self.g_aver_pooling1 = torch.nn.AdaptiveAvgPool2d(1)

        self.fc1 = torch.nn.Linear(in_features=num_filter, out_features=round(num_filter / 16))

        self.act_1 = torch.nn.ReLU(True)

        self.fc2 = torch.nn.Linear(in_features=round(num_filter / 16), out_features=num_filter)

        self.act_2 = torch.nn.Sigmoid()

        # self.avgpool_1 = torch.nn.AvgPool2d(8, 4, 2)

        # self.up_1 = DeconvBlock(num_filter, num_filter , kernel_size, stride, padding, activation='prelu', norm=None)

        # self.act_1 = torch.nn.ReLU(True)

    def forward(self, x):

        x1 = self.g_aver_pooling1(x)
        x1 = x1.view(x1.size(0), -1)
        c1 = self.fc1(x1)
        act1 = self.act_1(c1)
        c2 = self.fc2(act1)
        act2 = self.act_2(c2)
        act2 = act2.view(act2.size(0), act2.size(1), 1, 1)

        y = x + x * act2

        return y


class UpBlockP(torch.nn.Module):
    def __init__(self, num, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation="prelu", norm=None):
        super(UpBlockP, self).__init__()
        self.conv1 = ConvBlock(num * num_filter, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv1(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


# 2019.07.22
class MultiViewBlock1(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation="prelu", norm=None):
        super(MultiViewBlock1, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.dilaconv2 = DilaConvBlock(2 * 64, 64, 3, 1, 2, dilation=2, activation="prelu", norm=None)
        self.dilaconv3 = DilaConvBlock(3 * 64, 64, 3, 1, 3, dilation=3, activation="prelu", norm=None)
        self.dilaconv4 = DilaConvBlock(4 * 64, 64, 3, 1, 4, dilation=4, activation="prelu", norm=None)
        self.dilaconv1_2 = DilaConvBlock(5 * 64, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation="prelu", norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        # print(concat_p1.shape, '111111111111111111')
        x_prior1_2 = self.dilaconv1_2(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)
        # out = self.output_conv1(h_prior1)

        return h_prior1


class MultiViewBlock2(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation="prelu", norm=None):
        super(MultiViewBlock2, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.dilaconv2 = DilaConvBlock(3 * 64, 64, 3, 1, 2, dilation=2, activation="prelu", norm=None)
        self.dilaconv3 = DilaConvBlock(4 * 64, 64, 3, 1, 3, dilation=3, activation="prelu", norm=None)
        self.dilaconv4 = DilaConvBlock(5 * 64, 64, 3, 1, 4, dilation=4, activation="prelu", norm=None)
        self.dilaconv1_2 = DilaConvBlock(6 * 64, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation="prelu", norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)
        # out = self.output_conv1(h_prior1)

        return h_prior1


class MultiViewBlock3(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation="prelu", norm=None):
        super(MultiViewBlock3, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.dilaconv2 = DilaConvBlock(4 * 64, 64, 3, 1, 2, dilation=2, activation="prelu", norm=None)
        self.dilaconv3 = DilaConvBlock(5 * 64, 64, 3, 1, 3, dilation=3, activation="prelu", norm=None)
        self.dilaconv4 = DilaConvBlock(6 * 64, 64, 3, 1, 4, dilation=4, activation="prelu", norm=None)
        self.dilaconv1_2 = DilaConvBlock(7 * 64, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation="prelu", norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)

        return h_prior1


class MultiViewBlock4(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation="prelu", norm=None):
        super(MultiViewBlock4, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.dilaconv2 = DilaConvBlock(5 * 64, 64, 3, 1, 2, dilation=2, activation="prelu", norm=None)
        self.dilaconv3 = DilaConvBlock(6 * 64, 64, 3, 1, 3, dilation=3, activation="prelu", norm=None)
        self.dilaconv4 = DilaConvBlock(7 * 64, 64, 3, 1, 4, dilation=4, activation="prelu", norm=None)
        self.dilaconv1_2 = DilaConvBlock(8 * 64, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation="prelu", norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)

        return h_prior1


class MultiViewBlock5(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation="prelu", norm=None):
        super(MultiViewBlock5, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.dilaconv2 = DilaConvBlock(6 * 64, 64, 3, 1, 2, dilation=2, activation="prelu", norm=None)
        self.dilaconv3 = DilaConvBlock(7 * 64, 64, 3, 1, 3, dilation=3, activation="prelu", norm=None)
        self.dilaconv4 = DilaConvBlock(8 * 64, 64, 3, 1, 4, dilation=4, activation="prelu", norm=None)
        self.dilaconv1_2 = DilaConvBlock(9 * 64, 64, 3, 1, 1, dilation=1, activation="prelu", norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation="prelu", norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):

        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)

        return h_prior1


class FeedbackBlock1(torch.nn.Module):
    def __init__(
        self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation="prelu", norm=None
    ):
        super(FeedbackBlock1, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation="prelu", norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter, 8, 4, 2, activation="prelu", norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):

        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1 * (act1 * x)

        h0 = self.up_conv1(out_la)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - out_la)
        return h1 + h0


class FeedbackBlock2(torch.nn.Module):
    def __init__(
        self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation="prelu", norm=None
    ):
        super(FeedbackBlock2, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation="prelu", norm=None)
        self.conv1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation="prelu", norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter, 8, 4, 2, activation="prelu", norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):

        x = self.down1(x)
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1 * (act1 * x)

        h0 = self.up_conv1(out_la)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - out_la)
        return h1 + h0


class channel_attentionBlock(torch.nn.Module):
    def __init__(self, num_filter):
        super(channel_attentionBlock, self).__init__()

        self.g_aver_pooling1 = torch.nn.AdaptiveAvgPool2d(1)

        self.fc1 = torch.nn.Linear(in_features=num_filter, out_features=round(num_filter / 16))

        self.act_1 = torch.nn.ReLU(True)

        self.fc2 = torch.nn.Linear(in_features=round(num_filter / 16), out_features=num_filter)

        self.act_2 = torch.nn.Sigmoid()

        # self.avgpool_1 = torch.nn.AvgPool2d(8, 4, 2)

        # self.up_1 = DeconvBlock(num_filter, num_filter , kernel_size, stride, padding, activation='prelu', norm=None)

        # self.act_1 = torch.nn.ReLU(True)

    def forward(self, x):

        x1 = self.g_aver_pooling1(x)
        x1 = x1.view(x1.size(0), -1)
        c1 = self.fc1(x1)
        act1 = self.act_1(c1)
        c2 = self.fc2(act1)
        act2 = self.act_2(c2)
        act2 = act2.view(act2.size(0), act2.size(1), 1, 1)

        y = x + x * act2

        return y
