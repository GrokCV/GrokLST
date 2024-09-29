from torch import nn
import torch


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c//2)
        # return x * y.expand_as(x)
        return y




class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, stride=2, bias=False
        )
        # self.fc = nn.Linear(channel, channel // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2)
        y = y.squeeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        # return x * y.expand_as(x)
        return y


if __name__ == "__main__":
    x = torch.randn(2, 16, 32,32).cuda()
    # net = SELayer(16, 16).cuda()
    net = eca_layer(16).cuda()
    out = net(x)
    print(out.shape)