import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Model(nn.Module):
    def __init__(self, input_channel, n_class):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True)
        )

        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.up1 = up_conv(64, 32)

        # self.up2 = up_conv(32, 16)
        self.up2 = DoubleConv(64, 32)
        self.up3 = up_conv(32, 16)
        self.final = nn.Conv2d(32, n_class, kernel_size=3, padding=1)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.up1(e3)
        c1 = torch.cat((e2, e4), dim=1)
        e5 = self.up2(c1)
        # print(c1.shape)
        e6 = self.up3(e5)
        c2 = torch.cat((e6, e1), dim=1)
        out = self.final(c2)
        out = torch.sigmoid(out)
        return out


# model = Model(3, 1)
# inp = torch.rand(1, 3, 512, 512)
#
# outp = model(inp)
# print(outp.shape)