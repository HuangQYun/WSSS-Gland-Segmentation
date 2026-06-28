from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 32, dropout: float = 0.05):
        super().__init__()
        self.inc = ConvBlock(in_channels, base, dropout=0.0)
        self.down1 = Down(base, base * 2, dropout=dropout)
        self.down2 = Down(base * 2, base * 4, dropout=dropout)
        self.down3 = Down(base * 4, base * 8, dropout=dropout)
        self.down4 = Down(base * 8, base * 12, dropout=dropout)
        self.up1 = Up(base * 12, base * 8, base * 8, dropout=dropout)
        self.up2 = Up(base * 8, base * 4, base * 4, dropout=dropout)
        self.up3 = Up(base * 4, base * 2, base * 2, dropout=dropout)
        self.up4 = Up(base * 2, base, base, dropout=0.0)
        self.outc = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def build_model(name: str = "unet", base: int = 32, dropout: float = 0.05) -> nn.Module:
    name = name.lower()
    if name in {"unet", "small_unet", "smallunet"}:
        return SmallUNet(base=base, dropout=dropout)
    raise ValueError(f"Unsupported model: {name}")
