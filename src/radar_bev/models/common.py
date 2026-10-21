from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from ..geometry import BEVGeometry, PolarToBEV


def _groups(channels: int) -> int:
    return 8 if channels % 8 == 0 else 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dimensions: int = 2) -> None:
        convolution = nn.Conv3d if dimensions == 3 else nn.Conv2d
        super().__init__(
            convolution(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(ConvNormAct(channels, channels), ConvNormAct(channels, channels))

    def forward(self, value: Tensor) -> Tensor:
        return value + self.body(value)


class BEVDecoder(nn.Module):
    """Create the frozen [B,128,50,59] adapter interface and a dense BEV head."""

    def __init__(self, in_channels: int, geometry: BEVGeometry) -> None:
        super().__init__()
        self.geometry = geometry
        self.stem = nn.Sequential(ConvNormAct(in_channels, 64), ResidualBlock(64))
        self.down1 = nn.Sequential(ConvNormAct(64, 96, stride=2), ResidualBlock(96))
        self.down2 = nn.Sequential(ConvNormAct(96, 128, stride=2), ResidualBlock(128))
        self.decode = nn.Sequential(ConvNormAct(128, 96), ResidualBlock(96), ConvNormAct(96, 64))
        self.logit_head = nn.Conv2d(64, 1, kernel_size=1)
        # CenterNet-style sparse-positive prior: only a tiny fraction of BEV cells are people.
        nn.init.constant_(self.logit_head.bias, -2.19)

    def encode_shared(self, bev_feature: Tensor) -> Tensor:
        return self.down2(self.down1(self.stem(bev_feature)))

    def decode_shared(self, shared_feature: Tensor) -> Tensor:
        decoded = self.decode(shared_feature)
        decoded = functional.interpolate(
            decoded,
            size=(self.geometry.x_bins, self.geometry.y_bins),
            mode="bilinear",
            align_corners=False,
        )
        return self.logit_head(decoded)


class RadarBEVBase(nn.Module):
    def __init__(self, geometry: BEVGeometry | None = None) -> None:
        super().__init__()
        self.geometry = geometry or BEVGeometry()
        self.polar_to_bev = PolarToBEV(self.geometry)

    @staticmethod
    def _validate_input(radar: Tensor) -> None:
        if radar.ndim != 5 or radar.shape[2:] != (2, 128, 128):
            raise ValueError("radar must be [batch,time,2,128,128]")



