"""Canonical 1.203M DREAM-BEV adapted only at the public CARRADA sensor grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import CARRADABEVGeometry

try:
    from mamba_ssm import Mamba
except (ImportError, OSError):
    Mamba = None


def _groups(channels: int) -> int:
    return 8 if channels % 8 == 0 else 1


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(ConvNormAct(channels, channels), ConvNormAct(channels, channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.body(value)


class CARRADAPolarToBEV(nn.Module):
    """Sample public CARRADA polar features on the canonical metric BEV lattice."""

    def __init__(
        self,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        geometry: CARRADABEVGeometry | None = None,
    ) -> None:
        super().__init__()
        self.geometry = geometry or CARRADABEVGeometry()
        ranges = np.asarray(range_grid, dtype=np.float32)
        angles = np.asarray(angle_grid, dtype=np.float32)
        if ranges.shape != (128,) or angles.shape != (128,):
            raise ValueError("CARRADA range and angle grids must each contain 128 bins")
        if not (np.all(np.diff(ranges) > 0) and np.all(np.diff(angles) > 0)):
            raise ValueError("CARRADA sensor grids must be strictly increasing")
        self.register_buffer("sampling_grid", self._build_sampling_grid(ranges, angles), persistent=False)

    def _build_sampling_grid(self, ranges: np.ndarray, angles: np.ndarray) -> torch.Tensor:
        x = self.geometry.x_min_m + np.arange(self.geometry.x_bins, dtype=np.float32) * self.geometry.resolution_m
        y = self.geometry.y_min_m + np.arange(self.geometry.y_bins, dtype=np.float32) * self.geometry.resolution_m
        grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
        radial_distance = np.sqrt(grid_x**2 + grid_y**2)
        azimuth = np.arctan2(grid_x, grid_y)
        index_axis = np.arange(128, dtype=np.float32)
        range_index = np.interp(radial_distance, ranges, index_axis, left=-1.0, right=128.0)
        angle_index = np.interp(azimuth, angles, index_axis, left=-1.0, right=128.0)
        range_normalized = 2.0 * range_index / 127.0 - 1.0
        angle_normalized = 2.0 * angle_index / 127.0 - 1.0
        sampling = np.stack((angle_normalized, range_normalized), axis=-1)[None]
        return torch.from_numpy(sampling.astype(np.float32))

    def forward(self, polar_feature: torch.Tensor) -> torch.Tensor:
        if polar_feature.ndim != 4 or polar_feature.shape[-2:] != (128, 128):
            raise ValueError("polar_feature must be [batch,channels,128,128]")
        grid = self.sampling_grid.to(device=polar_feature.device, dtype=polar_feature.dtype)
        return F.grid_sample(
            polar_feature,
            grid.expand(polar_feature.shape[0], -1, -1, -1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


class _CausalMambaTokenMixer(nn.Module):
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError("canonical DREAM-BEV requires a working mamba_ssm CUDA environment")
        if min(d_state, d_conv, expand) < 1:
            raise ValueError("Mamba d_state, d_conv, and expand must be positive")
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.mixer = Mamba(
            d_model=channels,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(tokens + self.mixer(tokens))


class BEVDecoder(nn.Module):
    def __init__(self, in_channels: int, geometry: CARRADABEVGeometry) -> None:
        super().__init__()
        self.geometry = geometry
        self.stem = nn.Sequential(ConvNormAct(in_channels, 64), ResidualBlock(64))
        self.down1 = nn.Sequential(ConvNormAct(64, 96, stride=2), ResidualBlock(96))
        self.down2 = nn.Sequential(ConvNormAct(96, 128, stride=2), ResidualBlock(128))
        self.decode = nn.Sequential(ConvNormAct(128, 96), ResidualBlock(96), ConvNormAct(96, 64))
        self.logit_head = nn.Conv2d(64, 1, kernel_size=1)
        nn.init.constant_(self.logit_head.bias, -2.19)

    def encode_shared(self, bev_feature: torch.Tensor) -> torch.Tensor:
        return self.down2(self.down1(self.stem(bev_feature)))

    def decode_shared(self, shared_feature: torch.Tensor) -> torch.Tensor:
        decoded = self.decode(shared_feature)
        decoded = F.interpolate(
            decoded,
            size=(self.geometry.x_bins, self.geometry.y_bins),
            mode="bilinear",
            align_corners=False,
        )
        return self.logit_head(decoded)


@dataclass(frozen=True)
class DREAMBEVOutput:
    logits: torch.Tensor
    feature: torch.Tensor
    polar_feature: torch.Tensor


class DREAMBEV(nn.Module):
    """The paper radar anchor with only its parameter-free CARRADA grid sampler adapted."""

    def __init__(
        self,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        geometry: CARRADABEVGeometry | None = None,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ) -> None:
        super().__init__()
        self.geometry = geometry or CARRADABEVGeometry()
        self.stem = nn.Sequential(ConvNormAct(2, 64, stride=2), ConvNormAct(64, 64))
        self.temporal = _CausalMambaTokenMixer(
            64,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.reduce = nn.Sequential(ConvNormAct(64, 48), ResidualBlock(48))
        self.polar_to_bev = CARRADAPolarToBEV(range_grid, angle_grid, self.geometry)
        self.decoder = BEVDecoder(48, self.geometry)

    @staticmethod
    def _validate_input(radar: torch.Tensor) -> None:
        if radar.ndim != 5 or radar.shape[2:] != (2, 128, 128):
            raise ValueError("radar must be [batch,time,2,128,128]")

    def forward_polar_feature(self, radar: torch.Tensor) -> torch.Tensor:
        self._validate_input(radar)
        batch, frames = radar.shape[:2]
        encoded = self.stem(radar.reshape(batch * frames, 2, 128, 128))
        _, channels, height, width = encoded.shape
        sequence = encoded.reshape(batch, frames, channels, height, width)
        tokens = sequence.permute(0, 3, 4, 1, 2).reshape(batch * height * width, frames, channels)
        latest = self.temporal(tokens)[:, -1].reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        latest = F.interpolate(latest, size=(128, 128), mode="bilinear", align_corners=False)
        return self.reduce(latest)

    def forward(self, radar: torch.Tensor) -> DREAMBEVOutput:
        polar_feature = self.forward_polar_feature(radar)
        feature = self.decoder.encode_shared(self.polar_to_bev(polar_feature))
        return DREAMBEVOutput(
            logits=self.decoder.decode_shared(feature),
            feature=feature,
            polar_feature=polar_feature,
        )
