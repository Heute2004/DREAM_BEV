from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class BEVGeometry:
    """Frozen Radar-H metric lattice, with tensor axes ordered as [x, y]."""

    x_min_m: float = -10.0
    x_max_m: float = 10.0
    y_min_m: float = 0.0
    y_max_m: float = 23.4
    resolution_m: float = 0.1
    range_bins: int = 128
    azimuth_bins: int = 128
    sample_rate_hz: float = 6.25e6
    slope_hz_per_s: float = 9.994e12
    range_fft_size: int = 512

    @property
    def x_bins(self) -> int:
        return int(round((self.x_max_m - self.x_min_m) / self.resolution_m))

    @property
    def y_bins(self) -> int:
        return int(round((self.y_max_m - self.y_min_m) / self.resolution_m))

    @property
    def range_resolution_m(self) -> float:
        return 299_792_458.0 * self.sample_rate_hz / (2.0 * self.slope_hz_per_s * self.range_fft_size)

    @property
    def range_axis_m(self) -> Tensor:
        return torch.arange(self.range_bins, dtype=torch.float32) * self.range_resolution_m

    def coordinate_grid(self, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
        x = self.x_min_m + torch.arange(self.x_bins, device=device, dtype=dtype) * self.resolution_m
        y = self.y_min_m + torch.arange(self.y_bins, device=device, dtype=dtype) * self.resolution_m
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1)

    def metric_to_index(self, xy_m: Tensor) -> Tensor:
        if xy_m.shape[-1] != 2:
            raise ValueError("xy_m must end in two metric coordinates")
        x_index = torch.round((xy_m[..., 0] - self.x_min_m) / self.resolution_m).long()
        y_index = torch.round((xy_m[..., 1] - self.y_min_m) / self.resolution_m).long()
        return torch.stack((x_index.clamp(0, self.x_bins - 1), y_index.clamp(0, self.y_bins - 1)), dim=-1)

    def index_to_metric(self, indices: Tensor) -> Tensor:
        if indices.shape[-1] != 2:
            raise ValueError("indices must end in [x_index, y_index]")
        indices = indices.to(dtype=torch.float32)
        x = self.x_min_m + indices[..., 0] * self.resolution_m
        y = self.y_min_m + indices[..., 1] * self.resolution_m
        return torch.stack((x, y), dim=-1)


class PolarToBEV(nn.Module):
    """Differentiably sample native range-azimuth features on the metric BEV grid."""

    def __init__(self, geometry: BEVGeometry | None = None) -> None:
        super().__init__()
        self.geometry = geometry or BEVGeometry()
        self.register_buffer("sampling_grid", self._build_sampling_grid(), persistent=False)

    def _build_sampling_grid(self) -> Tensor:
        metric = self.geometry.coordinate_grid()
        x, y = metric.unbind(dim=-1)
        radial_distance = torch.sqrt(x.square() + y.square())
        sin_azimuth = torch.where(radial_distance > 1e-6, x / radial_distance, torch.zeros_like(radial_distance))
        range_index = radial_distance / self.geometry.range_resolution_m
        # FFT-shifted lambda/2 ULA bins are uniformly sampled in sin(theta).
        azimuth_index = (sin_azimuth + 1.0) * (self.geometry.azimuth_bins / 2.0)
        range_normalized = 2.0 * range_index / (self.geometry.range_bins - 1) - 1.0
        azimuth_normalized = 2.0 * azimuth_index / (self.geometry.azimuth_bins - 1) - 1.0
        return torch.stack((azimuth_normalized, range_normalized), dim=-1).unsqueeze(0)

    def forward(self, polar_feature: Tensor) -> Tensor:
        if polar_feature.ndim != 4 or polar_feature.shape[-2:] != (
            self.geometry.range_bins,
            self.geometry.azimuth_bins,
        ):
            raise ValueError(
                "polar_feature must be [batch, channels, 128 range bins, 128 azimuth bins]"
            )
        grid = self.sampling_grid.to(device=polar_feature.device, dtype=polar_feature.dtype)
        return functional.grid_sample(
            polar_feature,
            grid.expand(polar_feature.shape[0], -1, -1, -1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )



