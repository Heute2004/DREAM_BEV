from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from ..geometry import BEVGeometry


class PolarGeometryEmbedding(nn.Module):
    """Fixed polar coordinates available to frame-local reliability modules."""

    def __init__(self, geometry: BEVGeometry | None = None) -> None:
        super().__init__()
        geometry = geometry or BEVGeometry()
        ranges = geometry.range_axis_m
        range_norm = ranges / ranges[-1].clamp_min(1e-6)
        sin_azimuth = torch.linspace(-1.0, 1.0, geometry.azimuth_bins, dtype=torch.float32)
        cos_azimuth = torch.sqrt((1.0 - sin_azimuth.square()).clamp_min(0.0))
        features = torch.stack(
            (
                range_norm[:, None].expand(-1, geometry.azimuth_bins),
                sin_azimuth[None, :].expand(geometry.range_bins, -1),
                cos_azimuth[None, :].expand(geometry.range_bins, -1),
            ),
            dim=0,
        ).unsqueeze(0)
        self.register_buffer("features", features, persistent=False)

    @property
    def channels(self) -> int:
        return int(self.features.shape[1])

    def forward(self, reference: Tensor) -> Tensor:
        if reference.ndim != 4:
            raise ValueError("polar geometry reference must be [batch,channels,height,width]")
        geometry = self.features.to(device=reference.device, dtype=reference.dtype)
        if geometry.shape[-2:] != reference.shape[-2:]:
            geometry = functional.interpolate(geometry, size=reference.shape[-2:], mode="bilinear", align_corners=True)
        return geometry.expand(reference.shape[0], -1, -1, -1)


class ReliabilityConditionedDualMapGate(nn.Module):
    """Bounded, residual RA/RA-MTI allocation that preserves the two-map API."""

    def __init__(self, geometry: PolarGeometryEmbedding | None = None, hidden_channels: int = 8) -> None:
        super().__init__()
        self.geometry = geometry or PolarGeometryEmbedding()
        self.gate = nn.Sequential(
            nn.Conv2d(3 + self.geometry.channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        # Starts close to zero allocation strength; the gate can grow to, but never exceed, 0.5.
        self.raw_strength = nn.Parameter(torch.tensor(-4.0))

    @property
    def allocation_strength(self) -> Tensor:
        return 0.5 * torch.sigmoid(self.raw_strength)

    def forward(self, radar: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if radar.ndim != 5 or radar.shape[2] != 2:
            raise ValueError("radar must be [batch,time,2,range,azimuth]")
        batch, frames, _, height, width = radar.shape
        flat = radar.reshape(batch * frames, 2, height, width)
        ra, mti = flat[:, :1], flat[:, 1:]
        disagreement = (ra - mti).abs()
        allocation = torch.sigmoid(self.gate(torch.cat((ra, mti, disagreement, self.geometry(flat)), dim=1)))
        deviation = self.allocation_strength * (2.0 * allocation - 1.0)
        reweighted = torch.cat((ra * (1.0 + deviation), mti * (1.0 - deviation)), dim=1)
        evidence = {
            "allocation": allocation.reshape(batch, frames, 1, height, width),
            "disagreement": disagreement.reshape(batch, frames, 1, height, width),
        }
        return reweighted.reshape_as(radar), evidence


class ReliabilityConditionedTemporalInputModulation(nn.Module):
    """Frame-local quality modulation before a causal temporal mixer."""

    def __init__(
        self,
        feature_channels: int,
        geometry: PolarGeometryEmbedding | None = None,
        modulation_limit: float = 0.5,
    ) -> None:
        super().__init__()
        if feature_channels < 1:
            raise ValueError("feature_channels must be positive")
        if not 0.0 < modulation_limit <= 0.5:
            raise ValueError("modulation_limit must be in (0, 0.5]")
        self.feature_channels = feature_channels
        self.geometry = geometry or PolarGeometryEmbedding()
        self.modulation_limit = modulation_limit
        self.conditioner = nn.Conv2d(2 + self.geometry.channels, feature_channels * 2, kernel_size=1)
        nn.init.zeros_(self.conditioner.weight)
        nn.init.zeros_(self.conditioner.bias)

    def forward(self, feature: Tensor, evidence: dict[str, Tensor]) -> Tensor:
        if feature.ndim != 5:
            raise ValueError("feature must be [batch,time,channels,height,width]")
        batch, frames, channels, height, width = feature.shape
        if channels != self.feature_channels:
            raise ValueError("feature channels must match the configured modulation channels")
        try:
            allocation = evidence["allocation"]
            disagreement = evidence["disagreement"]
        except KeyError as error:
            raise ValueError("evidence must contain allocation and disagreement") from error
        expected_evidence = (batch, frames, 1, *allocation.shape[-2:])
        if allocation.shape != expected_evidence or disagreement.shape != expected_evidence:
            raise ValueError("evidence must be [batch,time,1,range,azimuth]")
        flat_feature = feature.reshape(batch * frames, channels, height, width)
        flat_allocation = allocation.reshape(batch * frames, 1, *allocation.shape[-2:])
        flat_disagreement = disagreement.reshape(batch * frames, 1, *disagreement.shape[-2:])
        if flat_allocation.shape[-2:] != (height, width):
            flat_allocation = functional.interpolate(flat_allocation, size=(height, width), mode="bilinear", align_corners=False)
            flat_disagreement = functional.interpolate(flat_disagreement, size=(height, width), mode="bilinear", align_corners=False)
        gamma, beta = self.conditioner(
            torch.cat((flat_allocation, flat_disagreement, self.geometry(flat_feature)), dim=1)
        ).chunk(2, dim=1)
        output = flat_feature * (1.0 + self.modulation_limit * torch.tanh(gamma))
        output = output + self.modulation_limit * torch.tanh(beta)
        return output.reshape(batch, frames, channels, height, width)


class GeometryConditionedBEVModulation(nn.Module):
    """Identity-initialized fixed-metric BEV FiLM modulation."""

    def __init__(self, feature_channels: int, geometry: BEVGeometry | None = None, modulation_limit: float = 0.5) -> None:
        super().__init__()
        if feature_channels < 1:
            raise ValueError("feature_channels must be positive")
        if not 0.0 < modulation_limit <= 0.5:
            raise ValueError("modulation_limit must be in (0, 0.5]")
        geometry = geometry or BEVGeometry()
        metric = geometry.coordinate_grid()
        x, y = metric.unbind(dim=-1)
        radius = torch.sqrt(x.square() + y.square())
        max_radius = math.sqrt(geometry.x_max_m**2 + geometry.y_max_m**2)
        sin_azimuth = torch.where(radius > 1e-6, x / radius, torch.zeros_like(radius))
        normalized_x = x / max(abs(geometry.x_min_m), abs(geometry.x_max_m))
        normalized_y = 2.0 * (y - geometry.y_min_m) / (geometry.y_max_m - geometry.y_min_m) - 1.0
        self.register_buffer(
            "geometry_features",
            torch.stack((radius / max_radius, sin_azimuth, normalized_x, normalized_y), dim=0).unsqueeze(0),
            persistent=False,
        )
        self.feature_channels = feature_channels
        self.modulation_limit = modulation_limit
        self.conditioner = nn.Conv2d(4, feature_channels * 2, kernel_size=1)
        nn.init.zeros_(self.conditioner.weight)
        nn.init.zeros_(self.conditioner.bias)

    def forward(self, feature: Tensor) -> Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.feature_channels:
            raise ValueError("feature must be [batch,configured_channels,height,width]")
        geometry = self.geometry_features.to(device=feature.device, dtype=feature.dtype)
        if geometry.shape[-2:] != feature.shape[-2:]:
            geometry = functional.interpolate(geometry, size=feature.shape[-2:], mode="bilinear", align_corners=True)
        gamma, beta = self.conditioner(geometry.expand(feature.shape[0], -1, -1, -1)).chunk(2, dim=1)
        return feature * (1.0 + self.modulation_limit * torch.tanh(gamma)) + self.modulation_limit * torch.tanh(beta)


RAGTLiteVariant = Literal["base", "rcdg", "rcdg_ctim", "full"]


class RAGTLiteAdapters(nn.Module):
    """Optional RAGT-Lite hooks shared by CPM and temporal-window backbones."""

    VALID_VARIANTS = ("base", "rcdg", "rcdg_ctim", "full")

    def __init__(self, feature_channels: int, variant: RAGTLiteVariant = "base") -> None:
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            choices = ", ".join(self.VALID_VARIANTS)
            raise ValueError(f"Unknown RAGT-Lite variant {variant!r}; choose one of: {choices}")
        self.variant = variant
        self.input_gate = ReliabilityConditionedDualMapGate() if variant != "base" else None
        self.temporal_modulation = (
            ReliabilityConditionedTemporalInputModulation(feature_channels) if variant in {"rcdg_ctim", "full"} else None
        )
        self.bev_modulation = GeometryConditionedBEVModulation(48) if variant == "full" else None

    def condition_radar(self, radar: Tensor) -> tuple[Tensor, dict[str, Tensor] | None]:
        if self.input_gate is None:
            return radar, None
        return self.input_gate(radar)

    def condition_temporal(self, feature: Tensor, evidence: dict[str, Tensor] | None) -> Tensor:
        if self.temporal_modulation is None:
            return feature
        if evidence is None:
            raise RuntimeError("CTIM requires RCDG evidence")
        return self.temporal_modulation(feature, evidence)

    def condition_bev(self, feature: Tensor) -> Tensor:
        if self.bev_modulation is None:
            return feature
        return self.bev_modulation(feature)



