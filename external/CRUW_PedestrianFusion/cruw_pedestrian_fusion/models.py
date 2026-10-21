"""Compact causal RA models for the CRUW pedestrian-only protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.layers(value))


@dataclass(frozen=True)
class RadarOutput:
    logits: torch.Tensor
    feature: torch.Tensor


class DREAMBEVRA(nn.Module):
    """DREAM-BEV public RA adaptation with dual-map causal polar-state encoding.

    The external protocol preserves DREAM-BEV's causal two-channel radar contract,
    but emits a native RA confidence map because CRUW exposes polar annotations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(2, 32, stride=2),
            ResidualBlock(32),
            ConvNormAct(32, 64, stride=2),
            ResidualBlock(64),
        )
        self.temporal = nn.GRU(64, 64, batch_first=True)
        self.decoder = nn.Sequential(
            ConvNormAct(64, 48),
            nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, radar: torch.Tensor) -> RadarOutput:
        if radar.ndim != 5 or radar.shape[2:] != (2, 128, 128):
            raise ValueError(f"expected [B,T,2,128,128], got {tuple(radar.shape)}")
        batch_size, steps = radar.shape[:2]
        feature = self.stem(radar.reshape(batch_size * steps, 2, 128, 128))
        _, channels, height, width = feature.shape
        tokens = feature.reshape(batch_size, steps, channels, height, width)
        tokens = tokens.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, steps, channels)
        latest, _ = self.temporal(tokens)
        latest = latest[:, -1].reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        return RadarOutput(logits=self.decoder(latest), feature=latest)


class _CausalMambaTokenMixer(nn.Module):
    """The causal token mixer used by the deployed DREAM-BEV radar encoder."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "dream_bev_mamba_ra requires mamba_ssm; use the server rvmg environment."
            )
        self.mixer = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(tokens + self.mixer(tokens))


class DREAMBEVMambaRA(nn.Module):
    """DREAM-BEV's MRODNet causal core with a CRUW-native RA output head.

    CRUW exposes range-azimuth labels rather than the local BEV grid. This adapter
    preserves the deployed spatial encoder and past-only Mamba temporal mixer, then
    replaces only the task-specific BEV decoder with a one-class RA confidence head.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(2, 64, stride=2),
            ConvNormAct(64, 64),
        )
        self.temporal = _CausalMambaTokenMixer(64)
        self.reduce = nn.Sequential(
            ConvNormAct(64, 48),
            ResidualBlock(48),
        )
        self.ra_head = nn.Sequential(
            ConvNormAct(48, 48),
            ResidualBlock(48),
            nn.Conv2d(48, 1, 1),
        )

    def forward(self, radar: torch.Tensor) -> RadarOutput:
        if radar.ndim != 5 or radar.shape[2:] != (2, 128, 128):
            raise ValueError(f"expected [B,T,2,128,128], got {tuple(radar.shape)}")
        batch_size, steps = radar.shape[:2]
        features = self.stem(radar.reshape(batch_size * steps, 2, 128, 128))
        _, channels, height, width = features.shape
        tokens = features.reshape(batch_size, steps, channels, height, width)
        tokens = tokens.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, steps, channels)
        latest = self.temporal(tokens)[:, -1].reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        latest = F.interpolate(latest, size=(128, 128), mode="bilinear", align_corners=False)
        polar_feature = self.reduce(latest)
        return RadarOutput(logits=self.ra_head(polar_feature), feature=polar_feature)


# Keep old checkpoints and scripts loadable while using the paper-facing name for
# all newly trained public-protocol runs.
RadarAnchor = DREAMBEVRA


class _RASequenceBase(nn.Module):
    """Shared RA encoder/decoder used by same-contract external baselines."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(2, 32, stride=2),
            ResidualBlock(32),
            ConvNormAct(32, 64, stride=2),
            ResidualBlock(64),
        )
        self.decoder = nn.Sequential(
            ConvNormAct(64, 48),
            nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def encode_sequence(self, radar: torch.Tensor) -> torch.Tensor:
        if radar.ndim != 5 or radar.shape[2:] != (2, 128, 128):
            raise ValueError(f"expected [B,T,2,128,128], got {tuple(radar.shape)}")
        batch_size, steps = radar.shape[:2]
        features = self.stem(radar.reshape(batch_size * steps, 2, 128, 128))
        _, channels, height, width = features.shape
        return features.reshape(batch_size, steps, channels, height, width)

    def decode(self, feature: torch.Tensor) -> RadarOutput:
        return RadarOutput(logits=self.decoder(feature), feature=feature)


class ConvGRURA(_RASequenceBase):
    """Causal ConvGRU adaptation under the shared CRUW RA-map contract."""

    def __init__(self) -> None:
        super().__init__()
        self.gates = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.candidate = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(8, 64)
        self.fuse = nn.Sequential(ResidualBlock(64), ConvNormAct(64, 64))

    def forward(self, radar: torch.Tensor) -> RadarOutput:
        sequence = self.encode_sequence(radar)
        hidden = torch.zeros_like(sequence[:, 0])
        for step in range(sequence.shape[1]):
            pair = torch.cat((sequence[:, step], hidden), dim=1)
            reset, update = self.gates(pair).chunk(2, dim=1)
            reset, update = torch.sigmoid(reset), torch.sigmoid(update)
            candidate = torch.tanh(self.norm(self.candidate(torch.cat((sequence[:, step], reset * hidden), dim=1))))
            hidden = (1.0 - update) * hidden + update * candidate
        return self.decode(self.fuse(hidden))


class RadarFormerLiteRA(_RASequenceBase):
    """Causal temporal self-attention adaptation with CRUW-compatible RA output."""

    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.fuse = nn.Sequential(ResidualBlock(64), ConvNormAct(64, 64))

    def forward(self, radar: torch.Tensor) -> RadarOutput:
        sequence = self.encode_sequence(radar)
        batch_size, steps, channels, height, width = sequence.shape
        tokens = sequence.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, steps, channels)
        mask = torch.triu(torch.ones(steps, steps, device=radar.device, dtype=torch.bool), diagonal=1)
        attended, _ = self.attention(tokens, tokens, tokens, attn_mask=mask, need_weights=False)
        latest = self.norm(tokens[:, -1] + attended[:, -1])
        feature = latest.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        return self.decode(self.fuse(feature))


class RWKVRA(_RASequenceBase):
    """Causal weighted-key-value recurrence adapted to the shared RA interface."""

    def __init__(self) -> None:
        super().__init__()
        self.key = nn.Conv2d(64, 64, kernel_size=1)
        self.value = nn.Conv2d(64, 64, kernel_size=1)
        self.receptance = nn.Conv2d(64, 64, kernel_size=1)
        self.decay_logit = nn.Parameter(torch.full((1, 64, 1, 1), -1.0))
        self.fuse = nn.Sequential(ResidualBlock(64), ConvNormAct(64, 64))

    def forward(self, radar: torch.Tensor) -> RadarOutput:
        sequence = self.encode_sequence(radar)
        numerator = denominator = None
        decay = torch.sigmoid(self.decay_logit)
        output = sequence[:, 0]
        for step in range(sequence.shape[1]):
            value = sequence[:, step]
            key = self.key(value).clamp(-6.0, 6.0)
            weighted_key = torch.exp(key)
            weighted_value = weighted_key * self.value(value)
            if numerator is None:
                numerator, denominator = weighted_value, weighted_key
            else:
                numerator = decay * numerator + weighted_value
                denominator = decay * denominator + weighted_key
            output = torch.sigmoid(self.receptance(value)) * numerator / denominator.clamp_min(1e-4)
        return self.decode(self.fuse(output))


RadarModelName = Literal[
    "dream_bev", "dream_bev_mamba_ra", "convgru", "radarformer_lite", "rwkv_bev",
    "official_rodnet_cdc", "official_rodnet_hg", "official_trodnet",
]


def build_radar_model(name: RadarModelName) -> nn.Module:
    """Build a causal RA model for the fixed public 8-frame input/output contract."""

    models: dict[str, type[nn.Module]] = {
        "dream_bev": DREAMBEVRA,
        "dream_bev_mamba_ra": DREAMBEVMambaRA,
        "convgru": ConvGRURA,
        "radarformer_lite": RadarFormerLiteRA,
        "rwkv_bev": RWKVRA,
    }
    if name == "official_rodnet_cdc":
        from .official_models import OfficialRODNetCDC

        return OfficialRODNetCDC()
    if name == "official_rodnet_hg":
        from .official_models import OfficialRODNetHG

        return OfficialRODNetHG()
    if name == "official_trodnet":
        from .official_models import OfficialTRODNet

        return OfficialTRODNet()
    try:
        return models[name]()
    except KeyError as error:
        raise ValueError(f"unsupported radar model: {name}") from error


class RadarClutterExpert(nn.Module):
    """Bounded residual RA expert initialized as an exact radar identity map."""

    def __init__(self, cap: float = 1.5) -> None:
        super().__init__()
        self.cap = float(cap)
        self.encoder = nn.Sequential(
            ConvNormAct(3, 24),
            ResidualBlock(24),
            ConvNormAct(24, 24),
            ResidualBlock(24),
        )
        self.delta_head = nn.Conv2d(24, 1, 1)
        self.gate_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(24, 1))
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)

    def forward(self, radar: torch.Tensor, base_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = radar[:, -1]
        hidden = self.encoder(torch.cat((current, base_logits), dim=1))
        residual = self.cap * torch.tanh(self.delta_head(hidden))
        gate = self.gate_head(hidden)
        acceptance = torch.sigmoid(gate).view(-1, 1, 1, 1)
        return base_logits + acceptance * residual, gate


def gaussian_focal_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = -target * F.logsigmoid(logits)
    negative = -(1.0 - target).pow(4.0) * F.logsigmoid(-logits)
    return (positive + negative).mean()


@torch.no_grad()
def initial_identity_error(expert: RadarClutterExpert, radar: torch.Tensor, base_logits: torch.Tensor) -> float:
    """Measure the expert identity on one already-computed frozen-anchor map."""

    refined, _ = expert(radar, base_logits)
    return float((refined - base_logits).abs().max())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
