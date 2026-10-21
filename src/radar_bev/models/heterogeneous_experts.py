from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from ..expert_primitives import (
    causal_previous_window,
    decode_peak_xy,
    gaussian_logit_delta,
    local_maximum_candidates,
    metric_bev_crop,
    metric_polar_crop,
)
from ..geometry import BEVGeometry
from .backbones import DREAMBEV, extract_dream_bev_state


ExpertKind = Literal["clor", "dcmd", "patei", "srre"]
EXPERT_KINDS: tuple[ExpertKind, ...] = ("clor", "dcmd", "patei", "srre")


def _zero_linear(layer: nn.Linear) -> nn.Linear:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def _positive_diagonal_covariance(raw_scale: Tensor) -> Tensor:
    sigma = functional.softplus(raw_scale) + 0.05
    return torch.diag_embed(sigma.square())


def _clamp_metric_xy(xy_m: Tensor, geometry: BEVGeometry) -> Tensor:
    return torch.stack(
        (
            xy_m[:, 0].clamp(geometry.x_min_m, geometry.x_max_m - geometry.resolution_m),
            xy_m[:, 1].clamp(geometry.y_min_m, geometry.y_max_m - geometry.resolution_m),
        ),
        dim=-1,
    )


def _nearest_bev_score(logits: Tensor, xy_m: Tensor, geometry: BEVGeometry) -> Tensor:
    if xy_m.ndim == 2:
        indices = geometry.metric_to_index(xy_m)
        batch = torch.arange(logits.shape[0], device=logits.device)
        return logits[batch, 0, indices[:, 0], indices[:, 1]]
    if xy_m.ndim != 3 or xy_m.shape[0] != logits.shape[0] or xy_m.shape[2] != 2:
        raise ValueError("xy_m must be [batch,2] or [batch,candidates,2]")
    indices = geometry.metric_to_index(xy_m)
    batch = torch.arange(logits.shape[0], device=logits.device)[:, None].expand(-1, xy_m.shape[1])
    return logits[batch, 0, indices[..., 0], indices[..., 1]]


class _ROIEncoder(nn.Module):
    def __init__(self, channels: int, output_channels: int) -> None:
        super().__init__()
        groups = 8 if output_channels % 8 == 0 else 1
        self.layers = nn.Sequential(
            nn.Conv2d(channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, groups=output_channels, bias=False),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, crop: Tensor) -> Tensor:
        return self.layers(crop).flatten(start_dim=1)


class _CausalRadarState(nn.Module):
    """Small causal state from the current RA/RA-MTI observation window."""

    def __init__(self, state_channels: int = 64) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(4, 32), nn.SiLU(inplace=True), nn.Linear(32, state_channels))
        self.gru = nn.GRU(state_channels, state_channels, batch_first=True)

    def forward(self, radar: Tensor) -> Tensor:
        if radar.ndim != 5 or radar.shape[2] != 2:
            raise ValueError("radar must be [batch,frames,2,128,128]")
        mean = radar.mean(dim=(-2, -1))
        std = radar.std(dim=(-2, -1), unbiased=False)
        sequence, _ = self.gru(self.project(torch.cat((mean, std), dim=-1)))
        return sequence[:, -1]


class _BaseHeterogeneousExpert(nn.Module):
    def __init__(self, geometry: BEVGeometry, offset_bound_m: float, sigma_m: float) -> None:
        super().__init__()
        self.geometry = geometry
        self.offset_bound_m = offset_bound_m
        self.sigma_m = sigma_m

    def _render_delta(self, base_xy_m: Tensor, proposal_xy_m: Tensor, strength: Tensor) -> Tensor:
        return gaussian_logit_delta(base_xy_m, proposal_xy_m, strength, self.geometry, self.sigma_m)


class CLOREXpert(_BaseHeterogeneousExpert):
    """Causal Local Offset Refiner for near-saturated indoor LOS localization."""

    def __init__(self, geometry: BEVGeometry) -> None:
        super().__init__(geometry, offset_bound_m=0.45, sigma_m=0.22)
        self.local_encoder = _ROIEncoder(128, 32)
        self.radar_state = _CausalRadarState(64)
        self.fuse = nn.Sequential(nn.Linear(32 + 64 + 4, 64), nn.SiLU(inplace=True), nn.Linear(64, 64), nn.SiLU(inplace=True))
        self.offset_head = nn.Linear(64, 2)
        self.scale_head = _zero_linear(nn.Linear(64, 1))
        self.acceptance_head = nn.Linear(64, 1)
        self.covariance_head = nn.Linear(64, 2)

    def forward(
        self,
        polar_feature: Tensor,
        shared_feature: Tensor,
        radar: Tensor,
        base_logits: Tensor,
        base_xy_m: Tensor,
        previous_xy_m: Tensor,
    ) -> dict[str, Tensor]:
        local = self.local_encoder(metric_bev_crop(shared_feature, base_xy_m, self.geometry, span_m=2.0, output_size=7))
        state = self.radar_state(radar)
        innovation = base_xy_m - previous_xy_m
        fused = self.fuse(torch.cat((local, state, base_xy_m / 10.0, innovation), dim=1))
        proposal = _clamp_metric_xy(base_xy_m + self.offset_bound_m * torch.tanh(self.offset_head(fused)), self.geometry)
        strength = torch.tanh(self.scale_head(fused))
        return {
            "delta_bev_logit": self._render_delta(base_xy_m, proposal, strength),
            "acceptance_logit": self.acceptance_head(fused),
            "covariance": _positive_diagonal_covariance(self.covariance_head(fused)),
            "causal_state": state,
            "proposal_xy_m": proposal,
        }


class DCMDEXpert(_BaseHeterogeneousExpert):
    """Dual-domain candidate mixture for clutter and multipath branch disambiguation."""

    def __init__(self, geometry: BEVGeometry, candidate_count: int = 4) -> None:
        super().__init__(geometry, offset_bound_m=0.35, sigma_m=0.28)
        if candidate_count < 3:
            raise ValueError("DCMD candidate_count must include base, previous, and one local maximum")
        self.candidate_count = candidate_count
        self.bev_encoder = _ROIEncoder(128, 24)
        self.polar_encoder = _ROIEncoder(48, 24)
        self.token_project = nn.Sequential(nn.Linear(24 + 24 + 3, 64), nn.SiLU(inplace=True), nn.Linear(64, 64))
        self.set_attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        self.token_norm = nn.LayerNorm(64)
        self.mixture_head = nn.Linear(64, 1)
        self.offset_head = nn.Linear(64, 2)
        self.scale_head = _zero_linear(nn.Linear(64, 1))
        self.acceptance_head = nn.Linear(64, 1)
        self.covariance_head = nn.Linear(64, 2)

    def _candidate_crops(self, feature: Tensor, candidates_xy_m: Tensor, cropper, span_m: float, output_size: int) -> Tensor:
        batch, count, _ = candidates_xy_m.shape
        repeated = feature.repeat_interleave(count, dim=0)
        crop = cropper(repeated, candidates_xy_m.reshape(batch * count, 2), self.geometry, span_m, output_size)
        return crop.reshape(batch, count, *crop.shape[1:])

    def forward(
        self,
        polar_feature: Tensor,
        shared_feature: Tensor,
        radar: Tensor,
        base_logits: Tensor,
        base_xy_m: Tensor,
        previous_xy_m: Tensor,
    ) -> dict[str, Tensor]:
        maxima_xy_m, _ = local_maximum_candidates(base_logits, self.candidate_count - 2, self.geometry)
        candidates = torch.cat((base_xy_m.unsqueeze(1), previous_xy_m.unsqueeze(1), maxima_xy_m), dim=1)
        batch, count, _ = candidates.shape
        bev_crops = self._candidate_crops(shared_feature, candidates, metric_bev_crop, span_m=1.6, output_size=7)
        polar_crops = self._candidate_crops(polar_feature, candidates, metric_polar_crop, span_m=1.6, output_size=9)
        bev_embedding = self.bev_encoder(bev_crops.reshape(batch * count, *bev_crops.shape[2:])).reshape(batch, count, -1)
        polar_embedding = self.polar_encoder(polar_crops.reshape(batch * count, *polar_crops.shape[2:])).reshape(batch, count, -1)
        score = _nearest_bev_score(base_logits, candidates, self.geometry).unsqueeze(-1)
        distance_to_prior = (candidates - previous_xy_m[:, None]).norm(dim=-1, keepdim=True)
        raw_crops = self._candidate_crops(radar[:, -1], candidates, metric_polar_crop, span_m=1.2, output_size=7)
        agreement = -(raw_crops[:, :, 0] - raw_crops[:, :, 1]).abs().mean(dim=(-2, -1), keepdim=False).unsqueeze(-1)
        tokens = self.token_project(torch.cat((bev_embedding, polar_embedding, score / 8.0, distance_to_prior, agreement), dim=-1))
        attended, _ = self.set_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.token_norm(tokens + attended)
        mixture = torch.softmax(self.mixture_head(tokens).squeeze(-1), dim=1)
        candidate_proposals = _clamp_metric_xy(
            (candidates + self.offset_bound_m * torch.tanh(self.offset_head(tokens))).reshape(batch * count, 2),
            self.geometry,
        ).reshape(batch, count, 2)
        proposal = (mixture.unsqueeze(-1) * candidate_proposals).sum(dim=1)
        state = tokens.mean(dim=1)
        strength = torch.tanh(self.scale_head(state))
        return {
            "delta_bev_logit": self._render_delta(base_xy_m, proposal, strength),
            "acceptance_logit": self.acceptance_head(state),
            "covariance": _positive_diagonal_covariance(self.covariance_head(state)),
            "causal_state": state,
            "proposal_xy_m": proposal,
            "candidate_xy_m": candidates,
            "candidate_mixture": mixture,
        }


class PATEIExpert(_BaseHeterogeneousExpert):
    """Path-aligned causal non-coherent RA/RA-MTI evidence integration."""

    def __init__(self, geometry: BEVGeometry) -> None:
        super().__init__(geometry, offset_bound_m=0.75, sigma_m=0.42)
        self.tube_encoder = _ROIEncoder(2, 32)
        self.temporal_gru = nn.GRU(32, 64, batch_first=True)
        self.dropout = nn.Dropout(p=0.10)
        self.fuse = nn.Sequential(nn.Linear(64 + 4, 64), nn.SiLU(inplace=True), nn.Linear(64, 64), nn.SiLU(inplace=True))
        self.offset_head = nn.Linear(64, 2)
        self.scale_head = _zero_linear(nn.Linear(64, 1))
        self.acceptance_head = nn.Linear(64, 1)
        self.covariance_head = nn.Linear(64, 2)

    def forward(
        self,
        polar_feature: Tensor,
        shared_feature: Tensor,
        radar: Tensor,
        base_logits: Tensor,
        base_xy_m: Tensor,
        previous_xy_m: Tensor,
    ) -> dict[str, Tensor]:
        batch, frames = radar.shape[:2]
        velocity = base_xy_m - previous_xy_m
        steps_back = torch.arange(frames - 1, -1, -1, device=radar.device, dtype=radar.dtype).view(1, frames, 1)
        aligned_centers = _clamp_metric_xy(
            (base_xy_m[:, None, :] - steps_back * velocity[:, None, :]).reshape(batch * frames, 2), self.geometry
        ).reshape(batch, frames, 2)
        local_tube = metric_polar_crop(
            radar.reshape(batch * frames, *radar.shape[2:]),
            aligned_centers.reshape(batch * frames, 2),
            self.geometry,
            span_m=1.8,
            output_size=9,
        )
        embedding = self.tube_encoder(local_tube).reshape(batch, frames, -1)
        sequence, _ = self.temporal_gru(self.dropout(embedding))
        state = sequence[:, -1]
        fused = self.fuse(torch.cat((state, base_xy_m / 10.0, velocity), dim=1))
        proposal = _clamp_metric_xy(base_xy_m + self.offset_bound_m * torch.tanh(self.offset_head(fused)), self.geometry)
        strength = torch.tanh(self.scale_head(fused))
        return {
            "delta_bev_logit": self._render_delta(base_xy_m, proposal, strength),
            "acceptance_logit": self.acceptance_head(fused),
            "covariance": _positive_diagonal_covariance(self.covariance_head(fused)),
            "causal_state": state,
            "proposal_xy_m": proposal,
            "aligned_centers_xy_m": aligned_centers,
        }


class SRREExpert(_BaseHeterogeneousExpert):
    """Selective recovery branch separated from Radar-Health acceptance control."""

    def __init__(self, geometry: BEVGeometry) -> None:
        super().__init__(geometry, offset_bound_m=0.60, sigma_m=0.38)
        self.local_encoder = _ROIEncoder(128, 32)
        self.radar_state = _CausalRadarState(64)
        self.recovery = nn.Sequential(nn.Linear(32 + 64, 64), nn.SiLU(inplace=True), nn.Linear(64, 64), nn.SiLU(inplace=True))
        self.offset_head = nn.Linear(64, 2)
        self.scale_head = _zero_linear(nn.Linear(64, 1))
        self.health_head = nn.Sequential(nn.Linear(8, 32), nn.SiLU(inplace=True), nn.Linear(32, 32), nn.SiLU(inplace=True))
        self.acceptance_head = nn.Linear(32, 1)
        self.covariance_head = nn.Linear(32, 2)

    def _health(self, radar: Tensor, base_logits: Tensor, base_xy_m: Tensor, previous_xy_m: Tensor) -> Tensor:
        scores = base_logits[:, 0].flatten(start_dim=1).topk(2, dim=1).values
        probability = base_logits.sigmoid().flatten(start_dim=1)
        entropy = -(probability.clamp(1e-5, 1.0 - 1e-5) * probability.clamp(1e-5, 1.0 - 1e-5).log()
                    + (1.0 - probability).clamp(1e-5, 1.0 - 1e-5) * (1.0 - probability).clamp(1e-5, 1.0 - 1e-5).log()).mean(dim=1)
        crop = metric_polar_crop(radar[:, -1], base_xy_m, self.geometry, span_m=1.2, output_size=9)
        agreement = -(crop[:, 0] - crop[:, 1]).abs().mean(dim=(-2, -1))
        latest = radar[:, -1]
        innovation = (base_xy_m - previous_xy_m).norm(dim=-1)
        return torch.stack(
            (
                probability.max(dim=1).values,
                scores[:, 0] - scores[:, 1],
                entropy,
                innovation,
                agreement,
                latest.mean(dim=(1, 2, 3)),
                latest.std(dim=(1, 2, 3), unbiased=False),
                (latest[:, 0] - latest[:, 1]).abs().mean(dim=(1, 2)),
            ),
            dim=1,
        )

    def forward(
        self,
        polar_feature: Tensor,
        shared_feature: Tensor,
        radar: Tensor,
        base_logits: Tensor,
        base_xy_m: Tensor,
        previous_xy_m: Tensor,
    ) -> dict[str, Tensor]:
        local = self.local_encoder(metric_bev_crop(shared_feature, base_xy_m, self.geometry, span_m=2.2, output_size=7))
        state = self.radar_state(radar)
        recovery = self.recovery(torch.cat((local, state), dim=1))
        proposal = _clamp_metric_xy(base_xy_m + self.offset_bound_m * torch.tanh(self.offset_head(recovery)), self.geometry)
        strength = torch.tanh(self.scale_head(recovery))
        health = self.health_head(self._health(radar, base_logits, base_xy_m, previous_xy_m))
        return {
            "delta_bev_logit": self._render_delta(base_xy_m, proposal, strength),
            "acceptance_logit": self.acceptance_head(health),
            "covariance": _positive_diagonal_covariance(self.covariance_head(health)),
            "causal_state": state,
            "proposal_xy_m": proposal,
            "radar_health": health,
        }


class DREAMHeterogeneousExpertBEV(nn.Module):
    """Frozen DREAM-BEV with a route-specific bounded-logit heterogeneous expert."""

    def __init__(self, expert_kind: ExpertKind, logit_cap: float = 3.0, candidate_count: int = 4) -> None:
        super().__init__()
        if expert_kind not in EXPERT_KINDS:
            raise ValueError(f"unsupported expert_kind={expert_kind!r}; expected one of {EXPERT_KINDS}")
        if logit_cap <= 0.0:
            raise ValueError("logit_cap must be positive")
        self.geometry = BEVGeometry()
        self.expert_kind = expert_kind
        self.logit_cap = logit_cap
        self.backbone = DREAMBEV()
        if expert_kind == "clor":
            self.expert: nn.Module = CLOREXpert(self.geometry)
        elif expert_kind == "dcmd":
            self.expert = DCMDEXpert(self.geometry, candidate_count=candidate_count)
        elif expert_kind == "patei":
            self.expert = PATEIExpert(self.geometry)
        else:
            self.expert = SRREExpert(self.geometry)
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "DREAMHeterogeneousExpertBEV":
        super().train(mode)
        self.backbone.eval()
        return self

    def load_base_checkpoint(self, checkpoint_path: str | Path) -> str:
        path = Path(checkpoint_path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.backbone.load_state_dict(extract_dream_bev_state(payload), strict=True)
        self._freeze_backbone()
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def expert_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.expert.parameters()

    def expert_state_dict(self) -> dict[str, object]:
        return {"expert_kind": self.expert_kind, "logit_cap": self.logit_cap, "expert": self.expert.state_dict()}

    def _base_observation(self, radar: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            polar_feature = self.backbone.forward_polar_feature(radar)
            shared_feature = self.backbone.shared_from_polar_feature(polar_feature)
            base_logits = self.backbone.decoder.decode_shared(shared_feature)
            previous_polar = self.backbone.forward_polar_feature(causal_previous_window(radar))
            previous_shared = self.backbone.shared_from_polar_feature(previous_polar)
            previous_logits = self.backbone.decoder.decode_shared(previous_shared)
            base_xy_m, _ = decode_peak_xy(base_logits, self.geometry)
            previous_xy_m, _ = decode_peak_xy(previous_logits, self.geometry)
        return polar_feature, shared_feature, base_logits, torch.cat((base_xy_m, previous_xy_m), dim=1)

    def forward(self, radar: Tensor) -> dict[str, Tensor]:
        self.backbone.eval()
        polar_feature, shared_feature, base_logits, peaks = self._base_observation(radar)
        base_xy_m, previous_xy_m = peaks[:, :2], peaks[:, 2:]
        expert_output = self.expert(polar_feature, shared_feature, radar, base_logits, base_xy_m, previous_xy_m)
        delta = expert_output["delta_bev_logit"].clamp(-self.logit_cap, self.logit_cap)
        acceptance = torch.sigmoid(expert_output["acceptance_logit"])
        logits = base_logits + acceptance[:, :, None, None] * delta
        return {
            "logits": logits,
            "base_logits": base_logits,
            "delta_bev_logit": delta,
            "acceptance": acceptance,
            "acceptance_logit": expert_output["acceptance_logit"],
            "covariance": expert_output["covariance"],
            "causal_state": expert_output["causal_state"],
            "proposal_xy_m": expert_output["proposal_xy_m"],
            "base_peak_xy_m": base_xy_m,
            "previous_peak_xy_m": previous_xy_m,
            "shared_feature": shared_feature,
            **{key: value for key, value in expert_output.items() if key not in {"delta_bev_logit", "acceptance_logit", "covariance", "causal_state", "proposal_xy_m"}},
        }



