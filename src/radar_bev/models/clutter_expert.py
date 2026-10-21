from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from ..expert_primitives import causal_previous_window, decode_peak_xy, gaussian_logit_delta, metric_bev_crop, metric_polar_crop
from ..geometry import BEVGeometry
from .backbones import DREAMBEV, extract_dream_bev_state
from .heterogeneous_experts import _ROIEncoder, _clamp_metric_xy, _positive_diagonal_covariance, _zero_linear


def _metric_grid(centers_xy_m: Tensor, span_m: float, output_size: int) -> Tensor:
    offset = torch.linspace(-span_m / 2.0, span_m / 2.0, output_size, dtype=centers_xy_m.dtype, device=centers_xy_m.device)
    offset_x, offset_y = torch.meshgrid(offset, offset, indexing="ij")
    return centers_xy_m[:, None, None, :] + torch.stack((offset_x, offset_y), dim=-1).unsqueeze(0)


def _weighted_geometric_median(grid: Tensor, weight: Tensor, iterations: int = 5) -> Tensor:
    location = (weight[..., None] * grid).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1e-6)[:, None]
    for _ in range(iterations):
        distance = (grid - location[:, None, None]).norm(dim=-1).clamp_min(1e-4)
        inverse = weight / distance
        location = (inverse[..., None] * grid).sum(dim=(1, 2)) / inverse.sum(dim=(1, 2)).clamp_min(1e-6)[:, None]
    return location


def _local_echo_candidates(radar_latest: Tensor, base_xy_m: Tensor, geometry: BEVGeometry) -> tuple[Tensor, Tensor]:
    """Return the exact eight local centres audited by R2-D1, plus scalar support features.

    The centres are non-learning transforms of the current RA/RA-MTI pair.  This
    intentionally prevents the selector from inventing a free coordinate: it can
    only choose the frozen base, the causal previous base, or one observed local
    robust centre.
    """
    locations: list[Tensor] = []
    summaries: list[Tensor] = []
    for span_m in (0.90, 1.20, 1.60, 2.00):
        crop = metric_polar_crop(radar_latest, base_xy_m, geometry, span_m=span_m, output_size=25)
        joint = 0.65 * crop[:, 0] + 0.35 * crop[:, 1]
        median = joint.flatten(start_dim=1).median(dim=1).values[:, None, None]
        excess = functional.softplus((joint - median) / 0.16)
        cap = torch.quantile(excess.flatten(start_dim=1), 0.90, dim=1)[:, None, None].clamp_min(1e-4)
        weight = excess.minimum(cap)
        grid = _metric_grid(base_xy_m, span_m, 25)
        centroid = (weight[..., None] * grid).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1e-6)[:, None]
        geometric_median = _weighted_geometric_median(grid, weight)
        locations.extend((_clamp_metric_xy(geometric_median, geometry), _clamp_metric_xy(centroid, geometry)))
        # Per-span evidence is duplicated over its two estimators.  It describes
        # peak separation and echo width without leaking any label-derived value.
        flat = joint.flatten(start_dim=1)
        peak_values = flat.topk(2, dim=1).values
        support = weight.sum(dim=(1, 2)) / float(25 * 25)
        dispersion = (weight * (grid - geometric_median[:, None, None]).square().sum(dim=-1)).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1e-6)
        stat = torch.stack((support, peak_values[:, 0] - peak_values[:, 1], dispersion.sqrt(), joint.std(dim=(1, 2), unbiased=False)), dim=1)
        summaries.extend((stat, stat))
    return torch.stack(locations, dim=1), torch.stack(summaries, dim=1)


class TECSEchoCandidateSelector(nn.Module):
    """Temporal Echo Candidate Selector for extended indoor human echoes.

    It uses a straight-through categorical selector.  Runtime proposals are
    always one physical candidate rather than an average between furniture and
    body echoes; gradients still train the candidate scores through the soft
    distribution.  The residual scale starts at exactly zero, hence insertion is
    an exact frozen-DREAM identity before training.
    """

    candidate_count = 10

    def __init__(self, geometry: BEVGeometry) -> None:
        super().__init__()
        self.geometry = geometry
        self.polar_encoder = _ROIEncoder(48, 20)
        self.raw_encoder = _ROIEncoder(6, 20)
        self.bev_encoder = _ROIEncoder(128, 16)
        self.type_embedding = nn.Embedding(self.candidate_count, 8)
        self.token_project = nn.Sequential(nn.Linear(20 + 20 + 16 + 8 + 11, 80), nn.SiLU(inplace=True), nn.Linear(80, 80), nn.SiLU(inplace=True))
        self.candidate_attention = nn.MultiheadAttention(80, num_heads=4, batch_first=True)
        self.token_norm = nn.LayerNorm(80)
        self.selector_head = nn.Linear(80, 1)
        self.state = nn.Sequential(nn.Linear(80 + 3, 72), nn.SiLU(inplace=True), nn.Linear(72, 64), nn.SiLU(inplace=True))
        self.scale_head = _zero_linear(nn.Linear(64, 1))
        self.acceptance_head = nn.Linear(64, 1)
        self.covariance_head = nn.Linear(64, 2)

    def _candidate_crops(self, feature: Tensor, candidates_xy_m: Tensor, cropper, span_m: float, output_size: int) -> Tensor:
        batch, count, _ = candidates_xy_m.shape
        repeated = feature.repeat_interleave(count, dim=0)
        crop = cropper(repeated, candidates_xy_m.reshape(batch * count, 2), self.geometry, span_m, output_size)
        return crop.reshape(batch, count, *crop.shape[1:])

    def _raw_history(self, radar: Tensor, candidates_xy_m: Tensor) -> Tensor:
        """Five-frame causal local RA/RA-MTI mean/std around each current candidate."""
        batch, frames = radar.shape[:2]
        history = radar[:, max(0, frames - 5):]
        count = candidates_xy_m.shape[1]
        source = history.reshape(-1, 2, 128, 128).repeat_interleave(count, dim=0)
        centres = candidates_xy_m[:, None].expand(-1, history.shape[1], -1, -1).reshape(-1, 2)
        crops = metric_polar_crop(source, centres, self.geometry, span_m=0.85, output_size=7)
        crops = crops.reshape(batch, history.shape[1], count, 2, 7, 7)
        current = crops[:, -1]
        mean = crops.mean(dim=1)
        std = crops.std(dim=1, unbiased=False)
        return torch.cat((current, mean, std), dim=2)

    def forward(
        self,
        polar_feature: Tensor,
        shared_feature: Tensor,
        radar: Tensor,
        base_logits: Tensor,
        base_xy_m: Tensor,
        previous_xy_m: Tensor,
    ) -> dict[str, Tensor]:
        local_xy_m, echo_summary = _local_echo_candidates(radar[:, -1], base_xy_m, self.geometry)
        candidates = torch.cat((base_xy_m[:, None], previous_xy_m[:, None], local_xy_m), dim=1)
        batch, count, _ = candidates.shape
        polar_crop = self._candidate_crops(polar_feature, candidates, metric_polar_crop, 0.85, 7)
        raw_crop = self._raw_history(radar, candidates)
        bev_crop = self._candidate_crops(shared_feature, candidates, metric_bev_crop, 1.4, 7)
        polar_embedding = self.polar_encoder(polar_crop.reshape(batch * count, *polar_crop.shape[2:])).reshape(batch, count, -1)
        raw_embedding = self.raw_encoder(raw_crop.reshape(batch * count, *raw_crop.shape[2:])).reshape(batch, count, -1)
        bev_embedding = self.bev_encoder(bev_crop.reshape(batch * count, *bev_crop.shape[2:])).reshape(batch, count, -1)
        type_ids = torch.arange(count, device=radar.device).expand(batch, -1)
        base_score = base_logits[:, 0].amax(dim=(1, 2), keepdim=False)[:, None, None].expand(-1, count, 1)
        displacement_base = candidates - base_xy_m[:, None]
        displacement_previous = candidates - previous_xy_m[:, None]
        latest = radar[:, -1]
        global_health = torch.stack((latest.mean(dim=(1, 2, 3)), latest.std(dim=(1, 2, 3), unbiased=False), (latest[:, 0] - latest[:, 1]).abs().mean(dim=(1, 2))), dim=1)
        per_candidate_stats = torch.zeros(batch, count, 4, dtype=radar.dtype, device=radar.device)
        per_candidate_stats[:, 2:] = echo_summary
        geometry_stats = torch.cat((
            candidates / torch.tensor((10.0, 23.4), device=radar.device, dtype=radar.dtype),
            displacement_base / 2.0,
            displacement_previous / 2.0,
            per_candidate_stats,
            base_score / 8.0,
        ), dim=-1)
        tokens = self.token_project(torch.cat((polar_embedding, raw_embedding, bev_embedding, self.type_embedding(type_ids), geometry_stats), dim=-1))
        attended, _ = self.candidate_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.token_norm(tokens + attended)
        selector_logits = self.selector_head(tokens).squeeze(-1)
        mixture = torch.softmax(selector_logits, dim=1)
        hard_index = mixture.argmax(dim=1)
        hard = functional.one_hot(hard_index, num_classes=count).to(dtype=mixture.dtype)
        straight_through = hard + mixture - mixture.detach()
        proposal_xy_m = (straight_through.unsqueeze(-1) * candidates).sum(dim=1)
        selected = (mixture.unsqueeze(-1) * tokens).sum(dim=1)
        state = self.state(torch.cat((selected, global_health), dim=1))
        strength = 3.5 * torch.tanh(self.scale_head(state))
        return {
            "delta_bev_logit": gaussian_logit_delta(base_xy_m, proposal_xy_m, strength, self.geometry, sigma_m=0.22),
            "acceptance_logit": self.acceptance_head(state),
            "covariance": _positive_diagonal_covariance(self.covariance_head(state)),
            "causal_state": state,
            "proposal_xy_m": proposal_xy_m,
            "candidate_xy_m": candidates,
            "candidate_selector_logits": selector_logits,
            "candidate_mixture": mixture,
            "selected_candidate_index": hard_index,
            "echo_summary": echo_summary,
        }


class DREAMClutterExpertBEV(nn.Module):
    """Frozen DREAM-BEV plus a TECS branch selector for indoor extended echoes."""

    def __init__(
        self,
        expert_kind: str = "tecs",
        logit_cap: float = 3.5,
        candidate_count: int = 10,
        force_acceptance: bool = False,
        acceptance_temperature: float = 1.0,
        acceptance_bias: float = 0.0,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if expert_kind != "tecs":
            raise ValueError("the clutter expert supports only expert_kind='tecs'")
        if candidate_count != TECSEchoCandidateSelector.candidate_count:
            raise ValueError("TECS uses its fixed audited candidate set of 10")
        self.geometry = BEVGeometry()
        self.expert_kind = expert_kind
        self.logit_cap = logit_cap
        self.force_acceptance = force_acceptance
        if acceptance_temperature <= 0.0:
            raise ValueError("acceptance_temperature must be positive")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        self.acceptance_temperature = float(acceptance_temperature)
        self.acceptance_bias = float(acceptance_bias)
        self.residual_scale = float(residual_scale)
        self.backbone = DREAMBEV()
        self.expert = TECSEchoCandidateSelector(self.geometry)
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "DREAMClutterExpertBEV":
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
        yield from (parameter for parameter in self.expert.parameters() if parameter.requires_grad)

    def expert_state_dict(self) -> dict[str, object]:
        return {
            "expert_kind": self.expert_kind,
            "logit_cap": self.logit_cap,
            "force_acceptance": self.force_acceptance,
            "acceptance_temperature": self.acceptance_temperature,
            "acceptance_bias": self.acceptance_bias,
            "residual_scale": self.residual_scale,
            "expert": self.expert.state_dict(),
        }

    def _base_observation(self, radar: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad():
            polar_feature = self.backbone.forward_polar_feature(radar)
            shared_feature = self.backbone.shared_from_polar_feature(polar_feature)
            base_logits = self.backbone.decoder.decode_shared(shared_feature)
            previous_polar = self.backbone.forward_polar_feature(causal_previous_window(radar))
            previous_logits = self.backbone.decoder.decode_shared(self.backbone.shared_from_polar_feature(previous_polar))
            base_xy_m, _ = decode_peak_xy(base_logits, self.geometry)
            previous_xy_m, _ = decode_peak_xy(previous_logits, self.geometry)
        return polar_feature, shared_feature, base_logits, torch.cat((base_xy_m, previous_xy_m), dim=1)

    def forward(self, radar: Tensor) -> dict[str, Tensor]:
        self.backbone.eval()
        polar_feature, shared_feature, base_logits, peaks = self._base_observation(radar)
        base_xy_m, previous_xy_m = peaks[:, :2], peaks[:, 2:]
        expert_output = self.expert(polar_feature, shared_feature, radar, base_logits, base_xy_m, previous_xy_m)
        delta = (self.residual_scale * expert_output["delta_bev_logit"]).clamp(-self.logit_cap, self.logit_cap)
        raw_acceptance_logit = expert_output["acceptance_logit"].float()
        acceptance = torch.ones_like(raw_acceptance_logit) if self.force_acceptance else torch.sigmoid(self.acceptance_temperature * raw_acceptance_logit + self.acceptance_bias)
        return {
            "logits": base_logits + acceptance[:, :, None, None] * delta,
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



