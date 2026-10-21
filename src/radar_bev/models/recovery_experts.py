from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from ..expert_primitives import causal_previous_window, decode_peak_xy, gaussian_logit_delta, metric_bev_crop, metric_polar_crop
from ..geometry import BEVGeometry
from .backbones import DREAMBEV, extract_dream_bev_state
from .heterogeneous_experts import _CausalRadarState, _ROIEncoder, _clamp_metric_xy, _positive_diagonal_covariance, _zero_linear


RecoveryExpertKind = Literal["cucre", "ltge", "gg_srre"]
RECOVERY_EXPERT_KINDS: tuple[RecoveryExpertKind, ...] = ("cucre", "ltge", "gg_srre")


def _zero_conv(layer: nn.Conv2d) -> nn.Conv2d:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def _logit_summary(logits: Tensor) -> Tensor:
    probability = logits.sigmoid().flatten(start_dim=1)
    top = logits[:, 0].flatten(start_dim=1).topk(2, dim=1).values
    entropy = -(
        probability.clamp(1e-5, 1.0 - 1e-5) * probability.clamp(1e-5, 1.0 - 1e-5).log()
        + (1.0 - probability).clamp(1e-5, 1.0 - 1e-5) * (1.0 - probability).clamp(1e-5, 1.0 - 1e-5).log()
    ).mean(dim=1)
    return torch.stack((probability.max(dim=1).values, top[:, 0] - top[:, 1], entropy, probability.std(dim=1)), dim=1)


def _polar_indices_to_xy(range_index: Tensor, azimuth_index: Tensor, geometry: BEVGeometry) -> Tensor:
    range_m = range_index.to(dtype=torch.float32) * geometry.range_resolution_m
    sin_azimuth = azimuth_index.to(dtype=torch.float32) / (geometry.azimuth_bins / 2.0) - 1.0
    x_m = range_m * sin_azimuth.clamp(-1.0, 1.0)
    y_m = torch.sqrt((range_m.square() - x_m.square()).clamp_min(0.0))
    return _clamp_metric_xy(torch.stack((x_m, y_m), dim=-1).reshape(-1, 2), geometry).reshape(*range_index.shape, 2)


def _low_threshold_polar_candidates(radar: Tensor, count: int, geometry: BEVGeometry) -> tuple[Tensor, Tensor]:
    """Retain local RA/RA-MTI maxima without a hard detection threshold."""
    if radar.ndim != 5 or radar.shape[2:] != (2, 128, 128):
        raise ValueError("radar must be [batch,frames,2,128,128]")
    if count < 1:
        raise ValueError("candidate count must be positive")
    batch, frames = radar.shape[:2]
    energy = radar.mean(dim=2).reshape(batch * frames, 1, 128, 128)
    pooled = functional.max_pool2d(energy, kernel_size=5, stride=1, padding=2)
    maxima = energy.masked_fill(energy.ne(pooled), torch.finfo(energy.dtype).min)
    score, index = maxima.flatten(start_dim=1).topk(count, dim=1)
    range_index = torch.div(index, geometry.azimuth_bins, rounding_mode="floor")
    azimuth_index = index.remainder(geometry.azimuth_bins)
    xy = _polar_indices_to_xy(range_index, azimuth_index, geometry).reshape(batch, frames, count, 2).to(dtype=radar.dtype)
    return xy, score.reshape(batch, frames, count)


class _BaseRecoveryExpert(nn.Module):
    def __init__(self, geometry: BEVGeometry, offset_bound_m: float, sigma_m: float) -> None:
        super().__init__()
        self.geometry = geometry
        self.offset_bound_m = offset_bound_m
        self.sigma_m = sigma_m

    def _render_delta(self, base_xy_m: Tensor, proposal_xy_m: Tensor, strength: Tensor) -> Tensor:
        return gaussian_logit_delta(base_xy_m, proposal_xy_m, strength, self.geometry, self.sigma_m)


class CUCREExpert(_BaseRecoveryExpert):
    """Causal unfolded clutter rejection before the frozen polar-to-BEV interface."""

    def __init__(self, geometry: BEVGeometry) -> None:
        super().__init__(geometry, offset_bound_m=0.0, sigma_m=0.30)
        self.raw_project = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
        )
        self.memory_decay = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)
        self.register_buffer("reference_mask", self._reference_mask(), persistent=False)
        self.reference_logits = nn.Parameter(torch.zeros(1, 1, 7, 7))
        self.suppression = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.polar_out = _zero_conv(nn.Conv2d(16, 48, kernel_size=1))
        self.health_encoder = nn.Sequential(nn.Linear(4, 32), nn.SiLU(inplace=True), nn.Linear(32, 32), nn.SiLU(inplace=True))
        self.human_gate = nn.Linear(32, 16)
        self.acceptance_head = nn.Linear(32, 1)
        self.covariance_head = nn.Linear(32, 2)
        self.state_summary = nn.Sequential(nn.Linear(16, 64), nn.SiLU(inplace=True))

    @staticmethod
    def _reference_mask() -> Tensor:
        mask = torch.ones(7, 7, dtype=torch.bool)
        mask[2:5, 2:5] = False
        return mask.view(1, 1, 7, 7)

    def _reference_statistics(self, current: Tensor) -> tuple[Tensor, Tensor]:
        masked_logits = self.reference_logits.masked_fill(~self.reference_mask, -1e4)
        kernel = torch.softmax(masked_logits.reshape(1, -1), dim=1).reshape(1, 1, 7, 7)
        weight = kernel.expand(current.shape[1], 1, -1, -1)
        mean = functional.conv2d(current, weight, padding=3, groups=current.shape[1])
        second = functional.conv2d(current.square(), weight, padding=3, groups=current.shape[1])
        return mean, (second - mean.square()).clamp_min(1e-5).sqrt()

    def _background_state(self, radar: Tensor) -> tuple[Tensor, Tensor]:
        batch, frames = radar.shape[:2]
        reduced = functional.avg_pool2d(radar.reshape(batch * frames, 2, 128, 128), kernel_size=4)
        sequence = self.raw_project(reduced).reshape(batch, frames, 16, 32, 32)
        state = torch.zeros_like(sequence[:, 0])
        for time_index in range(frames):
            current = sequence[:, time_index]
            decay = torch.sigmoid(self.memory_decay(current))
            state = decay * state + (1.0 - decay) * current
        return sequence[:, -1], state

    def forward(
        self,
        polar_feature: Tensor,
        shared_feature: Tensor,
        radar: Tensor,
        base_logits: Tensor,
        base_xy_m: Tensor,
        previous_xy_m: Tensor,
    ) -> dict[str, Tensor]:
        current, background = self._background_state(radar)
        reference_mean, reference_std = self._reference_statistics(current)
        health_input = torch.cat((_logit_summary(base_logits), (base_xy_m - previous_xy_m).norm(dim=1, keepdim=True)), dim=1)
        # Keep the health vector compact and physical: peak certainty, margin, entropy, innovation.
        health = self.health_encoder(torch.cat((health_input[:, :3], health_input[:, -1:]), dim=1))
        human_preservation = torch.sigmoid(self.human_gate(health)).view(-1, 16, 1, 1)
        normalized_cut = (current - reference_mean) / (reference_std + 1e-3)
        suppression = self.suppression(torch.cat((current, background, normalized_cut), dim=1))
        polar_delta = 0.30 * torch.tanh(
            self.polar_out(functional.interpolate((1.0 - human_preservation) * suppression, size=(128, 128), mode="bilinear", align_corners=False))
        )
        summary = self.state_summary(background.mean(dim=(-2, -1)))
        return {
            "polar_delta": polar_delta,
            "acceptance_logit": self.acceptance_head(health),
            "covariance": _positive_diagonal_covariance(self.covariance_head(health)),
            "causal_state": summary,
            "proposal_xy_m": base_xy_m,
            "clutter_score": normalized_cut.abs().mean(dim=(1, 2, 3), keepdim=True),
        }


class LTGEExpert(_BaseRecoveryExpert):
    """Low-threshold causal tracklet graph for weak, ambiguous polar evidence."""

    def __init__(self, geometry: BEVGeometry, candidate_count: int = 6) -> None:
        super().__init__(geometry, offset_bound_m=0.55, sigma_m=0.30)
        if candidate_count < 4:
            raise ValueError("LTGE needs base, previous, and at least two low-threshold candidates")
        self.candidate_count = candidate_count
        self.local_encoder = _ROIEncoder(2, 24)
        self.token_project = nn.Sequential(nn.Linear(29, 64), nn.SiLU(inplace=True), nn.Linear(64, 64))
        self.query = nn.Linear(64, 64, bias=False)
        self.key = nn.Linear(64, 64, bias=False)
        self.value = nn.Linear(64, 64, bias=False)
        self.message_norm = nn.LayerNorm(64)
        self.refine = nn.Sequential(nn.Linear(64, 64), nn.SiLU(inplace=True), nn.Linear(64, 64), nn.SiLU(inplace=True))
        self.mixture_head = nn.Linear(64, 1)
        self.offset_head = nn.Linear(64, 2)
        self.scale_head = _zero_linear(nn.Linear(64, 1))
        self.acceptance_head = nn.Linear(64, 1)
        self.covariance_head = nn.Linear(64, 2)

    def _candidate_features(self, radar: Tensor, candidates_xy_m: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, frames, count, _ = candidates_xy_m.shape
        source = radar.reshape(batch * frames, 2, 128, 128).repeat_interleave(count, dim=0)
        crops = metric_polar_crop(source, candidates_xy_m.reshape(batch * frames * count, 2), self.geometry, span_m=1.0, output_size=7)
        embedding = self.local_encoder(crops).reshape(batch, frames, count, -1)
        local_score = crops.mean(dim=(-2, -1)).mean(dim=1).reshape(batch, frames, count)
        agreement = -(crops[:, 0] - crops[:, 1]).abs().mean(dim=(-2, -1)).reshape(batch, frames, count)
        return embedding, local_score, agreement

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
        history_xy, _ = _low_threshold_polar_candidates(radar, self.candidate_count, self.geometry)
        low_current = history_xy[:, -1, : self.candidate_count - 2]
        current_xy = torch.cat((base_xy_m[:, None], previous_xy_m[:, None], low_current), dim=1)
        history_xy = history_xy.clone()
        history_xy[:, -1] = current_xy
        embedding, local_score, agreement = self._candidate_features(radar, history_xy)
        time_coordinate = torch.linspace(-1.0, 0.0, frames, device=radar.device, dtype=radar.dtype).view(1, frames, 1, 1)
        token_input = torch.cat(
            (
                embedding,
                local_score.unsqueeze(-1),
                agreement.unsqueeze(-1),
                history_xy / torch.tensor((10.0, 20.0), device=radar.device, dtype=radar.dtype),
                time_coordinate.expand(batch, frames, self.candidate_count, 1),
            ),
            dim=-1,
        )
        tokens = self.token_project(token_input)
        current = tokens[:, -1]
        history = tokens.reshape(batch, frames * self.candidate_count, 64)
        history_positions = history_xy.reshape(batch, frames * self.candidate_count, 2)
        query = self.query(current)
        key = self.key(history)
        edge_logits = torch.einsum("bmd,bnd->bmn", query, key) / math.sqrt(query.shape[-1])
        distance = (current_xy[:, :, None] - history_positions[:, None]).norm(dim=-1)
        frame_delta = (frames - 1 - torch.arange(frames, device=radar.device)).repeat_interleave(self.candidate_count)
        motion_bound = 0.45 + 0.25 * frame_delta.to(dtype=radar.dtype)
        valid_edge = distance <= motion_bound.view(1, 1, -1)
        edge_logits = edge_logits.masked_fill(~valid_edge, -1e4)
        attention = torch.softmax(edge_logits, dim=-1)
        message = torch.bmm(attention, self.value(history))
        state = self.refine(self.message_norm(current + message))
        mixture = torch.softmax(self.mixture_head(state).squeeze(-1), dim=1)
        proposal_candidates = _clamp_metric_xy(
            (current_xy + self.offset_bound_m * torch.tanh(self.offset_head(state))).reshape(batch * self.candidate_count, 2), self.geometry
        ).reshape(batch, self.candidate_count, 2)
        proposal = (mixture.unsqueeze(-1) * proposal_candidates).sum(dim=1)
        strength = torch.tanh(self.scale_head(state.mean(dim=1)))
        return {
            "delta_bev_logit": self._render_delta(base_xy_m, proposal, strength),
            "acceptance_logit": self.acceptance_head(state.mean(dim=1)),
            "covariance": _positive_diagonal_covariance(self.covariance_head(state.mean(dim=1))),
            "causal_state": state.mean(dim=1),
            "proposal_xy_m": proposal,
            "candidate_xy_m": current_xy,
            "candidate_mixture": mixture,
            "graph_link_logits": edge_logits,
            "graph_history_xy_m": history_positions,
            "graph_history_frame_delta": frame_delta,
        }


class GGSRREExpert(_BaseRecoveryExpert):
    """SRRE recovery with a gate trained on correction gain over the frozen base."""

    def __init__(self, geometry: BEVGeometry) -> None:
        super().__init__(geometry, offset_bound_m=0.60, sigma_m=0.38)
        self.local_encoder = _ROIEncoder(128, 32)
        self.radar_state = _CausalRadarState(64)
        self.recovery = nn.Sequential(nn.Linear(96, 64), nn.SiLU(inplace=True), nn.Linear(64, 64), nn.SiLU(inplace=True))
        self.offset_head = nn.Linear(64, 2)
        self.scale_head = _zero_linear(nn.Linear(64, 1))
        self.gain_encoder = nn.Sequential(nn.Linear(14, 48), nn.SiLU(inplace=True), nn.Linear(48, 32), nn.SiLU(inplace=True))
        self.gain_prediction_head = _zero_linear(nn.Linear(32, 1))
        self.acceptance_head = nn.Linear(32, 1)
        self.covariance_head = nn.Linear(32, 2)
        self.training_phase = "joint"

    def set_training_phase(self, phase: str) -> None:
        if phase not in {"joint", "recovery", "gain"}:
            raise ValueError("GG-SRRE phase must be 'joint', 'recovery', or 'gain'")
        self.training_phase = phase
        recovery_modules = (self.local_encoder, self.radar_state, self.recovery, self.offset_head, self.scale_head)
        gain_modules = (self.gain_encoder, self.gain_prediction_head, self.acceptance_head, self.covariance_head)
        for module in (*recovery_modules, *gain_modules):
            module.requires_grad_(phase == "joint")
        if phase == "recovery":
            for module in recovery_modules:
                module.requires_grad_(True)
        elif phase == "gain":
            for module in gain_modules:
                module.requires_grad_(True)

    def _radar_summary(self, radar: Tensor, base_xy_m: Tensor, previous_xy_m: Tensor) -> Tensor:
        latest = radar[:, -1]
        return torch.stack(
            (
                latest.mean(dim=(1, 2, 3)),
                latest.std(dim=(1, 2, 3), unbiased=False),
                (latest[:, 0] - latest[:, 1]).abs().mean(dim=(1, 2)),
                (base_xy_m - previous_xy_m).norm(dim=1),
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
        radar_state = self.radar_state(radar)
        recovery = self.recovery(torch.cat((local, radar_state), dim=1))
        proposal = _clamp_metric_xy(base_xy_m + self.offset_bound_m * torch.tanh(self.offset_head(recovery)), self.geometry)
        raw_delta = self._render_delta(base_xy_m, proposal, torch.tanh(self.scale_head(recovery)))
        candidate_logits = base_logits + raw_delta
        gate_input = torch.cat(
            (
                _logit_summary(base_logits),
                _logit_summary(candidate_logits),
                self._radar_summary(radar, base_xy_m, previous_xy_m),
                raw_delta.abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1),
                (proposal - base_xy_m).norm(dim=1, keepdim=True),
            ),
            dim=1,
        )
        gain_state = self.gain_encoder(gate_input)
        if self.training_phase == "recovery":
            acceptance_override = torch.ones((radar.shape[0], 1), dtype=radar.dtype, device=radar.device)
            acceptance_logit = torch.full_like(acceptance_override, 20.0)
        else:
            acceptance_override = torch.sigmoid(self.acceptance_head(gain_state))
            acceptance_logit = self.acceptance_head(gain_state)
        return {
            "delta_bev_logit": raw_delta,
            "acceptance_logit": acceptance_logit,
            "acceptance_override": acceptance_override,
            "covariance": _positive_diagonal_covariance(self.covariance_head(gain_state)),
            "causal_state": radar_state,
            "proposal_xy_m": proposal,
            "raw_candidate_logits": candidate_logits,
            "gain_prediction": self.gain_prediction_head(gain_state),
        }


class DREAMRecoveryRouteExpertBEV(nn.Module):
    """Frozen DREAM-BEV with a causal recovery expert and exact zero-residual start."""

    def __init__(self, expert_kind: RecoveryExpertKind, logit_cap: float = 3.0, candidate_count: int = 6) -> None:
        super().__init__()
        if expert_kind not in RECOVERY_EXPERT_KINDS:
            raise ValueError(f"unsupported recovery expert kind={expert_kind!r}; expected one of {RECOVERY_EXPERT_KINDS}")
        if logit_cap <= 0.0:
            raise ValueError("logit_cap must be positive")
        self.geometry = BEVGeometry()
        self.expert_kind = expert_kind
        self.logit_cap = logit_cap
        self.backbone = DREAMBEV()
        if expert_kind == "cucre":
            self.expert: nn.Module = CUCREExpert(self.geometry)
        elif expert_kind == "ltge":
            self.expert = LTGEExpert(self.geometry, candidate_count=candidate_count)
        else:
            self.expert = GGSRREExpert(self.geometry)
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "DREAMRecoveryRouteExpertBEV":
        super().train(mode)
        self.backbone.eval()
        return self

    def set_training_phase(self, phase: str) -> None:
        if self.expert_kind == "gg_srre":
            assert isinstance(self.expert, GGSRREExpert)
            self.expert.set_training_phase(phase)
        elif phase != "joint":
            raise ValueError(f"{self.expert_kind} only supports joint training")

    def load_base_checkpoint(self, checkpoint_path: str | Path) -> str:
        path = Path(checkpoint_path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.backbone.load_state_dict(extract_dream_bev_state(payload), strict=True)
        self._freeze_backbone()
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def load_expert_checkpoint(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        state = payload.get("expert_state", payload)
        if state.get("expert_kind") != self.expert_kind:
            raise ValueError("recovery checkpoint expert kind does not match the requested route")
        self.expert.load_state_dict(state["expert"], strict=True)

    def expert_parameters(self) -> Iterator[nn.Parameter]:
        yield from (parameter for parameter in self.expert.parameters() if parameter.requires_grad)

    def expert_state_dict(self) -> dict[str, object]:
        phase = self.expert.training_phase if isinstance(self.expert, GGSRREExpert) else "joint"
        return {"expert_kind": self.expert_kind, "logit_cap": self.logit_cap, "training_phase": phase, "expert": self.expert.state_dict()}

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
        if self.expert_kind == "cucre":
            corrected_polar = polar_feature + expert_output["polar_delta"]
            corrected_shared = self.backbone.shared_from_polar_feature(corrected_polar)
            corrected_logits = self.backbone.decoder.decode_shared(corrected_shared)
            raw_delta = corrected_logits - base_logits
            proposal_xy_m, _ = decode_peak_xy(corrected_logits, self.geometry)
            expert_output["proposal_xy_m"] = proposal_xy_m
            expert_output["corrected_polar_feature"] = corrected_polar
        else:
            raw_delta = expert_output["delta_bev_logit"]
        delta = raw_delta.clamp(-self.logit_cap, self.logit_cap)
        acceptance = expert_output.get("acceptance_override")
        if acceptance is None:
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
            **{
                key: value
                for key, value in expert_output.items()
                if key not in {"delta_bev_logit", "acceptance_logit", "acceptance_override", "covariance", "causal_state", "proposal_xy_m"}
            },
        }



