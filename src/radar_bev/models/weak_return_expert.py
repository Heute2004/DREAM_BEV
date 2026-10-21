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
from ..multiview import range_indices_from_xy
from .backbones import DREAMBEV, extract_dream_bev_state
from .recovery_experts import LTGEExpert, _low_threshold_polar_candidates
from .heterogeneous_experts import _ROIEncoder, _clamp_metric_xy, _positive_diagonal_covariance, _zero_linear


WeakReturnExpertKind = Literal["scar", "ltge_cen"]
WEAK_RETURN_EXPERT_KINDS: tuple[WeakReturnExpertKind, ...] = ("scar", "ltge_cen")


class _BaseWeakReturnExpert(nn.Module):
    def __init__(self, geometry: BEVGeometry, sigma_m: float) -> None:
        super().__init__()
        self.geometry = geometry
        self.sigma_m = sigma_m

    def _render_delta(self, base_xy_m: Tensor, proposal_xy_m: Tensor, strength: Tensor) -> Tensor:
        return gaussian_logit_delta(base_xy_m, proposal_xy_m, strength, self.geometry, self.sigma_m)


class SCARExpert(_BaseWeakReturnExpert):
    """Selective two-stage local accuracy refiner around a frozen base peak."""

    def __init__(self, geometry: BEVGeometry) -> None:
        super().__init__(geometry, sigma_m=0.24)
        self.stage1_polar = _ROIEncoder(48, 32)
        self.stage1_bev = _ROIEncoder(128, 32)
        self.stage1 = nn.Sequential(nn.Linear(68, 80), nn.SiLU(inplace=True), nn.Linear(80, 64), nn.SiLU(inplace=True))
        self.coarse_offset = nn.Linear(64, 2)
        self.stage2_polar = _ROIEncoder(48, 24)
        self.stage2_bev = _ROIEncoder(128, 24)
        self.stage2 = nn.Sequential(nn.Linear(112, 80), nn.SiLU(inplace=True), nn.Linear(80, 64), nn.SiLU(inplace=True))
        self.fine_offset = nn.Linear(64, 2)
        self.scale_head = _zero_linear(nn.Linear(64, 1))
        self.acceptance_head = nn.Linear(64, 1)
        self.covariance_head = nn.Linear(64, 2)

    def forward(self, polar_feature: Tensor, shared_feature: Tensor, radar: Tensor, base_logits: Tensor, base_xy_m: Tensor, previous_xy_m: Tensor) -> dict[str, Tensor]:
        innovation = base_xy_m - previous_xy_m
        stage1_feature = torch.cat(
            (
                self.stage1_polar(metric_polar_crop(polar_feature, base_xy_m, self.geometry, span_m=1.8, output_size=11)),
                self.stage1_bev(metric_bev_crop(shared_feature, base_xy_m, self.geometry, span_m=2.4, output_size=21)),
                innovation,
                base_xy_m / 20.0,
            ),
            dim=1,
        )
        stage1_state = self.stage1(stage1_feature)
        coarse_delta = 0.35 * torch.tanh(self.coarse_offset(stage1_state))
        coarse_xy_m = _clamp_metric_xy(base_xy_m + coarse_delta, self.geometry)
        stage2_feature = torch.cat(
            (
                stage1_state,
                self.stage2_polar(metric_polar_crop(polar_feature, coarse_xy_m, self.geometry, span_m=1.0, output_size=9)),
                self.stage2_bev(metric_bev_crop(shared_feature, coarse_xy_m, self.geometry, span_m=1.0, output_size=11)),
            ),
            dim=1,
        )
        stage2_state = self.stage2(stage2_feature)
        fine_delta = 0.12 * torch.tanh(self.fine_offset(stage2_state))
        proposal_xy_m = _clamp_metric_xy(coarse_xy_m + fine_delta, self.geometry)
        return {
            "delta_bev_logit": self._render_delta(base_xy_m, proposal_xy_m, torch.tanh(self.scale_head(stage2_state))),
            "acceptance_logit": self.acceptance_head(stage2_state),
            "covariance": _positive_diagonal_covariance(self.covariance_head(stage2_state)),
            "causal_state": stage2_state,
            "proposal_xy_m": proposal_xy_m,
            "coarse_xy_m": coarse_xy_m,
            "fine_delta_m": fine_delta,
        }


class LTGECENExpert(LTGEExpert):
    """LTGE with one dense causal corridor-evidence node per history frame."""

    def __init__(self, geometry: BEVGeometry, candidate_count: int = 8, rd_mode: Literal["off", "mti", "dual"] = "off") -> None:
        super().__init__(geometry, candidate_count=candidate_count)
        if rd_mode not in {"off", "mti", "dual"}:
            raise ValueError(f"unsupported rd_mode={rd_mode!r}")
        self.rd_mode = rd_mode
        self.corridor_offset = _zero_linear(nn.Linear(24, 2))
        if rd_mode != "off":
            channels = 1 if rd_mode == "mti" else 2
            self.rd_encoder = nn.Sequential(
                nn.Conv2d(channels, 8, kernel_size=(3, 5), stride=(1, 4), padding=(1, 2), bias=False),
                nn.GroupNorm(4, 8),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                nn.Linear(8, 16),
                nn.SiLU(inplace=True),
            )
            # This keeps the complete RA-only LTGE-CEN output exactly unchanged at initialization.
            self.rd_token_project = _zero_linear(nn.Linear(16, 64))

    def _corridor_nodes(self, radar: Tensor, centers_xy_m: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, frames, _ = centers_xy_m.shape
        source = radar.reshape(batch * frames, 2, 128, 128)
        crops = metric_polar_crop(source, centers_xy_m.reshape(batch * frames, 2), self.geometry, span_m=2.4, output_size=11)
        embedding = self.local_encoder(crops).reshape(batch, frames, -1)
        offset = 0.45 * torch.tanh(self.corridor_offset(embedding))
        node_xy_m = _clamp_metric_xy((centers_xy_m + offset).reshape(-1, 2), self.geometry).reshape(batch, frames, 2)
        energy = crops.mean(dim=1).flatten(start_dim=1)
        distribution = torch.softmax(energy / 0.25, dim=1)
        entropy = -(distribution * distribution.clamp_min(1e-6).log()).sum(dim=1).reshape(batch, frames)
        evidence = torch.logsumexp(energy, dim=1).reshape(batch, frames)
        agreement = -(crops[:, 0] - crops[:, 1]).abs().mean(dim=(-2, -1)).reshape(batch, frames)
        return node_xy_m, embedding, evidence - entropy, agreement

    def _rd_candidate_features(self, rd: Tensor | None, rd_mti: Tensor | None, candidates_xy_m: Tensor) -> Tensor:
        if self.rd_mode == "off":
            raise RuntimeError("RD side features are unavailable when rd_mode='off'")
        if rd_mti is None:
            raise ValueError("rd_mti is required when LTGE-CEN RD sidecar is enabled")
        if rd_mti.ndim != 4 or rd_mti.shape[2:] != (128, 256):
            raise ValueError("rd_mti must be [batch,frames,128,256]")
        if self.rd_mode == "dual":
            if rd is None or rd.shape != rd_mti.shape:
                raise ValueError("rd must match rd_mti for rd_mode='dual'")
            source = torch.stack((rd, rd_mti), dim=2)
        else:
            source = rd_mti.unsqueeze(2)
        batch, frames, channels, _, doppler_bins = source.shape
        if candidates_xy_m.shape[:3] != (batch, frames, self.candidate_count):
            raise ValueError("candidate coordinates must be [batch,frames,candidate_count,2]")
        count = self.candidate_count
        source = source.reshape(batch * frames, channels, 128, doppler_bins).repeat_interleave(count, dim=0)
        range_index = range_indices_from_xy(candidates_xy_m.reshape(batch * frames * count, 2), self.geometry)
        offsets = torch.arange(-2, 3, device=source.device)
        rows = (range_index[:, None] + offsets[None]).clamp(0, 127)
        patch = source.gather(2, rows[:, None, :, None].expand(-1, channels, -1, doppler_bins))
        return self.rd_encoder(patch).reshape(batch, frames, count, 16)

    def forward(
        self,
        polar_feature: Tensor,
        shared_feature: Tensor,
        radar: Tensor,
        base_logits: Tensor,
        base_xy_m: Tensor,
        previous_xy_m: Tensor,
        rd: Tensor | None = None,
        rd_mti: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch, frames = radar.shape[:2]
        low_xy_m, low_score = _low_threshold_polar_candidates(radar, self.candidate_count, self.geometry)
        velocity = base_xy_m - previous_xy_m
        offset = torch.arange(frames, device=radar.device, dtype=radar.dtype) - float(frames - 1)
        corridor_center = _clamp_metric_xy(
            (base_xy_m[:, None] + offset.view(1, frames, 1) * velocity[:, None]).reshape(-1, 2), self.geometry
        ).reshape(batch, frames, 2)
        corridor_xy_m, corridor_embedding, corridor_score, corridor_agreement = self._corridor_nodes(radar, corridor_center)
        candidates_xy_m = torch.cat((corridor_xy_m[:, :, None], low_xy_m[:, :, : self.candidate_count - 1]), dim=2)
        current_low = low_xy_m[:, -1, : self.candidate_count - 3]
        candidates_xy_m[:, -1] = torch.cat((corridor_xy_m[:, -1, None], base_xy_m[:, None], previous_xy_m[:, None], current_low), dim=1)
        embedding, local_score, agreement = self._candidate_features(radar, candidates_xy_m)
        embedding[:, :, 0] = corridor_embedding
        local_score[:, :, 0] = corridor_score
        agreement[:, :, 0] = corridor_agreement
        time_coordinate = torch.linspace(-1.0, 0.0, frames, device=radar.device, dtype=radar.dtype).view(1, frames, 1, 1)
        token_input = torch.cat(
            (
                embedding,
                local_score.unsqueeze(-1),
                agreement.unsqueeze(-1),
                candidates_xy_m / torch.tensor((10.0, 20.0), device=radar.device, dtype=radar.dtype),
                time_coordinate.expand(batch, frames, self.candidate_count, 1),
            ),
            dim=-1,
        )
        tokens = self.token_project(token_input)
        rd_candidate_embedding: Tensor | None = None
        if self.rd_mode != "off":
            rd_candidate_embedding = self._rd_candidate_features(rd, rd_mti, candidates_xy_m)
            tokens = tokens + self.rd_token_project(rd_candidate_embedding)
        current = tokens[:, -1]
        history = tokens.reshape(batch, frames * self.candidate_count, 64)
        history_positions = candidates_xy_m.reshape(batch, frames * self.candidate_count, 2)
        query = self.query(current)
        key = self.key(history)
        edge_logits = torch.einsum("bmd,bnd->bmn", query, key) / math.sqrt(query.shape[-1])
        current_xy_m = candidates_xy_m[:, -1]
        distance = (current_xy_m[:, :, None] - history_positions[:, None]).norm(dim=-1)
        frame_delta = (frames - 1 - torch.arange(frames, device=radar.device)).repeat_interleave(self.candidate_count)
        valid_edge = distance <= (0.45 + 0.25 * frame_delta.to(dtype=radar.dtype)).view(1, 1, -1)
        attention = torch.softmax(edge_logits.masked_fill(~valid_edge, -1e4), dim=-1)
        message = torch.bmm(attention, self.value(history))
        state = self.refine(self.message_norm(current + message))
        mixture = torch.softmax(self.mixture_head(state).squeeze(-1), dim=1)
        corrected_candidates = _clamp_metric_xy(
            (current_xy_m + self.offset_bound_m * torch.tanh(self.offset_head(state))).reshape(batch * self.candidate_count, 2), self.geometry
        ).reshape(batch, self.candidate_count, 2)
        proposal_xy_m = (mixture.unsqueeze(-1) * corrected_candidates).sum(dim=1)
        state_summary = state.mean(dim=1)
        output = {
            "delta_bev_logit": self._render_delta(base_xy_m, proposal_xy_m, torch.tanh(self.scale_head(state_summary))),
            "acceptance_logit": self.acceptance_head(state_summary),
            "covariance": _positive_diagonal_covariance(self.covariance_head(state_summary)),
            "causal_state": state_summary,
            "proposal_xy_m": proposal_xy_m,
            "candidate_xy_m": current_xy_m,
            "candidate_history_xy_m": candidates_xy_m,
            "candidate_mixture": mixture,
            "graph_link_logits": edge_logits,
            "graph_history_xy_m": history_positions,
            "graph_history_frame_delta": frame_delta,
            "corridor_node_xy_m": corridor_xy_m,
        }
        if rd_candidate_embedding is not None:
            output["rd_candidate_embedding"] = rd_candidate_embedding
        return output


class DREAMWeakReturnExpertBEV(nn.Module):
    """Frozen DREAM-BEV with a selective weak-return evidence expert."""

    def __init__(
        self,
        expert_kind: WeakReturnExpertKind,
        logit_cap: float = 3.0,
        candidate_count: int = 8,
        rd_mode: Literal["off", "mti", "dual"] = "off",
    ) -> None:
        super().__init__()
        if expert_kind not in WEAK_RETURN_EXPERT_KINDS:
            raise ValueError(f"unsupported weak-return expert kind={expert_kind!r}")
        self.geometry = BEVGeometry()
        self.expert_kind = expert_kind
        self.logit_cap = logit_cap
        self.rd_mode = rd_mode
        self.backbone = DREAMBEV()
        self.expert: nn.Module = SCARExpert(self.geometry) if expert_kind == "scar" else LTGECENExpert(self.geometry, candidate_count, rd_mode)
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "DREAMWeakReturnExpertBEV":
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
        return {"expert_kind": self.expert_kind, "logit_cap": self.logit_cap, "rd_mode": self.rd_mode, "expert": self.expert.state_dict()}

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

    def forward(self, radar: Tensor, rd: Tensor | None = None, rd_mti: Tensor | None = None) -> dict[str, Tensor]:
        self.backbone.eval()
        polar_feature, shared_feature, base_logits, peaks = self._base_observation(radar)
        base_xy_m, previous_xy_m = peaks[:, :2], peaks[:, 2:]
        if self.expert_kind == "ltge_cen":
            expert_output = self.expert(
                polar_feature, shared_feature, radar, base_logits, base_xy_m, previous_xy_m, rd=rd, rd_mti=rd_mti
            )
        else:
            expert_output = self.expert(polar_feature, shared_feature, radar, base_logits, base_xy_m, previous_xy_m)
        delta = expert_output["delta_bev_logit"].clamp(-self.logit_cap, self.logit_cap)
        acceptance = torch.sigmoid(expert_output["acceptance_logit"])
        logits = base_logits + acceptance[:, :, None, None] * delta
        return {
            "logits": logits, "base_logits": base_logits, "delta_bev_logit": delta,
            "acceptance": acceptance, "acceptance_logit": expert_output["acceptance_logit"],
            "covariance": expert_output["covariance"], "causal_state": expert_output["causal_state"],
            "proposal_xy_m": expert_output["proposal_xy_m"], "base_peak_xy_m": base_xy_m,
            "previous_peak_xy_m": previous_xy_m, "shared_feature": shared_feature,
            **{key: value for key, value in expert_output.items() if key not in {"delta_bev_logit", "acceptance_logit", "covariance", "causal_state", "proposal_xy_m"}},
        }



