from __future__ import annotations

"""PB-MHBR Stage 1: frozen-DREAM Polar-BEV candidate re-ranking.

This first stage intentionally has no free coordinate regressor and no learned
temporal state. It scores only causal candidates from frozen BEV, current
RA/RA-MTI Polar evidence, the previous frozen peak, and a null placeholder.
"""

import hashlib
from collections.abc import Iterator
from pathlib import Path

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
from .recovery_experts import _low_threshold_polar_candidates
from .heterogeneous_experts import _ROIEncoder, _positive_diagonal_covariance, _zero_linear


def calibrate_peak_confidence(base_logits: Tensor, quality_logit: Tensor) -> Tensor:
    """Set the peak confidence while preserving every spatial ordering."""
    if base_logits.ndim != 4 or quality_logit.shape != (base_logits.shape[0], 1):
        raise ValueError("expected base_logits [batch,1,x,y] and quality_logit [batch,1]")
    peak_logit = base_logits.flatten(start_dim=1).amax(dim=1, keepdim=True)
    shift = quality_logit - peak_logit
    return base_logits + shift[:, :, None, None]


def fixed_backbone_window(radar: Tensor, clip_length: int = 16) -> Tensor:
    """Keep the frozen backbone on its trained causal window contract."""
    if radar.ndim != 5:
        raise ValueError("radar must have shape [batch,time,channels,height,width]")
    if radar.shape[1] < clip_length:
        raise ValueError(
            f"radar history has {radar.shape[1]} frames; expected at least {clip_length}"
        )
    return radar[:, -clip_length:]


def range_doppler_candidate_statistics(
    rd: Tensor,
    rd_mti: Tensor,
    candidates_xy_m: Tensor,
    geometry: BEVGeometry,
) -> Tensor:
    """Extract candidate-level RD evidence without adding a dense RD branch."""
    if rd.shape != rd_mti.shape or rd.ndim != 4:
        raise ValueError("rd and rd_mti must have matching [batch,time,range,doppler] shapes")
    if candidates_xy_m.ndim != 3 or candidates_xy_m.shape[0] != rd.shape[0]:
        raise ValueError("candidates_xy_m must have shape [batch,candidates,2]")
    batch, time, _, doppler_bins = rd.shape
    count = candidates_xy_m.shape[1]
    candidate_range_m = candidates_xy_m.norm(dim=-1)
    range_index = torch.round(
        candidate_range_m / geometry.range_resolution_m
    ).long().clamp(0, rd.shape[2] - 1)
    gather_index = range_index[:, None, :, None].expand(
        batch,
        time,
        count,
        doppler_bins,
    )
    rd_spectrum = torch.gather(rd, dim=2, index=gather_index)
    mti_spectrum = torch.gather(rd_mti, dim=2, index=gather_index)
    latest = rd_spectrum[:, -1]
    latest_mti = mti_spectrum[:, -1]
    probability = torch.softmax(2.0 * latest.float(), dim=-1).to(latest.dtype)
    axis = torch.linspace(
        -1.0,
        1.0,
        doppler_bins,
        dtype=latest.dtype,
        device=latest.device,
    )
    centroid = (probability * axis).sum(dim=-1)
    spread = torch.sqrt(
        (probability * (axis - centroid[:, :, None]).square())
        .sum(dim=-1)
        .clamp_min(1e-8)
    )
    half_width = max(1, doppler_bins // 32)
    centre = doppler_bins // 2
    zero_ratio = probability[
        ...,
        max(0, centre - half_width) : min(doppler_bins, centre + half_width + 1),
    ].sum(dim=-1)
    nonzero_ratio = 1.0 - zero_ratio
    rd_peak = latest.amax(dim=-1)
    mti_peak = latest_mti.amax(dim=-1)
    temporal_peak_std = rd_spectrum.amax(dim=-1).std(
        dim=1,
        unbiased=False,
    )
    return torch.stack(
        (
            rd_peak,
            mti_peak,
            latest.mean(dim=-1),
            latest_mti.mean(dim=-1),
            centroid,
            spread,
            zero_ratio,
            nonzero_ratio,
            mti_peak - rd_peak,
            temporal_peak_std,
        ),
        dim=-1,
    )


class PBMHBRStage1CandidateReranker(nn.Module):
    """Score a fixed, causal eight-token Polar-BEV candidate set.

    Candidate order is fixed and auditable:
    base, BEV rank-2, BEV rank-3, Polar ranks 1-3, previous base, null.
    The null token reuses the base coordinate so selecting it never creates a
    synthetic radar measurement; Stage 2 will give it an explicit state role.
    """

    candidate_count = 8

    def __init__(self, geometry: BEVGeometry) -> None:
        super().__init__()
        self.geometry = geometry
        self.polar_encoder = _ROIEncoder(48, 24)
        self.raw_encoder = _ROIEncoder(2, 20)
        self.bev_encoder = _ROIEncoder(128, 24)
        self.type_embedding = nn.Embedding(5, 8)
        self.token_project = nn.Sequential(
            nn.Linear(24 + 20 + 24 + 8 + 7, 96),
            nn.SiLU(inplace=True),
            nn.Linear(96, 96),
            nn.SiLU(inplace=True),
        )
        self.token_attention = nn.MultiheadAttention(96, num_heads=4, batch_first=True)
        self.token_norm = nn.LayerNorm(96)
        # Equal initial scores make the deterministic argmax select candidate 0
        # (the frozen base), preserving exact insertion identity.
        self.selector_head = _zero_linear(nn.Linear(96, 1))
        self.state_head = nn.Sequential(nn.Linear(96 + 3, 72), nn.SiLU(inplace=True), nn.Linear(72, 64), nn.SiLU(inplace=True))
        self.scale_head = _zero_linear(nn.Linear(64, 1))
        self.acceptance_head = _zero_linear(nn.Linear(64, 1))
        self.covariance_head = nn.Linear(64, 2)

    def _candidate_sources(
        self,
        radar: Tensor,
        base_logits: Tensor,
        previous_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        base_xy_m, base_score = decode_peak_xy(base_logits, self.geometry)
        previous_xy_m, previous_score = decode_peak_xy(previous_logits, self.geometry)
        bev_xy_m, bev_score = local_maximum_candidates(base_logits, 3, self.geometry)
        polar_xy_m, polar_score = _low_threshold_polar_candidates(radar, 3, self.geometry)
        polar_xy_m = polar_xy_m[:, -1]
        polar_score = polar_score[:, -1]
        null_xy_m = base_xy_m
        candidates = torch.cat(
            (
                base_xy_m[:, None],
                bev_xy_m[:, 1:],
                polar_xy_m,
                previous_xy_m[:, None],
                null_xy_m[:, None],
            ),
            dim=1,
        )
        scores = torch.cat(
            (
                base_score,
                bev_score[:, 1:],
                polar_score,
                previous_score,
                torch.zeros_like(base_score),
            ),
            dim=1,
        )
        type_ids = torch.tensor((0, 1, 1, 2, 2, 2, 3, 4), device=radar.device).expand(radar.shape[0], -1)
        return candidates, scores, type_ids, base_xy_m

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
        previous_logits: Tensor,
        *,
        rd: Tensor | None = None,
        rd_mti: Tensor | None = None,
    ) -> dict[str, Tensor]:
        candidates, source_scores, type_ids, base_xy_m = self._candidate_sources(radar, base_logits, previous_logits)
        previous_xy_m, _ = decode_peak_xy(previous_logits, self.geometry)
        batch, count, _ = candidates.shape
        polar_crop = self._candidate_crops(polar_feature, candidates, metric_polar_crop, 0.90, 9)
        raw_crop = self._candidate_crops(radar[:, -1], candidates, metric_polar_crop, 0.90, 9)
        bev_crop = self._candidate_crops(shared_feature, candidates, metric_bev_crop, 1.50, 9)
        polar_embedding = self.polar_encoder(polar_crop.reshape(batch * count, *polar_crop.shape[2:])).reshape(batch, count, -1)
        raw_embedding = self.raw_encoder(raw_crop.reshape(batch * count, *raw_crop.shape[2:])).reshape(batch, count, -1)
        bev_embedding = self.bev_encoder(bev_crop.reshape(batch * count, *bev_crop.shape[2:])).reshape(batch, count, -1)
        candidate_geometry = torch.cat(
            (
                candidates / torch.tensor((10.0, 23.4), dtype=radar.dtype, device=radar.device),
                (candidates - base_xy_m[:, None]) / 2.0,
                (candidates - previous_xy_m[:, None]) / 2.0,
                source_scores[:, :, None] / 4.0,
            ),
            dim=-1,
        )
        tokens = self.token_project(
            torch.cat((polar_embedding, raw_embedding, bev_embedding, self.type_embedding(type_ids), candidate_geometry), dim=-1)
        )
        attended, _ = self.token_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.token_norm(tokens + attended)
        selector_logits = self.selector_head(tokens).squeeze(-1)
        mixture = torch.softmax(selector_logits, dim=1)
        selected_index = mixture.argmax(dim=1)
        hard = functional.one_hot(selected_index, num_classes=count).to(dtype=mixture.dtype)
        straight_through = hard + mixture - mixture.detach()
        proposal_xy_m = (straight_through[:, :, None] * candidates).sum(dim=1)
        nonbase_soft = mixture[:, 1:-1].sum(dim=1, keepdim=True)
        nonbase_hard = ((selected_index > 0) & (selected_index < count - 1)).to(mixture.dtype)[:, None]
        stage1_acceptance = nonbase_hard + nonbase_soft - nonbase_soft.detach()
        selected_token = (mixture[:, :, None] * tokens).sum(dim=1)
        latest = radar[:, -1]
        global_health = torch.stack(
            (
                latest.mean(dim=(1, 2, 3)),
                latest.std(dim=(1, 2, 3), unbiased=False),
                (latest[:, 0] - latest[:, 1]).abs().mean(dim=(1, 2)),
            ),
            dim=1,
        )
        state = self.state_head(torch.cat((selected_token, global_health), dim=1))
        strength = 3.0 * torch.tanh(self.scale_head(state))
        return {
            "delta_bev_logit": gaussian_logit_delta(base_xy_m, proposal_xy_m, strength, self.geometry, sigma_m=0.22),
            "acceptance_logit": self.acceptance_head(state),
            "covariance": _positive_diagonal_covariance(self.covariance_head(state)),
            "causal_state": state,
            "proposal_xy_m": proposal_xy_m,
            "candidate_xy_m": candidates,
            "candidate_selector_logits": selector_logits,
            "candidate_mixture": mixture,
            "selected_candidate_index": selected_index,
            "stage1_acceptance": stage1_acceptance,
            "base_peak_xy_m": base_xy_m,
            "previous_peak_xy_m": previous_xy_m,
            "candidate_type_ids": type_ids,
            "candidate_source_scores": source_scores,
        }


class PBMHBRStage2CandidateReranker(PBMHBRStage1CandidateReranker):
    """Add causal local evidence and three previous-frame branch hypotheses."""

    temporal_embedding_dim = 32
    branch_count = 3

    def __init__(self, geometry: BEVGeometry, use_rd: bool = False) -> None:
        super().__init__(geometry)
        self.use_rd = bool(use_rd)
        self.temporal_gru = nn.GRU(20, self.temporal_embedding_dim, batch_first=True)
        self.rd_encoder = (
            nn.Sequential(
                nn.Linear(10, 32),
                nn.SiLU(inplace=True),
                nn.Linear(32, 16),
                nn.SiLU(inplace=True),
            )
            if self.use_rd
            else None
        )
        rd_embedding_dim = 16 if self.use_rd else 0
        self.token_project = nn.Sequential(
            nn.Linear(
                24
                + 20
                + 24
                + 8
                + 7
                + self.temporal_embedding_dim
                + self.branch_count
                + rd_embedding_dim,
                96,
            ),
            nn.SiLU(inplace=True),
            nn.Linear(96, 96),
            nn.SiLU(inplace=True),
        )
        self.selector_head = _zero_linear(nn.Linear(96, 1))

    def _temporal_candidate_embedding(self, radar: Tensor, candidates_xy_m: Tensor) -> Tensor:
        batch, time, channels, height, width = radar.shape
        count = candidates_xy_m.shape[1]
        frames = radar.reshape(batch * time, channels, height, width)
        repeated_frames = frames.repeat_interleave(count, dim=0)
        repeated_candidates = (
            candidates_xy_m[:, None]
            .expand(batch, time, count, 2)
            .reshape(batch * time * count, 2)
        )
        crop = metric_polar_crop(
            repeated_frames,
            repeated_candidates,
            self.geometry,
            span_m=0.90,
            output_size=9,
        )
        frame_embedding = self.raw_encoder(crop)
        sequence = (
            frame_embedding.reshape(batch, time, count, -1)
            .permute(0, 2, 1, 3)
            .reshape(batch * count, time, -1)
        )
        _, hidden = self.temporal_gru(sequence)
        return hidden[-1].reshape(batch, count, self.temporal_embedding_dim)

    def forward(
        self,
        polar_feature: Tensor,
        shared_feature: Tensor,
        radar: Tensor,
        base_logits: Tensor,
        previous_logits: Tensor,
        *,
        rd: Tensor | None = None,
        rd_mti: Tensor | None = None,
    ) -> dict[str, Tensor]:
        candidates, source_scores, type_ids, base_xy_m = self._candidate_sources(
            radar,
            base_logits,
            previous_logits,
        )
        previous_xy_m, _ = decode_peak_xy(previous_logits, self.geometry)
        previous_branches, previous_branch_scores = local_maximum_candidates(
            previous_logits,
            self.branch_count,
            self.geometry,
        )
        batch, count, _ = candidates.shape
        polar_crop = self._candidate_crops(
            polar_feature,
            candidates,
            metric_polar_crop,
            0.90,
            9,
        )
        raw_crop = self._candidate_crops(
            radar[:, -1],
            candidates,
            metric_polar_crop,
            0.90,
            9,
        )
        bev_crop = self._candidate_crops(
            shared_feature,
            candidates,
            metric_bev_crop,
            1.50,
            9,
        )
        polar_embedding = self.polar_encoder(
            polar_crop.reshape(batch * count, *polar_crop.shape[2:])
        ).reshape(batch, count, -1)
        raw_embedding = self.raw_encoder(
            raw_crop.reshape(batch * count, *raw_crop.shape[2:])
        ).reshape(batch, count, -1)
        bev_embedding = self.bev_encoder(
            bev_crop.reshape(batch * count, *bev_crop.shape[2:])
        ).reshape(batch, count, -1)
        temporal_embedding = self._temporal_candidate_embedding(radar, candidates)
        rd_statistics = None
        rd_embedding = None
        if self.use_rd:
            if rd is None or rd_mti is None or self.rd_encoder is None:
                raise ValueError("PB-MHBR Stage2-RD requires aligned rd and rd_mti tensors")
            rd_statistics = range_doppler_candidate_statistics(
                rd,
                rd_mti,
                candidates,
                self.geometry,
            )
            rd_embedding = self.rd_encoder(rd_statistics)
        branch_distance = (
            candidates[:, :, None] - previous_branches[:, None]
        ).norm(dim=-1)
        candidate_branch_logits = -2.0 * branch_distance
        score_scale = previous_branch_scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        candidate_branch_logits = candidate_branch_logits + (
            previous_branch_scores - previous_branch_scores.mean(dim=1, keepdim=True)
        )[:, None] / score_scale[:, None]
        candidate_branch_posterior = torch.softmax(candidate_branch_logits, dim=-1)
        candidate_geometry = torch.cat(
            (
                candidates
                / torch.tensor(
                    (10.0, 23.4),
                    dtype=radar.dtype,
                    device=radar.device,
                ),
                (candidates - base_xy_m[:, None]) / 2.0,
                (candidates - previous_xy_m[:, None]) / 2.0,
                source_scores[:, :, None] / 4.0,
            ),
            dim=-1,
        )
        token_parts = [
            polar_embedding,
            raw_embedding,
            bev_embedding,
            self.type_embedding(type_ids),
            candidate_geometry,
            temporal_embedding,
            branch_distance / 2.0,
        ]
        if rd_embedding is not None:
            token_parts.append(rd_embedding)
        tokens = self.token_project(
            torch.cat(tuple(token_parts), dim=-1)
        )
        attended, _ = self.token_attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.token_norm(tokens + attended)
        selector_logits = self.selector_head(tokens).squeeze(-1)
        mixture = torch.softmax(selector_logits, dim=1)
        selected_index = mixture.argmax(dim=1)
        hard = functional.one_hot(selected_index, num_classes=count).to(dtype=mixture.dtype)
        straight_through = hard + mixture - mixture.detach()
        proposal_xy_m = (straight_through[:, :, None] * candidates).sum(dim=1)
        nonbase_soft = mixture[:, 1:-1].sum(dim=1, keepdim=True)
        nonbase_hard = ((selected_index > 0) & (selected_index < count - 1)).to(
            mixture.dtype
        )[:, None]
        stage1_acceptance = nonbase_hard + nonbase_soft - nonbase_soft.detach()
        selected_token = (mixture[:, :, None] * tokens).sum(dim=1)
        latest = radar[:, -1]
        global_health = torch.stack(
            (
                latest.mean(dim=(1, 2, 3)),
                latest.std(dim=(1, 2, 3), unbiased=False),
                (latest[:, 0] - latest[:, 1]).abs().mean(dim=(1, 2)),
            ),
            dim=1,
        )
        state = self.state_head(torch.cat((selected_token, global_health), dim=1))
        strength = 3.0 * torch.tanh(self.scale_head(state))
        branch_posterior = (
            mixture[:, :, None] * candidate_branch_posterior
        ).sum(dim=1)
        output = {
            "delta_bev_logit": gaussian_logit_delta(
                base_xy_m,
                proposal_xy_m,
                strength,
                self.geometry,
                sigma_m=0.22,
            ),
            "acceptance_logit": self.acceptance_head(state),
            "covariance": _positive_diagonal_covariance(
                self.covariance_head(state)
            ),
            "causal_state": state,
            "proposal_xy_m": proposal_xy_m,
            "candidate_xy_m": candidates,
            "candidate_selector_logits": selector_logits,
            "candidate_mixture": mixture,
            "selected_candidate_index": selected_index,
            "stage1_acceptance": stage1_acceptance,
            "base_peak_xy_m": base_xy_m,
            "previous_peak_xy_m": previous_xy_m,
            "candidate_type_ids": type_ids,
            "candidate_source_scores": source_scores,
            "temporal_candidate_embedding": temporal_embedding,
            "previous_branch_xy_m": previous_branches,
            "candidate_branch_posterior": candidate_branch_posterior,
            "branch_posterior": branch_posterior,
        }
        if rd_statistics is not None and rd_embedding is not None:
            output["rd_candidate_statistics"] = rd_statistics
            output["rd_candidate_embedding"] = rd_embedding
        return output


class PBMHBRStage2RDCandidateReranker(PBMHBRStage2CandidateReranker):
    """Stage-2 branch resolver augmented by candidate-level RD summaries."""

    def __init__(self, geometry: BEVGeometry) -> None:
        super().__init__(geometry, use_rd=True)


class PBMHBRQualityCandidateReranker(PBMHBRStage2CandidateReranker):
    """Predict correctness@0.30 m without moving the frozen backbone peak."""

    def __init__(self, geometry: BEVGeometry, use_rd: bool = False) -> None:
        super().__init__(geometry, use_rd=use_rd)
        self.quality_residual_head = nn.Sequential(
            nn.Linear(64 + 10, 32),
            nn.SiLU(inplace=True),
            _zero_linear(nn.Linear(32, 1)),
        )

    def forward(
        self,
        polar_feature: Tensor,
        shared_feature: Tensor,
        radar: Tensor,
        base_logits: Tensor,
        previous_logits: Tensor,
        *,
        rd: Tensor | None = None,
        rd_mti: Tensor | None = None,
    ) -> dict[str, Tensor]:
        output = super().forward(
            polar_feature,
            shared_feature,
            radar,
            base_logits,
            previous_logits,
            rd=rd,
            rd_mti=rd_mti,
        )
        top_two = base_logits.flatten(start_dim=1).topk(k=2, dim=1).values
        base_peak_logit = top_two[:, :1]
        base_xy_m = output["base_peak_xy_m"]
        previous_delta = base_xy_m - output["previous_peak_xy_m"]
        mixture = output["candidate_mixture"].clamp_min(1e-8)
        quality_features = torch.cat(
            (
                base_peak_logit,
                top_two[:, :1] - top_two[:, 1:2],
                base_peak_logit.sigmoid(),
                base_xy_m
                / torch.tensor(
                    (10.0, 23.4),
                    dtype=base_xy_m.dtype,
                    device=base_xy_m.device,
                ),
                previous_delta / 2.0,
                previous_delta.norm(dim=1, keepdim=True) / 2.0,
                mixture.amax(dim=1, keepdim=True),
                -(mixture * mixture.log()).sum(dim=1, keepdim=True),
            ),
            dim=1,
        )
        quality_residual = self.quality_residual_head(
            torch.cat((output["causal_state"], quality_features), dim=1)
        )
        output["quality_residual_logit"] = quality_residual
        output["correctness_logit"] = base_peak_logit + quality_residual
        return output


class DREAMDryComplexExpertBEV(nn.Module):
    """Frozen DREAM-BEV with PB-MHBR Stage-1 candidate re-ranking."""

    def __init__(self, logit_cap: float = 3.0, residual_scale: float = 1.0) -> None:
        super().__init__()
        if logit_cap <= 0.0 or residual_scale <= 0.0:
            raise ValueError("logit_cap and residual_scale must be positive")
        self.geometry = BEVGeometry()
        self.logit_cap = float(logit_cap)
        self.residual_scale = float(residual_scale)
        self.backbone = DREAMBEV()
        self.expert = PBMHBRStage1CandidateReranker(self.geometry)
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "DREAMDryComplexExpertBEV":
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
            "stage": "pb_mhbr_stage1",
            "logit_cap": self.logit_cap,
            "residual_scale": self.residual_scale,
            "expert": self.expert.state_dict(),
        }

    def forward(
        self,
        radar: Tensor,
        *,
        rd: Tensor | None = None,
        rd_mti: Tensor | None = None,
    ) -> dict[str, Tensor]:
        self.backbone.eval()
        backbone_radar = fixed_backbone_window(radar)
        with torch.no_grad():
            polar_feature = self.backbone.forward_polar_feature(backbone_radar)
            shared_feature = self.backbone.shared_from_polar_feature(polar_feature)
            base_logits = self.backbone.decoder.decode_shared(shared_feature)
            previous_radar = causal_previous_window(backbone_radar)
            previous_feature = self.backbone.forward_polar_feature(previous_radar)
            previous_logits = self.backbone.decoder.decode_shared(self.backbone.shared_from_polar_feature(previous_feature))
        expert_output = self.expert(
            polar_feature,
            shared_feature,
            radar,
            base_logits,
            previous_logits,
            rd=rd,
            rd_mti=rd_mti,
        )
        delta = (self.residual_scale * expert_output["delta_bev_logit"]).clamp(-self.logit_cap, self.logit_cap)
        learned_acceptance = torch.sigmoid(expert_output["acceptance_logit"].float())
        acceptance = expert_output["stage1_acceptance"].float()
        return {
            "logits": base_logits + acceptance[:, :, None, None] * delta,
            "base_logits": base_logits,
            "delta_bev_logit": delta,
            "acceptance": acceptance,
            "learned_acceptance": learned_acceptance,
            "shared_feature": shared_feature,
            **expert_output,
        }


class DREAMDryComplexTemporalBEV(DREAMDryComplexExpertBEV):
    """Frozen DREAM-BEV with causal temporal multi-branch candidate evidence."""

    def __init__(self, logit_cap: float = 3.0, residual_scale: float = 1.0) -> None:
        super().__init__(logit_cap=logit_cap, residual_scale=residual_scale)
        self.expert = PBMHBRStage2CandidateReranker(self.geometry)
        self._freeze_backbone()

    def expert_state_dict(self) -> dict[str, object]:
        return {
            "stage": "pb_mhbr_stage2",
            "logit_cap": self.logit_cap,
            "residual_scale": self.residual_scale,
            "expert": self.expert.state_dict(),
        }


class DREAMDryComplexTemporalRDBEV(DREAMDryComplexTemporalBEV):
    """PB-MHBR Stage-2 with candidate-level aligned RD/RD-MTI evidence."""

    def __init__(self, logit_cap: float = 3.0, residual_scale: float = 1.0) -> None:
        super().__init__(logit_cap=logit_cap, residual_scale=residual_scale)
        self.expert = PBMHBRStage2RDCandidateReranker(self.geometry)
        self._freeze_backbone()

    def expert_state_dict(self) -> dict[str, object]:
        return {
            "stage": "dry_complex_refinement",
            "logit_cap": self.logit_cap,
            "residual_scale": self.residual_scale,
            "expert": self.expert.state_dict(),
        }


class DREAMDryComplexQualityBEV(DREAMDryComplexTemporalBEV):
    """PB-MHBR quality stage with exact coordinate preservation."""

    def __init__(
        self,
        logit_cap: float = 3.0,
        residual_scale: float = 1.0,
        use_rd: bool = False,
    ) -> None:
        super().__init__(logit_cap=logit_cap, residual_scale=residual_scale)
        self.expert = PBMHBRQualityCandidateReranker(self.geometry, use_rd=use_rd)
        self._freeze_backbone()
        self.use_rd = bool(use_rd)

    def expert_state_dict(self) -> dict[str, object]:
        return {
            "stage": "pb_mhbr_quality_rd" if self.use_rd else "pb_mhbr_quality",
            "logit_cap": self.logit_cap,
            "residual_scale": self.residual_scale,
            "expert": self.expert.state_dict(),
        }

    def forward(
        self,
        radar: Tensor,
        *,
        rd: Tensor | None = None,
        rd_mti: Tensor | None = None,
    ) -> dict[str, Tensor]:
        output = super().forward(radar, rd=rd, rd_mti=rd_mti)
        quality_logit = output["correctness_logit"]
        output["logits"] = calibrate_peak_confidence(
            output["base_logits"],
            quality_logit,
        )
        output["acceptance"] = quality_logit.sigmoid()
        return output


class DREAMDryComplexQualityRDBEV(DREAMDryComplexQualityBEV):
    """Quality calibration with aligned RD/RD-MTI candidate evidence."""

    def __init__(self, logit_cap: float = 3.0, residual_scale: float = 1.0) -> None:
        super().__init__(
            logit_cap=logit_cap,
            residual_scale=residual_scale,
            use_rd=True,
        )



