from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional

from .contracts import RVBearingOutput


class ProposalGeometryRVBearingNet(nn.Module):
    """Frozen S1 proposals plus a causal geometry-to-bearing calibrator."""

    def __init__(
        self,
        *,
        evidence_width: float = 1.25,
        evidence_token_dim: int = 128,
        evidence_top_k: int = 3,
        evidence_quality_head: bool = False,
        angle_bins: int = 181,
        half_fov_rad: float = 0.80,
        temporal_dim: int = 96,
        minimum_sigma_rad: float = math.radians(0.25),
        max_missing_age: int = 3,
        missing_age_tau: float = 2.0,
    ) -> None:
        super().__init__()
        try:
            from rv_evidence_net.model import RVEvidenceNetSTMamba
        except ImportError as error:
            raise ImportError(
                "ProposalGeometryRVBearingNet requires sibling "
                "the local rv_evidence_net package on PYTHONPATH"
            ) from error
        if evidence_top_k < 1:
            raise ValueError("evidence_top_k must be positive")
        self.evidence_width = float(evidence_width)
        self.evidence_token_dim = int(evidence_token_dim)
        self.evidence_top_k = int(evidence_top_k)
        self.evidence_quality_head = bool(evidence_quality_head)
        self.angle_bins = int(angle_bins)
        self.half_fov_rad = float(half_fov_rad)
        self.temporal_dim = int(temporal_dim)
        self.minimum_sigma_rad = float(minimum_sigma_rad)
        self.max_missing_age = int(max_missing_age)
        self.missing_age_tau = float(missing_age_tau)

        self.detector = RVEvidenceNetSTMamba(
            width=self.evidence_width,
            token_dim=self.evidence_token_dim,
            top_k=self.evidence_top_k,
            quality_head=self.evidence_quality_head,
        )
        # Only the spatial detector and proposal adapter are used here.
        self.detector.recovery = nn.Identity()
        self.detector.requires_grad_(False)
        self.detector.eval()

        candidate_input_dim = self.evidence_token_dim + 6
        self.candidate_projection = nn.Sequential(
            nn.Linear(candidate_input_dim, temporal_dim),
            nn.SiLU(),
            nn.LayerNorm(temporal_dim),
        )
        self.candidate_attention = nn.Sequential(
            nn.Linear(candidate_input_dim, temporal_dim // 2),
            nn.SiLU(),
            nn.Linear(temporal_dim // 2, 1),
        )
        nn.init.zeros_(self.candidate_attention[-1].weight)
        nn.init.zeros_(self.candidate_attention[-1].bias)
        self.temporal_cell = nn.GRUCell(temporal_dim + 5, temporal_dim)
        self.theta_head = nn.Sequential(
            nn.Linear(temporal_dim + 5, temporal_dim),
            nn.SiLU(),
            nn.Linear(temporal_dim, 1),
        )
        self.sigma_head = nn.Linear(temporal_dim, 1)
        self.reliability_head = nn.Sequential(
            nn.Linear(temporal_dim + 3, temporal_dim),
            nn.SiLU(),
            nn.Linear(temporal_dim, 1),
        )
        self.register_buffer(
            "angle_centers",
            torch.linspace(
                -self.half_fov_rad,
                self.half_fov_rad,
                self.angle_bins,
            ),
            persistent=True,
        )
        self.register_buffer(
            "person_u_centers",
            torch.linspace(0.0, 1.0, self.angle_bins),
            persistent=True,
        )

    @property
    def proposal_contract(self) -> dict[str, object]:
        return {
            "evidence_width": self.evidence_width,
            "evidence_token_dim": self.evidence_token_dim,
            "evidence_top_k": self.evidence_top_k,
            "evidence_quality_head": self.evidence_quality_head,
            "angle_bins": self.angle_bins,
            "half_fov_rad": self.half_fov_rad,
            "temporal_dim": self.temporal_dim,
        }

    @classmethod
    def from_contract(
        cls,
        contract: Mapping[str, object],
    ) -> ProposalGeometryRVBearingNet:
        return cls(
            evidence_width=float(contract.get("evidence_width", 1.25)),
            evidence_token_dim=int(
                contract.get("evidence_token_dim", 128)
            ),
            evidence_top_k=int(contract.get("evidence_top_k", 3)),
            evidence_quality_head=bool(
                contract.get("evidence_quality_head", False)
            ),
            angle_bins=int(contract.get("angle_bins", 181)),
            half_fov_rad=float(contract.get("half_fov_rad", 0.80)),
            temporal_dim=int(contract.get("temporal_dim", 96)),
        )

    def load_evidence_state(
        self,
        payload: Mapping[str, object],
    ) -> None:
        raw_state = payload.get("model", payload)
        if not isinstance(raw_state, Mapping):
            raise TypeError("evidence checkpoint must contain a model state")
        expected = self.detector.state_dict()
        spatial_state = {
            str(name): value
            for name, value in raw_state.items()
            if str(name) in expected and isinstance(value, Tensor)
        }
        missing = sorted(set(expected) - set(spatial_state))
        if missing:
            raise ValueError(
                "evidence checkpoint lacks spatial parameters: "
                + ", ".join(missing[:5])
            )
        self.detector.load_state_dict(spatial_state, strict=True)
        self.detector.requires_grad_(False)
        self.detector.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.detector.eval()
        return self

    @staticmethod
    def _proposal_geometry(
        boxes: Tensor,
        image_height: int,
        image_width: int,
    ) -> Tensor:
        center_u = (boxes[..., 0] + boxes[..., 2]) * 0.5 / image_width
        top_v = boxes[..., 1] / image_height
        width = (boxes[..., 2] - boxes[..., 0]).clamp_min(0.0) / image_width
        height = (
            (boxes[..., 3] - boxes[..., 1]).clamp_min(0.0)
            / image_height
        )
        return torch.stack((center_u, top_v, width, height), dim=-1)

    def _encode_proposals(
        self,
        *,
        boxes: Tensor,
        scores: Tensor,
        tokens: Tensor,
        valid: Tensor,
        image_height: int,
        image_width: int,
        quality: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        geometry = self._proposal_geometry(
            boxes,
            image_height,
            image_width,
        )
        scores = scores.clamp(0.0, 1.0)
        if quality is None:
            quality = torch.ones_like(scores)
        else:
            quality = quality.clamp(0.0, 1.0)
        candidate_values = torch.cat(
            (
                tokens,
                geometry,
                scores.unsqueeze(-1),
                quality.unsqueeze(-1),
            ),
            dim=-1,
        )
        embeddings = self.candidate_projection(candidate_values)
        attention_logits = self.candidate_attention(
            candidate_values
        ).squeeze(-1) + scores.clamp_min(1e-6).log()
        attention_logits = attention_logits.masked_fill(~valid, -1e4)
        weights = attention_logits.softmax(dim=1)
        observed = valid.any(dim=1)
        weights = torch.where(
            observed[:, None],
            weights,
            torch.zeros_like(weights),
        )
        pooled_embedding = (weights.unsqueeze(-1) * embeddings).sum(dim=1)
        pooled_geometry = (weights.unsqueeze(-1) * geometry).sum(dim=1)
        pooled_score = (weights * scores).sum(dim=1, keepdim=True)
        return (
            pooled_embedding,
            torch.cat((pooled_geometry, pooled_score), dim=-1),
            observed,
        )

    def _encode_frame(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            output = self.detector.forward_frame(
                images,
                extract_evidence=True,
                detection_score_mode="objectness",
            )
        return self._encode_proposals(
            boxes=output.boxes,
            scores=output.raw_scores,
            tokens=output.raw_tokens,
            valid=output.valid_mask,
            quality=output.proposal_quality,
            image_height=images.shape[-2],
            image_width=images.shape[-1],
        )

    def _causal_state_sequence(
        self,
        images: Tensor,
        frame_valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, time_steps = images.shape[:2]
        hidden = images.new_zeros(batch, self.temporal_dim)
        held_geometry = images.new_zeros(batch, 5)
        ever_valid = torch.zeros(batch, dtype=torch.bool, device=images.device)
        age = images.new_full((batch,), float(time_steps + 1))
        hidden_history: list[Tensor] = []
        geometry_history: list[Tensor] = []
        age_history: list[Tensor] = []
        valid_history: list[Tensor] = []
        for time_index in range(time_steps):
            embedding, geometry, detected = self._encode_frame(
                images[:, time_index]
            )
            observed = frame_valid[:, time_index] & detected
            proposal = self.temporal_cell(
                torch.cat((embedding, geometry), dim=1),
                hidden,
            )
            hidden = torch.where(observed[:, None], proposal, hidden)
            held_geometry = torch.where(
                observed[:, None],
                geometry,
                held_geometry,
            )
            age = torch.where(observed, torch.zeros_like(age), age + 1.0)
            ever_valid = ever_valid | observed
            hidden_history.append(hidden)
            geometry_history.append(held_geometry)
            age_history.append(age)
            valid_history.append(
                ever_valid & (age <= self.max_missing_age)
            )
        return (
            torch.stack(hidden_history, dim=1),
            torch.stack(geometry_history, dim=1),
            torch.stack(age_history, dim=1),
            torch.stack(valid_history, dim=1),
        )

    def _causal_cached_state_sequence(
        self,
        *,
        boxes: Tensor,
        scores: Tensor,
        tokens: Tensor,
        valid_mask: Tensor,
        frame_valid: Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if boxes.ndim != 4 or boxes.shape[-1] != 4:
            raise ValueError("boxes must have shape [batch,time,k,4]")
        batch, time_steps, candidates = boxes.shape[:3]
        expected_scores = (batch, time_steps, candidates)
        expected_tokens = (
            batch,
            time_steps,
            candidates,
            self.evidence_token_dim,
        )
        if tuple(scores.shape) != expected_scores:
            raise ValueError(
                f"scores must have shape {expected_scores}"
            )
        if tuple(valid_mask.shape) != expected_scores:
            raise ValueError(
                f"valid_mask must have shape {expected_scores}"
            )
        if tuple(tokens.shape) != expected_tokens:
            raise ValueError(
                f"tokens must have shape {expected_tokens}"
            )
        if tuple(frame_valid.shape) != (batch, time_steps):
            raise ValueError("frame_valid must have shape [batch,time]")

        hidden = boxes.new_zeros(batch, self.temporal_dim)
        held_geometry = boxes.new_zeros(batch, 5)
        ever_valid = torch.zeros(
            batch,
            dtype=torch.bool,
            device=boxes.device,
        )
        age = boxes.new_full((batch,), float(time_steps + 1))
        hidden_history: list[Tensor] = []
        geometry_history: list[Tensor] = []
        age_history: list[Tensor] = []
        valid_history: list[Tensor] = []
        for time_index in range(time_steps):
            embedding, geometry, detected = self._encode_proposals(
                boxes=boxes[:, time_index],
                scores=scores[:, time_index],
                tokens=tokens[:, time_index],
                valid=valid_mask[:, time_index],
                image_height=image_height,
                image_width=image_width,
            )
            observed = frame_valid[:, time_index] & detected
            proposal = self.temporal_cell(
                torch.cat((embedding, geometry), dim=1),
                hidden,
            )
            hidden = torch.where(observed[:, None], proposal, hidden)
            held_geometry = torch.where(
                observed[:, None],
                geometry,
                held_geometry,
            )
            age = torch.where(observed, torch.zeros_like(age), age + 1.0)
            ever_valid = ever_valid | observed
            hidden_history.append(hidden)
            geometry_history.append(held_geometry)
            age_history.append(age)
            valid_history.append(
                ever_valid & (age <= self.max_missing_age)
            )
        return (
            torch.stack(hidden_history, dim=1),
            torch.stack(geometry_history, dim=1),
            torch.stack(age_history, dim=1),
            torch.stack(valid_history, dim=1),
        )

    def _decode_state(
        self,
        hidden: Tensor,
        geometry: Tensor,
        age: Tensor,
        valid: Tensor,
    ) -> RVBearingOutput:
        theta = self.half_fov_rad * torch.tanh(
            self.theta_head(torch.cat((hidden, geometry), dim=1))
        )
        sigma = (
            self.minimum_sigma_rad
            + 0.02 * functional.softplus(self.sigma_head(hidden))
        )
        centers = self.angle_centers.to(theta)
        bearing_logits = -0.5 * (
            (centers[None] - theta) / sigma.clamp_min(1e-4)
        ).square()
        person_u = geometry[:, :1].clamp(0.0, 1.0)
        person_u_logits = -0.5 * (
            (self.person_u_centers.to(theta)[None] - person_u) / 0.025
        ).square()
        score = geometry[:, 4:5].clamp(0.0, 1.0)
        reliability = torch.sigmoid(
            self.reliability_head(
                torch.cat(
                    (
                        hidden,
                        score,
                        age[:, None] / max(self.missing_age_tau, 1e-6),
                        valid[:, None].to(hidden.dtype),
                    ),
                    dim=1,
                )
            )
        )
        reliability = (
            reliability
            * torch.exp(-age[:, None] / self.missing_age_tau)
            * valid[:, None].to(reliability.dtype)
        )
        output = RVBearingOutput(
            bearing_logits=bearing_logits,
            theta=theta,
            sigma_theta=sigma,
            reliability=reliability,
            temporal_feature=hidden,
            person_u_logits=person_u_logits,
            valid=valid,
        )
        output.validate(
            batch_size=hidden.shape[0],
            angle_bins=self.angle_bins,
            temporal_dim=self.temporal_dim,
        )
        return output

    def _validated_frame_mask(
        self,
        images: Tensor,
        frame_valid: Tensor | None,
    ) -> Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(
                "images must have shape [batch,time,3,height,width]"
            )
        batch, time_steps = images.shape[:2]
        if frame_valid is None:
            return torch.ones(
                batch,
                time_steps,
                dtype=torch.bool,
                device=images.device,
            )
        if tuple(frame_valid.shape) != (batch, time_steps):
            raise ValueError("frame_valid must have shape [batch,time]")
        return frame_valid

    def forward(
        self,
        images: Tensor,
        *,
        frame_valid: Tensor | None = None,
        output_time_index: int | None = None,
    ) -> RVBearingOutput:
        frame_valid = self._validated_frame_mask(images, frame_valid)
        if output_time_index is None:
            output_time_index = images.shape[1] - 1
        hidden, geometry, age, valid = self._causal_state_sequence(
            images,
            frame_valid,
        )
        return self._decode_state(
            hidden[:, output_time_index],
            geometry[:, output_time_index],
            age[:, output_time_index],
            valid[:, output_time_index],
        )

    def forward_pair(
        self,
        images: Tensor,
        *,
        frame_valid: Tensor | None = None,
    ) -> tuple[RVBearingOutput, RVBearingOutput]:
        frame_valid = self._validated_frame_mask(images, frame_valid)
        if images.shape[1] < 2:
            raise ValueError("forward_pair requires at least two frames")
        hidden, geometry, age, valid = self._causal_state_sequence(
            images,
            frame_valid,
        )
        return (
            self._decode_state(
                hidden[:, -2],
                geometry[:, -2],
                age[:, -2],
                valid[:, -2],
            ),
            self._decode_state(
                hidden[:, -1],
                geometry[:, -1],
                age[:, -1],
                valid[:, -1],
            ),
        )

    def forward_cached_pair(
        self,
        *,
        boxes: Tensor,
        scores: Tensor,
        tokens: Tensor,
        valid_mask: Tensor,
        image_height: int,
        image_width: int,
        frame_valid: Tensor | None = None,
    ) -> tuple[RVBearingOutput, RVBearingOutput]:
        if boxes.shape[1] < 2:
            raise ValueError(
                "forward_cached_pair requires at least two frames"
            )
        if frame_valid is None:
            frame_valid = torch.ones(
                boxes.shape[:2],
                dtype=torch.bool,
                device=boxes.device,
            )
        hidden, geometry, age, valid = (
            self._causal_cached_state_sequence(
                boxes=boxes,
                scores=scores,
                tokens=tokens,
                valid_mask=valid_mask,
                frame_valid=frame_valid,
                image_height=image_height,
                image_width=image_width,
            )
        )
        return (
            self._decode_state(
                hidden[:, -2],
                geometry[:, -2],
                age[:, -2],
                valid[:, -2],
            ),
            self._decode_state(
                hidden[:, -1],
                geometry[:, -1],
                age[:, -1],
                valid[:, -1],
            ),
        )

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )



