from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional

from .contracts import RVBearingOutput
from .proposal_geometry import ProposalGeometryRVBearingNet


class MonotonicResidualRVBearingNet(ProposalGeometryRVBearingNet):
    """Monotonic image-bearing prior with a bounded causal correction."""

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
        max_residual_rad: float = 0.12,
    ) -> None:
        super().__init__(
            evidence_width=evidence_width,
            evidence_token_dim=evidence_token_dim,
            evidence_top_k=evidence_top_k,
            evidence_quality_head=evidence_quality_head,
            angle_bins=angle_bins,
            half_fov_rad=half_fov_rad,
            temporal_dim=temporal_dim,
            minimum_sigma_rad=minimum_sigma_rad,
            max_missing_age=max_missing_age,
            missing_age_tau=missing_age_tau,
        )
        if max_residual_rad <= 0.0:
            raise ValueError("max_residual_rad must be positive")
        self.max_residual_rad = float(max_residual_rad)
        self.principal_u_raw = nn.Parameter(torch.zeros(()))
        self.focal_raw = nn.Parameter(torch.zeros(()))
        self.theta_head = nn.Sequential(
            nn.Linear(temporal_dim + 5, temporal_dim),
            nn.SiLU(),
            nn.Linear(temporal_dim, 1),
        )
        nn.init.zeros_(self.theta_head[-1].weight)
        nn.init.zeros_(self.theta_head[-1].bias)

    @property
    def proposal_contract(self) -> dict[str, object]:
        contract = super().proposal_contract
        contract["max_residual_rad"] = self.max_residual_rad
        return contract

    @classmethod
    def from_contract(
        cls,
        contract: Mapping[str, object],
    ) -> MonotonicResidualRVBearingNet:
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
            max_residual_rad=float(
                contract.get("max_residual_rad", 0.12)
            ),
        )

    def monotonic_prior(self, person_u: Tensor) -> Tensor:
        principal_u = 0.35 + 0.30 * torch.sigmoid(self.principal_u_raw)
        focal = 0.35 + 0.65 * torch.sigmoid(self.focal_raw)
        return self.half_fov_rad * torch.tanh(
            (person_u - principal_u) / focal
        )

    def bounded_residual(self, hidden: Tensor, geometry: Tensor) -> Tensor:
        return self.max_residual_rad * torch.tanh(
            self.theta_head(torch.cat((hidden, geometry), dim=1))
        )

    def _decode_state(
        self,
        hidden: Tensor,
        geometry: Tensor,
        age: Tensor,
        valid: Tensor,
    ) -> RVBearingOutput:
        person_u = geometry[:, :1].clamp(0.0, 1.0)
        prior = self.monotonic_prior(person_u)
        residual = self.bounded_residual(hidden, geometry)
        theta = (prior + residual).clamp(
            -self.half_fov_rad,
            self.half_fov_rad,
        )
        sigma = (
            self.minimum_sigma_rad
            + 0.02 * functional.softplus(self.sigma_head(hidden))
        )
        centers = self.angle_centers.to(theta)
        bearing_logits = -0.5 * (
            (centers[None] - theta) / sigma.clamp_min(1e-4)
        ).square()
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



