"""Bridge the five paper recovery experts onto frozen CRUW DREAM-BEV evidence.

The public transfer keeps the original expert bodies and their bounded-output
contract, while obtaining polar/BEV evidence from the CRUW-adapted frozen
anchor.  It deliberately does not load or modify a private backbone checkpoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from .allscene import CRUWBEVGeometry


PaperExpertKind = Literal["clor", "tecs", "ltge_cen", "pb_mhbr", "gg_srre"]
PAPER_EXPERT_KINDS: tuple[PaperExpertKind, ...] = (
    "clor",
    "tecs",
    "ltge_cen",
    "pb_mhbr",
    "gg_srre",
)


def decode_paper_expert_output_xy(
    output: dict[str, torch.Tensor], geometry: CRUWBEVGeometry
) -> torch.Tensor:
    """Decode the deployed paper-expert output map into its metric BEV peak.

    ``proposal_xy_m`` is an internal training/candidate signal.  Downstream
    fusion must instead start from the final bounded expert logit map, exactly
    as the standalone paper-expert evaluation does.
    """

    logits = output["logits"]
    if tuple(logits.shape[1:]) != (1, geometry.x_bins, geometry.y_bins):
        raise ValueError("paper expert logits do not match the canonical CRUW BEV geometry")
    peak = logits.flatten(1).argmax(dim=1)
    x_index = peak // geometry.y_bins
    y_index = peak % geometry.y_bins
    return torch.stack(
        (
            geometry.x_min_m + x_index.to(logits.dtype) * geometry.resolution_m,
            geometry.y_min_m + y_index.to(logits.dtype) * geometry.resolution_m,
        ),
        dim=1,
    )


def _paper_source_root() -> Path:
    configured = os.environ.get("DREAM_BEV_PAPER_SOURCE_ROOT")
    root = Path(configured) if configured else Path(__file__).resolve().parents[3] / "DREAM-BEV_Posterior_DevelopmentStudy" / "src"
    if not (root / "radar_bev").is_dir():
        raise FileNotFoundError(
            "paper expert source is unavailable; expected "
            f"{root / 'radar_bev'} or set DREAM_BEV_PAPER_SOURCE_ROOT"
        )
    return root


def _import_paper_experts():
    source_root = _paper_source_root()
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from radar_bev.expert_primitives import decode_peak_xy
    from radar_bev.geometry import BEVGeometry
    from radar_bev.models.clutter_expert import TECSEchoCandidateSelector
    from radar_bev.models.dry_complex_expert import PBMHBRStage1CandidateReranker
    from radar_bev.models.heterogeneous_experts import CLOREXpert
    from radar_bev.models.recovery_experts import GGSRREExpert
    from radar_bev.models.weak_return_expert import LTGECENExpert

    return (
        BEVGeometry,
        decode_peak_xy,
        CLOREXpert,
        TECSEchoCandidateSelector,
        LTGECENExpert,
        PBMHBRStage1CandidateReranker,
        GGSRREExpert,
    )


def paper_expert_target_indices(
    candidate_xy_m: torch.Tensor,
    target_xy_m: torch.Tensor,
    *,
    minimum_improvement_m: float = 0.05,
) -> torch.Tensor:
    """Keep candidate zero unless another observed candidate materially improves."""
    if candidate_xy_m.ndim != 3 or candidate_xy_m.shape[:1] != target_xy_m.shape[:1]:
        raise ValueError("candidate coordinates and targets must share the batch axis")
    errors = torch.linalg.vector_norm(candidate_xy_m - target_xy_m[:, None], dim=-1)
    best = errors.argmin(dim=1)
    improvement = errors[:, 0] - errors.gather(1, best[:, None]).squeeze(1)
    return torch.where(improvement >= minimum_improvement_m, best, torch.zeros_like(best))


class CRUWPaperExpert(nn.Module):
    """One paper specialist driven by an externally frozen CRUW anchor."""

    def __init__(self, expert_kind: PaperExpertKind, geometry: CRUWBEVGeometry) -> None:
        super().__init__()
        if expert_kind not in PAPER_EXPERT_KINDS:
            raise ValueError(f"unsupported paper expert kind={expert_kind!r}")
        (
            paper_geometry_type,
            decode_peak_xy,
            clor_type,
            tecs_type,
            ltge_type,
            pbmhbr_type,
            gg_srre_type,
        ) = _import_paper_experts()
        self.expert_kind = expert_kind
        self.geometry = geometry
        self.paper_geometry = paper_geometry_type()
        for attribute in ("x_min_m", "x_max_m", "y_min_m", "y_max_m", "resolution_m"):
            if float(getattr(self.paper_geometry, attribute)) != float(getattr(geometry, attribute)):
                raise ValueError(f"CRUW geometry does not match paper expert geometry at {attribute}")
        self._decode_peak_xy = decode_peak_xy
        if expert_kind == "clor":
            self.expert: nn.Module = clor_type(self.paper_geometry)
            self.logit_cap = 3.0
        elif expert_kind == "tecs":
            self.expert = tecs_type(self.paper_geometry)
            self.logit_cap = 3.5
        elif expert_kind == "ltge_cen":
            self.expert = ltge_type(self.paper_geometry, candidate_count=8, rd_mode="off")
            self.logit_cap = 3.0
        elif expert_kind == "pb_mhbr":
            self.expert = pbmhbr_type(self.paper_geometry)
            self.logit_cap = 3.0
        else:
            self.expert = gg_srre_type(self.paper_geometry)
            self.logit_cap = 3.0

    def _peaks(self, base_logits: torch.Tensor, previous_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_xy_m, _ = self._decode_peak_xy(base_logits, self.paper_geometry)
        previous_xy_m, _ = self._decode_peak_xy(previous_logits, self.paper_geometry)
        return base_xy_m, previous_xy_m

    def forward(
        self,
        *,
        polar_feature: torch.Tensor,
        shared_feature: torch.Tensor,
        radar: torch.Tensor,
        base_logits: torch.Tensor,
        previous_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base_xy_m, previous_xy_m = self._peaks(base_logits, previous_logits)
        if self.expert_kind == "pb_mhbr":
            raw = self.expert(
                polar_feature, shared_feature, radar, base_logits, previous_logits
            )
        else:
            raw = self.expert(
                polar_feature,
                shared_feature,
                radar,
                base_logits,
                base_xy_m,
                previous_xy_m,
            )
        delta = raw["delta_bev_logit"].clamp(-self.logit_cap, self.logit_cap)
        if self.expert_kind == "pb_mhbr":
            acceptance = raw["stage1_acceptance"]
        else:
            acceptance = raw.get("acceptance_override")
            if acceptance is None:
                acceptance = torch.sigmoid(raw["acceptance_logit"])
        candidate_xy_m = raw.get("candidate_xy_m")
        if candidate_xy_m is None:
            candidate_xy_m = torch.stack((base_xy_m, raw["proposal_xy_m"]), dim=1)
        candidate_logits = raw.get("candidate_selector_logits")
        if candidate_logits is None:
            mixture = raw.get("candidate_mixture")
            candidate_logits = (
                mixture.clamp_min(1e-8).log()
                if mixture is not None
                else torch.stack(
                    (torch.zeros_like(raw["acceptance_logit"][:, 0]), raw["acceptance_logit"][:, 0]),
                    dim=1,
                )
            )
        return {
            "logits": base_logits + acceptance[:, :, None, None] * delta,
            "base_logits": base_logits,
            "delta_bev_logit": delta,
            "acceptance": acceptance,
            "acceptance_logit": raw["acceptance_logit"],
            "proposal_xy_m": raw["proposal_xy_m"],
            "base_xy_m": base_xy_m,
            "previous_xy_m": previous_xy_m,
            "candidate_xy_m": candidate_xy_m,
            "candidate_selector_logits": candidate_logits,
            **{
                key: value
                for key, value in raw.items()
                if key
                not in {
                    "delta_bev_logit",
                    "acceptance_logit",
                    "acceptance_override",
                    "proposal_xy_m",
                    "candidate_xy_m",
                    "candidate_selector_logits",
                }
            },
        }
