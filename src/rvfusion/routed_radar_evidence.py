from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from .contracts import RadarEvidence


def _normalized_route(route: object) -> str:
    return str(route).strip().lower()


class RoutedRadarEvidenceAdapter(nn.Module):
    """Route frozen radar windows through optional specialist evidence modules."""

    def __init__(
        self,
        base_adapter: nn.Module,
        route_experts: Mapping[str, nn.Module],
    ) -> None:
        super().__init__()
        self.base_adapter = base_adapter
        self.route_experts = nn.ModuleDict(
            {
                _normalized_route(route): module
                for route, module in route_experts.items()
            }
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "RoutedRadarEvidenceAdapter":
        super().train(False)
        self.base_adapter.eval()
        self.route_experts.eval()
        return self

    @staticmethod
    def _expert_evidence(output: object, batch: int) -> RadarEvidence:
        if not isinstance(output, dict):
            raise TypeError("route expert must return a tensor dictionary")
        required = ("polar_feature", "shared_feature", "logits")
        missing = [name for name in required if name not in output]
        if missing:
            raise KeyError(f"route expert output is missing: {missing}")
        evidence = RadarEvidence(
            polar_feature=output["polar_feature"],
            bev_feature=output["shared_feature"],
            logits=output["logits"],
            position_override_xy_m=output.get("position_override_xy_m"),
            position_override_mask=output.get("position_override_mask"),
        )
        evidence.validate()
        if evidence.logits.shape[0] != batch:
            raise ValueError("route expert output batch does not match input")
        return evidence

    def reset_causal_state(self) -> None:
        for expert in self.route_experts.values():
            reset = getattr(expert, "reset_causal_state", None)
            if reset is not None:
                reset()

    def forward(
        self,
        radar: Tensor,
        routes: Sequence[str],
        *,
        expert_bypass_mask: Tensor | None = None,
        sequences: Sequence[str] | None = None,
        frame_ids: Sequence[int] | None = None,
    ) -> RadarEvidence:
        batch = radar.shape[0]
        if len(routes) != batch:
            raise ValueError("route count must match radar batch")
        if expert_bypass_mask is None:
            expert_bypass_mask = torch.zeros(
                batch,
                device=radar.device,
                dtype=torch.bool,
            )
        if (
            tuple(expert_bypass_mask.shape) != (batch,)
            or expert_bypass_mask.dtype != torch.bool
        ):
            raise ValueError(
                "expert_bypass_mask must be boolean [batch]"
            )
        expert_bypass_mask = expert_bypass_mask.to(device=radar.device)
        bypass = expert_bypass_mask.detach().cpu().tolist()
        with torch.no_grad(), torch.autocast(
            device_type=radar.device.type,
            enabled=False,
        ):
            radar_fp32 = radar.float()
            base = self.base_adapter(radar_fp32)
            if not isinstance(base, RadarEvidence):
                raise TypeError("base adapter must return RadarEvidence")
            base.validate()
            if not self.route_experts:
                return base

            polar = base.polar_feature.clone()
            bev = base.bev_feature.clone()
            logits = base.logits.clone()
            position_override_xy_m = (
                None
                if base.position_override_xy_m is None
                else base.position_override_xy_m.clone()
            )
            position_override_mask = (
                None
                if base.position_override_mask is None
                else base.position_override_mask.clone()
            )
            normalized = tuple(_normalized_route(route) for route in routes)
            for route, expert in self.route_experts.items():
                indices = [
                    index
                    for index, sample_route in enumerate(normalized)
                    if (
                        sample_route == route
                        and not bypass[index]
                    )
                ]
                if not indices:
                    continue
                index = torch.tensor(
                    indices,
                    device=radar.device,
                    dtype=torch.long,
                )
                selected_radar = radar_fp32.index_select(0, index)
                if bool(getattr(expert, "requires_causal_metadata", False)):
                    if sequences is None or frame_ids is None:
                        raise ValueError(
                            "causal route expert requires sequences and frame_ids"
                        )
                    if len(sequences) != batch or len(frame_ids) != batch:
                        raise ValueError(
                            "sequence and frame metadata must match radar batch"
                        )
                    expert_raw = expert(
                        selected_radar,
                        sequences=tuple(str(sequences[item]) for item in indices),
                        frame_ids=tuple(int(frame_ids[item]) for item in indices),
                    )
                else:
                    expert_raw = expert(selected_radar)
                expert_output = self._expert_evidence(expert_raw, len(indices))
                polar.index_copy_(
                    0,
                    index,
                    expert_output.polar_feature.to(
                        device=polar.device,
                        dtype=polar.dtype,
                    ),
                )
                bev.index_copy_(
                    0,
                    index,
                    expert_output.bev_feature.to(
                        device=bev.device,
                        dtype=bev.dtype,
                    ),
                )
                logits.index_copy_(
                    0,
                    index,
                    expert_output.logits.to(
                        device=logits.device,
                        dtype=logits.dtype,
                    ),
                )
                if expert_output.position_override_xy_m is not None:
                    assert expert_output.position_override_mask is not None
                    if position_override_xy_m is None:
                        position_override_xy_m = (
                            expert_output.position_override_xy_m.new_zeros(
                                batch,
                                2,
                            )
                        )
                        position_override_mask = torch.zeros(
                            batch,
                            device=logits.device,
                            dtype=torch.bool,
                        )
                    assert position_override_mask is not None
                    position_override_xy_m.index_copy_(
                        0,
                        index,
                        expert_output.position_override_xy_m.to(
                            device=position_override_xy_m.device,
                            dtype=position_override_xy_m.dtype,
                        ),
                    )
                    position_override_mask.index_copy_(
                        0,
                        index,
                        expert_output.position_override_mask.to(
                            device=position_override_mask.device,
                        ),
                    )
        result = RadarEvidence(
            polar_feature=polar,
            bev_feature=bev,
            logits=logits,
            position_override_xy_m=position_override_xy_m,
            position_override_mask=position_override_mask,
        )
        result.validate()
        return result



