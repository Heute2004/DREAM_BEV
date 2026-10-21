"""Fast radar--vision consistency checks for guarded lateral fusion."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def _require_shape(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must be {list(shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def radar_lateral_sigma(
    radar_probability: Tensor,
    radar_xy_m: Tensor,
    *,
    radius_m: float,
    minimum_sigma_m: float = 0.05,
    maximum_sigma_m: float = 0.50,
) -> Tensor:
    """Estimate local radar cross-range uncertainty at the frozen radar range.

    The estimate is the conditional standard deviation of the local BEV x
    profile. It is deliberately independent of the visual branch, so a broad
    or multi-peak radar slice raises the tolerance while a sharp radar peak
    remains protective against a displaced visual ray.
    """

    if radar_probability.ndim != 4 or tuple(radar_probability.shape[1:]) != (
        1,
        200,
        234,
    ):
        raise ValueError("radar_probability must be [batch,1,200,234]")
    batch = radar_probability.shape[0]
    _require_shape("radar_xy_m", radar_xy_m, (batch, 2))
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    if minimum_sigma_m <= 0.0 or maximum_sigma_m < minimum_sigma_m:
        raise ValueError("invalid radar sigma bounds")

    dtype = radar_probability.dtype
    device = radar_probability.device
    x_coordinates_m = (
        torch.arange(200, device=device, dtype=dtype) * 0.1 - 10.0
    )
    y_index = (
        radar_xy_m[:, 1].div(0.1).floor().long().clamp(0, 233)
    )
    batch_index = torch.arange(batch, device=device)
    x_profile = radar_probability[batch_index, 0, :, y_index]
    local_mask = (
        x_coordinates_m[None, :]
        - radar_xy_m[:, 0:1].to(dtype)
    ).abs() <= float(radius_m) + 1e-6
    local_weight = x_profile * local_mask.to(dtype)
    local_weight = local_weight / local_weight.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-12)
    mean_x_m = (local_weight * x_coordinates_m[None, :]).sum(
        dim=1,
        keepdim=True,
    )
    variance_m2 = (
        local_weight * (x_coordinates_m[None, :] - mean_x_m).square()
    ).sum(dim=1, keepdim=True)
    return variance_m2.sqrt().clamp(
        min=float(minimum_sigma_m),
        max=float(maximum_sigma_m),
    )


def cross_modal_innovation(
    *,
    visual_x_m: Tensor,
    visual_sigma_x_m: Tensor,
    radar_x_m: Tensor,
    radar_sigma_x_m: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    """Return the standardized lateral disagreement between visual and radar."""

    shape = tuple(visual_x_m.shape)
    if len(shape) != 2 or shape[1] != 1:
        raise ValueError("visual_x_m must be [batch,1]")
    for name, value in (
        ("visual_sigma_x_m", visual_sigma_x_m),
        ("radar_x_m", radar_x_m),
        ("radar_sigma_x_m", radar_sigma_x_m),
    ):
        _require_shape(name, value, shape)
    _require_shape("visual_x_m", visual_x_m, shape)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    variance = (
        visual_sigma_x_m.square()
        + radar_sigma_x_m.square()
        + float(epsilon)
    )
    return (visual_x_m - radar_x_m).abs() / variance.sqrt()


def should_accept_visual_evidence(
    innovation: Tensor,
    *,
    threshold: float,
) -> Tensor:
    """Fail closed when a visual ray disagrees with local radar evidence."""

    if not torch.isfinite(innovation).all():
        raise ValueError("innovation must be finite")
    if not torch.isfinite(torch.tensor(float(threshold))) or threshold <= 0.0:
        raise ValueError("threshold must be positive and finite")
    return innovation <= float(threshold)


class CausalSignedInnovationGuard:
    """Reject a persistent signed radar--vision disagreement causally.

    A one-frame discrepancy can arise from local radar ambiguity or visual
    jitter. This stateful guard therefore filters the *signed* standardized
    innovation per sequence and only rejects evidence after the disagreement
    has remained directionally consistent over past frames.
    """

    def __init__(
        self,
        *,
        ema_alpha: float,
        ema_threshold: float,
        instantaneous_threshold: float | None = None,
    ) -> None:
        if not 0.0 < float(ema_alpha) <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not torch.isfinite(torch.tensor(float(ema_threshold))) or ema_threshold <= 0.0:
            raise ValueError("ema_threshold must be positive and finite")
        if (
            instantaneous_threshold is not None
            and (
                not torch.isfinite(torch.tensor(float(instantaneous_threshold)))
                or instantaneous_threshold <= 0.0
            )
        ):
            raise ValueError("instantaneous_threshold must be positive and finite")
        self.ema_alpha = float(ema_alpha)
        self.ema_threshold = float(ema_threshold)
        self.instantaneous_threshold = (
            None
            if instantaneous_threshold is None
            else float(instantaneous_threshold)
        )
        self.reset()

    def reset(self) -> None:
        """Clear all sequence-local causal state."""

        self._ema_by_sequence: dict[str, float] = {}
        self._last_frame_by_sequence: dict[str, int] = {}

    def current_ema(self, sequence: str) -> float:
        """Return the latest causal innovation state for one sequence."""

        return self._ema_by_sequence.get(str(sequence), 0.0)

    def step(
        self,
        *,
        sequences: Sequence[str],
        frame_ids: Sequence[int],
        signed_innovation: Tensor,
        eligible: Tensor,
    ) -> Tensor:
        """Update from the current batch and return one causal accept mask.

        The caller must preserve frame order within each sequence. Frames not
        eligible for visual fusion do not update the evidence state and always
        pass; their policy-level identity fallback is applied elsewhere.
        """

        batch = len(sequences)
        if len(frame_ids) != batch:
            raise ValueError("frame_ids must align with sequences")
        _require_shape("signed_innovation", signed_innovation, (batch, 1))
        if tuple(eligible.shape) != (batch,) or eligible.dtype != torch.bool:
            raise ValueError("eligible must be boolean [batch]")

        passed = torch.ones(batch, device=signed_innovation.device, dtype=torch.bool)
        values = signed_innovation[:, 0].detach().cpu().tolist()
        eligible_values = eligible.detach().cpu().tolist()
        for index, (sequence, frame_id, innovation, is_eligible) in enumerate(
            zip(sequences, frame_ids, values, eligible_values, strict=True)
        ):
            key = str(sequence)
            frame = int(frame_id)
            previous = self._last_frame_by_sequence.get(key)
            if previous is not None and frame <= previous:
                raise ValueError("frame_ids must be strictly increasing per sequence")
            self._last_frame_by_sequence[key] = frame
            if not is_eligible:
                continue
            ema = (1.0 - self.ema_alpha) * self._ema_by_sequence.get(key, 0.0)
            ema += self.ema_alpha * float(innovation)
            self._ema_by_sequence[key] = ema
            persistent_pass = abs(ema) <= self.ema_threshold
            instantaneous_pass = (
                True
                if self.instantaneous_threshold is None
                else abs(float(innovation)) <= self.instantaneous_threshold
            )
            passed[index] = persistent_pass and instantaneous_pass
        return passed
