from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LocalXLattice:
    x_m: Tensor
    relative_x_m: Tensor
    valid_mask: Tensor
    source_id: Tensor
    radar_support: Tensor
    visual_log_likelihood: Tensor
    no_op_index: int = 0

    def validate(self) -> None:
        batch, candidates = self.x_m.shape
        expected = (batch, candidates)
        for name, value in (
            ("relative_x_m", self.relative_x_m),
            ("valid_mask", self.valid_mask),
            ("source_id", self.source_id),
            ("radar_support", self.radar_support),
            ("visual_log_likelihood", self.visual_log_likelihood),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must match x_m")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean")
        if self.source_id.dtype != torch.long:
            raise ValueError("source_id must be int64")
        if self.no_op_index != 0:
            raise ValueError("no-op must occupy index zero")
        if not self.valid_mask[:, 0].all():
            raise ValueError("no-op must always be valid")
        if not torch.equal(self.source_id[:, 0], torch.zeros(batch, dtype=torch.long, device=self.source_id.device)):
            raise ValueError("no-op source id must be zero")
        for name, value in (
            ("x_m", self.x_m),
            ("relative_x_m", self.relative_x_m),
            ("radar_support", self.radar_support),
            ("visual_log_likelihood", self.visual_log_likelihood),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")


def build_local_x_lattice(
    radar_probability: Tensor,
    radar_xy_m: Tensor,
    visual_x_m: Tensor,
    visual_sigma_x_m: Tensor,
    *,
    radius_m: float = 0.5,
    step_m: float = 0.025,
) -> LocalXLattice:
    if radar_probability.ndim != 4 or radar_probability.shape[1:] != (
        1,
        200,
        234,
    ):
        raise ValueError("radar_probability must be [batch,1,200,234]")
    batch = radar_probability.shape[0]
    for name, value in (
        ("radar_xy_m", radar_xy_m),
        ("visual_x_m", visual_x_m),
        ("visual_sigma_x_m", visual_sigma_x_m),
    ):
        expected = (batch, 2) if name == "radar_xy_m" else (batch, 1)
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must be {list(expected)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if radius_m <= 0.0 or step_m <= 0.0:
        raise ValueError("radius_m and step_m must be positive")
    radius_steps = int(round(float(radius_m) / float(step_m)))
    if (
        radius_steps < 1
        or abs(radius_steps * float(step_m) - float(radius_m)) > 1e-7
    ):
        raise ValueError("radius_m must be an integer multiple of step_m")

    device = radar_probability.device
    dtype = radar_probability.dtype
    negative = torch.arange(
        -radius_steps,
        0,
        device=device,
        dtype=dtype,
    )
    positive = torch.arange(
        1,
        radius_steps + 1,
        device=device,
        dtype=dtype,
    )
    offset = torch.cat(
        (
            torch.zeros(1, device=device, dtype=dtype),
            negative,
            positive,
        )
    ) * float(step_m)
    relative_x_m = offset[None].expand(batch, -1)
    x_m = radar_xy_m[:, 0:1].to(dtype) + relative_x_m
    valid_mask = (x_m >= -10.0 - 1e-6) & (x_m <= 9.9 + 1e-6)
    valid_mask[:, 0] = True

    row_index = torch.arange(batch, device=device)
    radar_y_index = (
        radar_xy_m[:, 1].div(0.1).floor().long().clamp(0, 233)
    )
    x_profile = radar_probability[row_index, 0, :, radar_y_index]
    x_profile = x_profile / x_profile.sum(dim=1, keepdim=True).clamp_min(1e-12)
    fractional_x_index = ((x_m + 10.0) / 0.1).clamp(0.0, 199.0)
    lower_index = fractional_x_index.floor().long()
    upper_index = (lower_index + 1).clamp_max(199)
    upper_weight = fractional_x_index - lower_index.to(dtype)
    lower_value = x_profile.gather(1, lower_index)
    upper_value = x_profile.gather(1, upper_index)
    radar_support = (
        (1.0 - upper_weight) * lower_value + upper_weight * upper_value
    )
    radar_support = torch.where(
        valid_mask,
        radar_support,
        torch.zeros_like(radar_support),
    )

    sigma = visual_sigma_x_m.to(dtype).clamp_min(float(step_m))
    visual_log_likelihood = -0.5 * (
        (x_m - visual_x_m.to(dtype)) / sigma
    ).square()
    visual_log_likelihood = torch.where(
        valid_mask,
        visual_log_likelihood,
        torch.full_like(visual_log_likelihood, -1e4),
    )
    source_id = torch.ones_like(x_m, dtype=torch.long)
    source_id[:, 0] = 0
    result = LocalXLattice(
        x_m=x_m,
        relative_x_m=relative_x_m,
        valid_mask=valid_mask,
        source_id=source_id,
        radar_support=radar_support,
        visual_log_likelihood=visual_log_likelihood,
    )
    result.validate()
    return result


def build_visual_blend_x_lattice(
    radar_probability: Tensor,
    radar_xy_m: Tensor,
    visual_x_m: Tensor,
    visual_sigma_x_m: Tensor,
    *,
    radius_m: float = 0.5,
    blend_fractions: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0),
) -> LocalXLattice:
    """Build bounded radar-to-visual interpolation candidates."""
    if radar_probability.ndim != 4 or radar_probability.shape[1:] != (
        1,
        200,
        234,
    ):
        raise ValueError("radar_probability must be [batch,1,200,234]")
    batch = radar_probability.shape[0]
    for name, value in (
        ("radar_xy_m", radar_xy_m),
        ("visual_x_m", visual_x_m),
        ("visual_sigma_x_m", visual_sigma_x_m),
    ):
        expected = (batch, 2) if name == "radar_xy_m" else (batch, 1)
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must be {list(expected)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    if (
        len(blend_fractions) < 2
        or blend_fractions[0] != 0.0
        or any(
            current <= previous
            for previous, current in zip(
                blend_fractions,
                blend_fractions[1:],
            )
        )
        or blend_fractions[-1] > 1.0
    ):
        raise ValueError(
            "blend_fractions must start at zero and increase through at most one"
        )

    device = radar_probability.device
    dtype = radar_probability.dtype
    blend = torch.tensor(
        blend_fractions,
        device=device,
        dtype=dtype,
    )
    visual_correction_m = (
        visual_x_m.to(dtype) - radar_xy_m[:, 0:1].to(dtype)
    ).clamp(-float(radius_m), float(radius_m))
    relative_x_m = visual_correction_m * blend[None]
    x_m = radar_xy_m[:, 0:1].to(dtype) + relative_x_m
    valid_mask = (x_m >= -10.0 - 1e-6) & (x_m <= 9.9 + 1e-6)
    valid_mask[:, 0] = True

    row_index = torch.arange(batch, device=device)
    radar_y_index = (
        radar_xy_m[:, 1].div(0.1).floor().long().clamp(0, 233)
    )
    x_profile = radar_probability[row_index, 0, :, radar_y_index]
    x_profile = x_profile / x_profile.sum(dim=1, keepdim=True).clamp_min(1e-12)
    fractional_x_index = ((x_m + 10.0) / 0.1).clamp(0.0, 199.0)
    lower_index = fractional_x_index.floor().long()
    upper_index = (lower_index + 1).clamp_max(199)
    upper_weight = fractional_x_index - lower_index.to(dtype)
    lower_value = x_profile.gather(1, lower_index)
    upper_value = x_profile.gather(1, upper_index)
    radar_support = (
        (1.0 - upper_weight) * lower_value + upper_weight * upper_value
    )
    radar_support = torch.where(
        valid_mask,
        radar_support,
        torch.zeros_like(radar_support),
    )

    sigma = visual_sigma_x_m.to(dtype).clamp_min(0.025)
    visual_log_likelihood = -0.5 * (
        (x_m - visual_x_m.to(dtype)) / sigma
    ).square()
    visual_log_likelihood = torch.where(
        valid_mask,
        visual_log_likelihood,
        torch.full_like(visual_log_likelihood, -1e4),
    )
    source_id = torch.arange(
        blend.numel(),
        device=device,
        dtype=torch.long,
    )[None].expand(batch, -1)
    result = LocalXLattice(
        x_m=x_m,
        relative_x_m=relative_x_m,
        valid_mask=valid_mask,
        source_id=source_id,
        radar_support=radar_support,
        visual_log_likelihood=visual_log_likelihood,
    )
    result.validate()
    return result



