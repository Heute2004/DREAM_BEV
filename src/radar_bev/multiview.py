from __future__ import annotations

import math

import torch
from torch import Tensor

from .geometry import BEVGeometry


def range_indices_from_xy(xy_m: Tensor, geometry: BEVGeometry) -> Tensor:
    """Map metric BEV positions to nearest native radar range bins."""
    if xy_m.ndim != 2 or xy_m.shape[-1] != 2:
        raise ValueError("xy_m must be [batch,2]")
    radial_range = xy_m.square().sum(dim=-1).sqrt()
    return (radial_range / geometry.range_resolution_m).round().long().clamp(0, geometry.range_bins - 1)


def _range_band(view: Tensor, range_index: Tensor, half_width: int) -> Tensor:
    if view.ndim != 3 or view.shape[1:] != (128, 256):
        raise ValueError("view must be [batch,128,256]")
    if range_index.shape != (view.shape[0],):
        raise ValueError("range_index must be [batch]")
    offsets = torch.arange(-half_width, half_width + 1, device=view.device)
    rows = (range_index[:, None] + offsets[None]).clamp(0, view.shape[1] - 1)
    return view.gather(1, rows[:, :, None].expand(-1, -1, view.shape[2]))


def _profile_statistics(profile: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    top_two = profile.topk(2, dim=1).values
    probability = torch.softmax(profile / 0.25, dim=1)
    entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=1) / math.log(profile.shape[1])
    peak_index = profile.argmax(dim=1).to(dtype=profile.dtype)
    zero_index = profile.shape[1] // 2
    zero_band = profile[:, zero_index - 2 : zero_index + 3].mean(dim=1)
    return top_two[:, 0], top_two[:, 0] - top_two[:, 1], entropy, peak_index - float(zero_index) + zero_band * 0.0


def rd_ad_causal_descriptors(
    rd: Tensor,
    rd_mti: Tensor,
    ad: Tensor,
    reference_xy_m: Tensor,
    geometry: BEVGeometry | None = None,
    range_half_width: int = 2,
) -> dict[str, Tensor]:
    """Extract current/past-only RD/RD-MIT/AD descriptors around a BEV range reference."""
    if rd.ndim != 4 or rd.shape[2:] != (128, 256):
        raise ValueError("rd must be [batch,frames,128,256]")
    if rd_mti.shape != rd.shape or ad.shape != rd.shape:
        raise ValueError("rd_mti and ad must match rd shape")
    geometry = geometry or BEVGeometry()
    range_index = range_indices_from_xy(reference_xy_m, geometry)
    rd_current = _range_band(rd[:, -1], range_index, range_half_width)
    rd_mti_current = _range_band(rd_mti[:, -1], range_index, range_half_width)
    raw_profile = rd_current.mean(dim=1)
    mti_profile = rd_mti_current.mean(dim=1)
    raw_peak, raw_margin, raw_entropy, raw_peak_offset = _profile_statistics(raw_profile)
    mti_peak, mti_margin, mti_entropy, mti_peak_offset = _profile_statistics(mti_profile)
    if rd.shape[1] > 1:
        previous = _range_band(rd_mti[:, -2], range_index, range_half_width).mean(dim=1)
        current = mti_profile
        previous = previous - previous.mean(dim=1, keepdim=True)
        current = current - current.mean(dim=1, keepdim=True)
        persistence = (previous * current).sum(dim=1) / (previous.norm(dim=1) * current.norm(dim=1)).clamp_min(1e-6)
    else:
        persistence = torch.ones_like(mti_peak)
    ad_profile = ad[:, -1].mean(dim=1)
    ad_peak, ad_margin, ad_entropy, ad_peak_offset = _profile_statistics(ad_profile)
    return {
        "reference_range_bin": range_index.to(dtype=rd.dtype),
        "rd_local_energy": rd_current.mean(dim=(1, 2)),
        "rd_mti_local_energy": rd_mti_current.mean(dim=(1, 2)),
        "rd_mti_minus_rd": rd_mti_current.mean(dim=(1, 2)) - rd_current.mean(dim=(1, 2)),
        "rd_peak_margin": raw_margin,
        "rd_mti_peak_margin": mti_margin,
        "rd_entropy": raw_entropy,
        "rd_mti_entropy": mti_entropy,
        "rd_peak_doppler_offset": raw_peak_offset,
        "rd_mti_peak_doppler_offset": mti_peak_offset,
        "rd_mti_temporal_persistence": persistence,
        "ad_global_energy": ad[:, -1].mean(dim=(1, 2)),
        "ad_peak_margin": ad_margin,
        "ad_entropy": ad_entropy,
        "ad_peak_doppler_offset": ad_peak_offset,
    }



