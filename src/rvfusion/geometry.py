from __future__ import annotations

import torch
from torch import Tensor


def wrap_angle(angle: Tensor) -> Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def build_visual_ray(
    theta: Tensor,
    sigma_theta: Tensor,
    *,
    x_min_m: float = -10.0,
    y_min_m: float = 0.0,
    resolution_m: float = 0.1,
    height: int = 200,
    width: int = 234,
) -> Tensor:
    if theta.ndim != 2 or theta.shape[1] != 1:
        raise ValueError("theta must be [batch,1]")
    if tuple(sigma_theta.shape) != tuple(theta.shape):
        raise ValueError("sigma_theta must match theta")
    batch = theta.shape[0]
    if not (sigma_theta > 0.0).all():
        raise ValueError("sigma_theta must be positive")

    dtype = theta.dtype
    device = theta.device
    x = x_min_m + (torch.arange(height, device=device, dtype=dtype) + 0.5) * resolution_m
    y = y_min_m + (torch.arange(width, device=device, dtype=dtype) + 0.5) * resolution_m
    x_grid = x.view(1, height, 1).expand(batch, height, width)
    y_grid = y.view(1, 1, width).expand(batch, height, width)
    theta_cell = torch.atan2(x_grid, y_grid.clamp_min(1e-4))
    error = wrap_angle(theta_cell - theta[:, :, None])
    sigma = sigma_theta[:, :, None].clamp_min(1e-4)
    ray = torch.exp(-0.5 * (error / sigma).square())
    return ray.unsqueeze(1)



