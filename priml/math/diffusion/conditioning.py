"""Stateless diffusion conditioning functions."""

from __future__ import annotations

import math

from torch import Tensor

import torch


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply per-example shift and scale to every token."""
    return x * (1 + scale[:, None]) + shift[:, None]


def timestep_embedding(t: Tensor, width: int = 256, period: float = 10_000.0) -> Tensor:
    """Sinusoidal embedding for fractional diffusion times."""
    half = width // 2
    frequencies = torch.exp(
        -math.log(period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    angles = t.float()[:, None] * frequencies[None]
    result = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if width % 2:
        result = torch.cat((result, torch.zeros_like(result[:, :1])), dim=-1)
    return result
