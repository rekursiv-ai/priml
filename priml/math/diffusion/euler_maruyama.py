"""Reverse-SDE math and two-stream Euler--Maruyama integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

import torch


if TYPE_CHECKING:
    from collections.abc import Callable


def repa_diffusion(t: Tensor) -> Tensor:
    """Squared diffusion coefficient of the REPA sampler, ``g(t)^2 = 2t``."""
    return 2 * t


def velocity_to_score(
    velocity: Tensor,
    state: Tensor,
    *,
    alpha: Tensor,
    sigma: Tensor,
    d_alpha: Tensor | float,
    d_sigma: Tensor | float,
) -> Tensor:
    """Recover the score under ``x_t = alpha*x + sigma*noise``."""
    ratio = alpha / d_alpha
    variance = sigma**2 - ratio * d_sigma * sigma
    return (ratio * velocity - state) / variance


def velocity_to_drift(velocity: Tensor, score: Tensor, diffusion: Tensor) -> Tensor:
    """Reverse-SDE drift for a predicted probability-path velocity."""
    return velocity - 0.5 * diffusion * score


def guide_drift(strong: Tensor, weak: Tensor, scale: float) -> Tensor:
    """Apply classifier-free guidance to reverse-SDE drifts."""
    return weak + scale * (strong - weak)


def euler_maruyama_grid(
    num_steps: int,
    *,
    last_time: float = 0.04,
    device: torch.device | None = None,
) -> Tensor:
    """Return a float64 grid from one to zero with a final deterministic step."""
    grid = torch.linspace(1.0, last_time, num_steps, dtype=torch.float64, device=device)
    return torch.cat([grid, grid.new_zeros(1)])


def integrate_two_streams(
    media: Tensor,
    cls_token: Tensor,
    grid: Tensor,
    drift: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
    diffusion: Callable[[Tensor], Tensor] = repa_diffusion,
) -> tuple[Tensor, Tensor]:
    """Advance coupled latent and CLS streams with Euler--Maruyama.

    The forward callback evaluates both drifts at the current time. Keeping
    the two streams in one loop ensures that both see the same time grid and
    that their independent noise tensors are drawn in a fixed order.
    """
    latent, cls = media.to(torch.float64), cls_token.to(torch.float64)
    last = grid.shape[0] - 2
    for index in range(last + 1):
        t_curr, t_next = grid[index], grid[index + 1]
        drift_latent, drift_cls = drift(latent, cls, t_curr)
        dt = t_next - t_curr
        if index == last:
            latent = latent + dt * drift_latent
            cls = cls + dt * drift_cls
            break
        noise = torch.randn_like(latent) * torch.sqrt(torch.abs(dt))
        noise_cls = torch.randn_like(cls) * torch.sqrt(torch.abs(dt))
        scale = torch.sqrt(diffusion(t_curr))
        latent = latent + drift_latent * dt + scale * noise
        cls = cls + drift_cls * dt + scale * noise_cls
    return latent, cls
