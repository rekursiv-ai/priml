"""Resolution-aware monotone timestep shifting for flow matching."""

from __future__ import annotations

from torch import Tensor


def time_shift(
    t: Tensor, latent_dimensions: int, reference_dimensions: int = 4096
) -> Tensor:
    """Shift uniform times toward the noisier end for larger latent spaces."""
    if latent_dimensions <= 0 or reference_dimensions <= 0:
        raise ValueError("latent and reference dimensions must be positive")
    shift = (latent_dimensions / reference_dimensions) ** 0.5
    return (shift * t / (1 + (shift - 1) * t)).clamp(0, 1)
