"""Forward-only costs the frozen autoencoders share.

The published weights are frozen and every op runs forward only, so these
price no adjoint.
"""

from __future__ import annotations

from typing import Final

import torch

from priml.cost import Cost, map_cost, traffic


POINTWISE: Final = map_cost(primal=1, adjoint=0, adjoint_inputs=0, adjoint_outputs=0)
"""A forward elementwise op with one scalar operand."""

SUM: Final = map_cost(
    primal=1,
    adjoint=0,
    inputs=2,
    adjoint_inputs=0,
    adjoint_outputs=0,
)
"""A forward sum or product of two tensors."""

CLAMP: Final = map_cost(primal=2, adjoint=0, adjoint_inputs=0, adjoint_outputs=0)
"""A forward two-sided clamp."""


def kl_round_trip_cost(
    *,
    pixels_in: int,
    latents: int,
    pixels_out: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost what a KL round trip runs around its halves.

    ``uint8 / 127.5 - 1`` in, the diagonal Gaussian's clamp and draw, and
    ``(x + 1) / 2`` clamped out.

    Args:
      pixels_in: uint8 elements encoded.
      latents: Latent elements drawn.
      pixels_out: Elements decoded.
      dtype: Activation dtype; ``None`` is torch's default.

    Returns:
      cost: Forward FLOPs and logical bytes.

    """
    f32 = torch.float32
    return (
        traffic("primal", kernel="elementwise", elements=pixels_in, dtype=torch.uint8)
        + traffic("primal", kernel="elementwise", elements=pixels_in, dtype=f32)
        + POINTWISE(channels=pixels_in, dtype=f32).tile(2)
        # Clamp the log-variance; ``exp(0.5 * logvar)``.
        + CLAMP(channels=latents, dtype=dtype)
        + POINTWISE(channels=latents, dtype=dtype).tile(2)
        # ``mean + std * noise``: the noise written, a product, a sum.
        + traffic("primal", kernel="elementwise", elements=latents, dtype=dtype)
        + SUM(channels=latents, dtype=dtype).tile(2)
        # ``(x + 1) / 2`` and a two-sided clamp out.
        + POINTWISE(channels=pixels_out, dtype=dtype).tile(2)
        + CLAMP(channels=pixels_out, dtype=dtype)
    )
