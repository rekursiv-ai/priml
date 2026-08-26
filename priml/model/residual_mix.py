"""Per-layer mix of a running stream with the value it started from.

A deep stack progressively processes its residual stream, so a layer wanting
the raw input has to reconstruct it. Re-mixing the original in at every layer
supplies it directly, with a learned per-layer weight on each side, and costs
two scalars per layer rather than a skip connection.

The weights are held as ONE vector per side rather than a scalar per layer, so
a stack contributes two parameters to an optimizer instead of ``2 * layers``.
"""

from __future__ import annotations

from typing import override

from configgle import Fig
from torch import Tensor, nn

import torch


class ResidualMix(nn.Module):
    """``running[i] * x + original[i] * x0``, per layer."""

    class Config(Fig["ResidualMix"], kw_only=False):
        num_layers: int = -1
        """Layers being mixed; -1 inherits from the stack."""

        running: float = 1.0
        """Initial weight on the running stream.

        One, so a fresh stack passes its stream through unchanged and the mix
        starts as the identity on it."""

        original: float = 0.1
        """Initial weight on the value the stream started from.

        Small but nonzero: the path has to exist from step one for its weight
        to receive a gradient, while starting near zero keeps a fresh stack
        close to a plain residual network."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.num_layers <= 0:
            raise ValueError(
                f"num_layers must be positive; got {config.num_layers}. It is "
                "normally inherited from the stack during finalize.",
            )
        self.config = config
        self.running = nn.Parameter(torch.empty(config.num_layers))
        self.original = nn.Parameter(torch.empty(config.num_layers))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Fill both weight vectors with their configured constants."""
        with torch.no_grad():
            self.running.fill_(self.config.running)
            self.original.fill_(self.config.original)

    @override
    def forward(
        self,
        x: Tensor,
        *,
        original: Tensor,
        layer: int,
        **kwargs: object,
    ) -> Tensor:
        """Mix layer ``layer``'s stream with the value it started from.

        The weights are indexed rather than cast, and that is load-bearing: a
        0-dim tensor does not widen the tensor it multiplies, so a float32
        weight against a bfloat16 stream stays bfloat16 without a cast. Writing
        the cast out gives the same forward and a DIFFERENT backward -- the
        cast is its own autograd node, and the gradient accumulating through it
        rounds at each layer rather than once.

        Args:
          x: The running stream.
          original: The value the stream started from, same shape.
          layer: Which layer's weights to apply.
          **kwargs: Open message bus ignored by this terminal layer.

        Returns:
          mixed: The weighted sum, same shape as ``x``.

        """
        del kwargs
        return self.running[layer] * x + self.original[layer] * original
