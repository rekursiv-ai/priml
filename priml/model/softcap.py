"""Bound a projection's output by ``cap * tanh(x / cap)``.

A confident logit contributes a gradient proportional to its magnitude, so one
token far outside the rest sets the step for the whole batch. Squashing the
output bounds that contribution without bounding the weights, which is what
keeps a run stable at a large learning rate.

Wraps the projection rather than post-processing it, so a model reads one slot
and a caller who wants no cap supplies the bare projection.

References:
    https://arxiv.org/abs/2408.00118
      Gemma 2, which caps both its attention logits and its final logits.

"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Any, Self, override

import math

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    TensorModule,
    propagate_attr,
)
from priml.model.linear import Linear


class SoftCap(nn.Module):
    """A projection whose output is squashed into ``[-cap, cap]``."""

    class Config(Fig["SoftCap"], kw_only=False):
        cap: float = 15.0
        """Bound the output is squashed into, in both directions."""

        _: KW_ONLY

        inner: Makeable[TensorModule] = field(default_factory=Linear.Config)
        """The projection being bounded."""

        channels_in: int = -1
        """Input width; forwarded to ``inner``."""

        channels_out: int = -1
        """Output width; forwarded to ``inner``."""

        dtype: torch.dtype | None = torch.float32
        """Width the squash is computed at; None keeps the projection's own.

        Float32 by default, and that is arithmetic rather than storage: a
        ``tanh`` evaluated in bfloat16 rounds its own derivative, so capping
        before the upcast and capping after it give the same forward and
        DIFFERENT gradients (measured: 0.5 on a 32x16 readout). A capped
        readout is the last thing before a loss that runs in float32 anyway,
        so the upcast belongs on this side of it."""

        @override
        def finalize(self) -> Self:
            propagate_attr(
                self.inner, "channels_in", self.channels_in, protocol=ChannelsIn
            )
            propagate_attr(
                self.inner, "channels_out", self.channels_out, protocol=ChannelsOut
            )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        if not math.isfinite(config.cap) or config.cap <= 0:
            raise ValueError(f"cap must be finite and positive; got {config.cap}.")
        self.cap = config.cap
        self.dtype = config.dtype
        self.inner = config.inner.make()

    def reset_parameters(self) -> None:
        """Re-initialize the wrapped projection."""
        self.inner.reset_parameters()

    @override
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        out = self.inner(x)
        if self.dtype is not None:
            out = out.to(self.dtype)
        return self.cap * torch.tanh(out / self.cap)
