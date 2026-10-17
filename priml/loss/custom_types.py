"""Custom types for loss module."""

from __future__ import annotations

from typing import Literal, Protocol
from typing_extensions import TypedDict

from torch import Tensor


__all__ = [
    "LossOutput",
    "Reduction",
    "SimpleLossFn",
    "TokenLoss",
]


class LossOutput(TypedDict, extra_items=Tensor):
    """Output from loss functions.

    Primary loss must be in 'loss' key.
    Additional keys can contain auxiliary losses for logging.
    """

    loss: Tensor


class TokenLoss(Protocol):
    """Map token logits and labels to one loss per token."""

    def __call__(self, logits: Tensor, labels: Tensor, *, ignore_index: int) -> Tensor:
        """Return per-token losses with ignored positions zeroed."""
        ...


class Reduction(Protocol):
    """Reduce per-sample losses over the rows active in one optimizer step."""

    def __call__(self, per_sample: Tensor, *, active: Tensor) -> Tensor:
        """Return one scalar objective for the active samples."""
        ...


class SimpleLossFn(Protocol):
    """Protocol for simple loss functions (PyTorch-style).

    All PyTorch losses support reduction kwarg.
    """

    def __call__(
        self,
        input: Tensor,
        target: Tensor,
        *,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> Tensor:
        """Compute loss between input and target."""
        ...
