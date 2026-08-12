"""Custom types for loss module."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable
from typing_extensions import TypedDict

from torch import Tensor


__all__ = [
    "LossOutput",
    "LossProtocol",
    "SimpleLossFn",
]


class LossOutput(TypedDict, extra_items=Tensor):
    """Output from loss functions.

    Primary loss must be in 'loss' key.
    Additional keys can contain auxiliary losses for logging.
    """

    loss: Tensor


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


@runtime_checkable
class LossProtocol(Protocol):
    """Protocol for loss functions."""

    def __call__(self, **batch: dict[str, Any]) -> LossOutput:
        """Compute loss.

        Args:
          **batch: Batch data as kwargs (typically includes logits, targets, etc).

        Returns:
          loss: Dict with 'loss' key (and optional auxiliary losses).

        """
        ...
