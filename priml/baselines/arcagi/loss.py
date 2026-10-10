"""Token losses and batch reductions for the reference TRM recipes.

Two axes the recipes vary independently, so each is a slot:

* the per-token loss -- softmax cross-entropy with label smoothing
  (:class:`CrossEntropyTokens`) or the float64 stablemax surrogate
  (:class:`StablemaxTokens`);
* how per-sample losses become one number -- averaged over the slots that
  trained (:class:`MeanOverActive`) or summed and divided by the pool width
  (:class:`MeanOverBatch`), the reference's data-parallel form.
"""

from __future__ import annotations

from typing import Protocol

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.math.loss import stablemax_cross_entropy


class TokenLoss(Protocol):
    """Map ``[B, S, V]`` logits and ``[B, S]`` labels to ``[B, S]`` losses."""

    def __call__(self, logits: Tensor, labels: Tensor, *, ignore_index: int) -> Tensor:
        """Apply to the input."""
        ...


class Reduction(Protocol):
    """Reduce ``[B]`` per-sample losses over the slots that trained."""

    def __call__(self, per_sample: Tensor, *, active: Tensor) -> Tensor:
        """Apply to the input."""
        ...


class CrossEntropyTokens:
    """Softmax cross-entropy per token, in the logits' dtype."""

    class Config(Fig["CrossEntropyTokens"]):
        """Label smoothing."""

        label_smoothing: float = 0.0
        """Probability mass spread uniformly over the vocabulary."""

    def __init__(self, config: Config) -> None:
        self.label_smoothing = config.label_smoothing

    def __call__(self, logits: Tensor, labels: Tensor, *, ignore_index: int) -> Tensor:
        """Return ``[B, S]`` per-token losses; ignored positions are zero."""
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1).long(),
            label_smoothing=self.label_smoothing,
            ignore_index=ignore_index,
            reduction="none",
        ).reshape(logits.shape[0], -1)


class StablemaxTokens:
    """Stablemax cross-entropy per token, computed in float64.

    Float64 for the caller's sum, not for the surrogate: accumulating hundreds
    of bfloat16 terms costs far more than bfloat16's own rounding.
    """

    class Config(Fig["StablemaxTokens"]):
        """No parameters."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, logits: Tensor, labels: Tensor, *, ignore_index: int) -> Tensor:
        """Return ``[B, S]`` float64 per-token losses; ignored positions are zero."""
        return stablemax_cross_entropy(
            logits.double(),
            labels.long(),
            ignore_index=ignore_index,
        )


class MeanOverActive:
    """Average over the slots that trained this step."""

    class Config(Fig["MeanOverActive"]):
        """No parameters."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, per_sample: Tensor, *, active: Tensor) -> Tensor:
        """Return the masked sum divided by the active count (at least one)."""
        masked = torch.where(active, per_sample, torch.zeros_like(per_sample))
        return masked.sum() / active.sum().clamp(min=1)


class MeanOverBatch:
    """Sum over the slots that trained, divided by the fixed pool width.

    The reference's data-parallel form: every rank divides by the same
    constant, so the mean the gradient all-reduce takes is the global one.
    """

    class Config(Fig["MeanOverBatch"]):
        """Pool width; inherited from the train step."""

        batch_size: int = -1
        """Divisor; -1 inherits the pool width."""

    def __init__(self, config: Config) -> None:
        if config.batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive; got {config.batch_size}. It is "
                "normally inherited from the pool during finalize.",
            )
        self.batch_size = config.batch_size

    def __call__(self, per_sample: Tensor, *, active: Tensor) -> Tensor:
        """Return the masked sum divided by the pool width."""
        masked = torch.where(active, per_sample, torch.zeros_like(per_sample))
        return masked.sum() / float(self.batch_size)
