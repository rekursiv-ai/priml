"""Tokenwise losses and reductions over pooled training rows."""

from __future__ import annotations

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.math.custom_types import LogNormalizerFn
from priml.math.loss import log_stablemax, stablemax_cross_entropy


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
    """Stablemax cross-entropy per token, computed in float64."""

    class Config(Fig["StablemaxTokens"]):
        """Normalizer and accumulation precision for the stablemax objective."""

        log_normalizer: LogNormalizerFn = log_stablemax
        """Log-probability normalizer; its operation order is part of the recipe."""

    def __init__(self, config: Config) -> None:
        self.log_normalizer = config.log_normalizer

    def __call__(self, logits: Tensor, labels: Tensor, *, ignore_index: int) -> Tensor:
        """Return ``[B, S]`` float64 losses; ignored positions are zero."""
        return stablemax_cross_entropy(
            logits.double(),
            labels.long(),
            ignore_index=ignore_index,
            log_normalizer=self.log_normalizer,
        )


class MeanOverActive:
    """Average losses over the rows active in the current optimizer step."""

    class Config(Fig["MeanOverActive"]):
        """No parameters."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, per_sample: Tensor, *, active: Tensor) -> Tensor:
        """Return the active loss sum divided by at least one active row."""
        masked = torch.where(active, per_sample, torch.zeros_like(per_sample))
        return masked.sum() / active.sum().clamp(min=1)


class MeanOverBatch:
    """Sum active losses and divide by the fixed pool width."""

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
        """Return the masked sum divided by the configured pool width."""
        masked = torch.where(active, per_sample, torch.zeros_like(per_sample))
        return masked.sum() / float(self.batch_size)
