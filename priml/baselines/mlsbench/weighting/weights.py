"""Per-class loss weights of the MLS-Bench ``cv-sample-weighting`` baselines.

Each maps the training split's per-class counts to the weight vector of a
weighted cross-entropy, as the task's editable ``compute_class_weights`` does.
They fill :attr:`~priml.baselines.mlsbench.weighting.loss.ClassWeightedCrossEntropy.Config.class_weight`.

Every expression keeps the reference's operation order: weights that must
reproduce the reference's bit for bit follow its arithmetic, so an
algebraically equal rewrite is not a refactor.
"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

import torch


class ClassWeightFn(Protocol):
    """Maps per-class training counts to per-class loss weights."""

    def __call__(self, class_counts: Tensor, /, *, num_classes: int) -> Tensor:
        """Return ``(num_classes,)`` weights for ``(num_classes,)`` float32 counts."""
        raise NotImplementedError


def uniform(class_counts: Tensor, /, *, num_classes: int) -> Tensor:
    """Weigh every class one: the task's pristine, unweighted recipe."""
    del class_counts
    return torch.ones(num_classes)


def inverse_frequency(class_counts: Tensor, /, *, num_classes: int) -> Tensor:
    """Weigh each class by the square root of its inverse frequency.

    Args:
      class_counts: ``(C,)`` training images per class.
      num_classes: ``C``.

    Returns:
      weights: ``sqrt(N / (C n_c))``, rescaled to sum to ``C``.

    """
    total = class_counts.sum().float()
    weights = total / (num_classes * class_counts.float())
    weights = torch.sqrt(weights)
    return weights / weights.sum() * num_classes


def effective_number(
    class_counts: Tensor,
    /,
    *,
    num_classes: int,
    beta: float,
) -> Tensor:
    """Weigh each class by the square root of its inverse effective number of samples.

    Args:
      class_counts: ``(C,)`` training images per class.
      num_classes: ``C``.
      beta: Overlap of samples; ``(1 - beta^n) / (1 - beta)`` is the effective count.

    Returns:
      weights: ``sqrt((1 - beta) / (1 - beta^n_c))``, rescaled to sum to ``C``.

    References:
      https://arxiv.org/abs/1901.05555
        Cui et al. 2019, "Class-Balanced Loss Based on Effective Number of
        Samples."

    """
    effective_num = 1.0 - torch.pow(beta, class_counts.float())
    weights = (1.0 - beta) / effective_num
    weights = torch.sqrt(weights)
    return weights / weights.sum() * num_classes


def log_frequency_gap(
    class_counts: Tensor,
    /,
    *,
    num_classes: int,
    spread: float,
) -> Tensor:
    """Weigh each class by how far its log count falls below the largest.

    Args:
      class_counts: ``(C,)`` training images per class.
      num_classes: ``C``.
      spread: Weight added to the rarest class over the most frequent, before
        rescaling.

    Returns:
      weights: ``1 + spread * gap_c / max(gap)`` with ``gap_c = log n_max -
        log n_c``, all ones when every count is equal, rescaled to sum to ``C``.

    References:
      https://arxiv.org/abs/2007.10740
        Ren et al. 2020, "Balanced Meta-Softmax for Long-Tailed Visual
        Recognition."

    """
    log_counts = torch.log(class_counts.float())
    gap = log_counts.max() - log_counts
    max_gap = gap.max()
    # A select rather than the reference's Python ``if``, which would read the
    # value off the device; the dead branch's 0 / 0 is discarded either way.
    weights = torch.where(
        max_gap > 0,
        1.0 + spread * (gap / max_gap),
        torch.ones(num_classes, device=gap.device),
    )
    return weights / weights.sum() * num_classes
