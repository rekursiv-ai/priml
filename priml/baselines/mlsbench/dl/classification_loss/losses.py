"""The published losses of MLS-Bench ``cv-classification-loss``.

Each is a value for the step's ``loss`` slot, called as the scaffold calls
:func:`~priml.baselines.mlsbench.dl.vision.train_step.cross_entropy`, the
task's pristine loss: logits, labels and a
:class:`~priml.baselines.mlsbench.dl.vision.train_step.LossContext`. A
loss's own setting is a keyword a ``PartialConfig`` binds, so it prints with the
experiment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn import functional

import torch


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.mlsbench.dl.vision.train_step import LossContext
    from priml.loss.custom_types import LossOutput


def label_smoothing(
    logits: Tensor,
    *,
    label: Tensor,
    context: LossContext,
    smoothing: float,
) -> LossOutput:
    """Return cross-entropy against ``(1 - smoothing) * one_hot + smoothing / C``.

    Args:
      logits: ``(B, C)`` class scores.
      label: ``(B,)`` target classes.
      context: Unused; the loss has no schedule.
      smoothing: Probability mass spread uniformly over the classes.

    Returns:
      loss: Scalar under ``"loss"``.

    References:
      https://arxiv.org/abs/1512.00567
        Szegedy et al. 2016, "Rethinking the Inception Architecture for
        Computer Vision."

    """
    del context
    return {
        "loss": functional.cross_entropy(logits, label, label_smoothing=smoothing),
    }


def focal(
    logits: Tensor,
    *,
    label: Tensor,
    context: LossContext,
    gamma: float,
) -> LossOutput:
    """Return cross-entropy scaled per example by ``(1 - p) ** gamma``.

    Args:
      logits: ``(B, C)`` class scores.
      label: ``(B,)`` target classes.
      context: Unused; the loss has no schedule.
      gamma: Focusing exponent; zero recovers cross-entropy.

    Returns:
      loss: Scalar under ``"loss"``, where ``p`` is the probability the model
        gives the target class.

    References:
      https://arxiv.org/abs/1708.02002
        Lin et al. 2017, "Focal Loss for Dense Object Detection."

    """
    del context
    ce = functional.cross_entropy(logits, label, reduction="none")
    return {"loss": ((1 - torch.exp(-ce)) ** gamma * ce).mean()}


def poly(
    logits: Tensor,
    *,
    label: Tensor,
    context: LossContext,
    epsilon: float,
) -> LossOutput:
    """Return cross-entropy plus ``epsilon * (1 - p)`` for the first polynomial term.

    Args:
      logits: ``(B, C)`` class scores.
      label: ``(B,)`` target classes.
      context: Unused; the loss has no schedule.
      epsilon: Weight of the correction; zero recovers cross-entropy.

    Returns:
      loss: Scalar under ``"loss"``, where ``p`` is the probability the model
        gives the target class.

    References:
      https://arxiv.org/abs/2204.12511
        Leng et al. 2022, "PolyLoss: A Polynomial Expansion Perspective of
        Classification Loss Functions."

    """
    del context
    ce = functional.cross_entropy(logits, label)
    p = functional.softmax(logits, dim=-1).gather(1, label.unsqueeze(1)).squeeze(1)
    return {"loss": ce + epsilon * (1 - p).mean()}
