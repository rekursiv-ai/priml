"""The published robust losses of MLS-Bench ``security-poison-robust-learning``.

Each is a value for the step's ``loss`` slot, called as the vision scaffold
calls its pristine
:func:`~priml.baselines.mlsbench.dl.vision.train_step.cross_entropy`, which
is this task's pristine ``RobustLoss`` and its ``cross_entropy`` baseline. A
loss's settings are keywords a ``PartialConfig`` binds, so they print with the
experiment.

Every expression keeps the reference's operations in the reference's order: a
loss that must reproduce the reference's updates bit for bit follows its
arithmetic, so an algebraically equal rewrite is not a refactor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn import functional

import torch


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.mlsbench.dl.vision.train_step import LossContext
    from priml.loss.custom_types import LossOutput


def generalized_ce(
    logits: Tensor,
    *,
    label: Tensor,
    context: LossContext,
    q: float,
    min_probability: float,
) -> LossOutput:
    """Return the generalized cross-entropy ``mean((1 - p^q) / q)``.

    Args:
      logits: ``(B, C)`` class scores.
      label: ``(B,)`` target classes.
      context: Unused; the loss has no schedule.
      q: Exponent; ``q -> 0`` recovers cross-entropy, ``q = 1`` mean absolute error.
      min_probability: Floor on the true class's probability ``p``.

    Returns:
      loss: Scalar under ``"loss"``.

    References:
      https://arxiv.org/abs/1805.07836
        Zhang and Sabuncu 2018, "Generalized Cross Entropy Loss for Training
        Deep Neural Networks with Noisy Labels."

    """
    del context
    probs = torch.softmax(logits, dim=1)
    p = probs.gather(1, label[:, None]).clamp_min(min_probability)
    return {"loss": ((1.0 - p.pow(q)) / q).mean()}


def symmetric_ce(
    logits: Tensor,
    *,
    label: Tensor,
    context: LossContext,
    alpha: float,
    beta: float,
    min_probability: float,
    min_target: float,
) -> LossOutput:
    """Return ``alpha * CE + beta * RCE``, cross-entropy plus its reverse.

    The reverse cross-entropy swaps the roles of the prediction and the one-hot
    label, whose zeros are clamped to ``min_target`` so their logarithm is
    finite.

    Args:
      logits: ``(B, C)`` class scores.
      label: ``(B,)`` target classes.
      context: Unused; the loss has no schedule.
      alpha: Weight of the cross-entropy.
      beta: Weight of the reverse cross-entropy.
      min_probability: Floor on each predicted probability.
      min_target: Floor on each entry of the one-hot label.

    Returns:
      loss: Scalar under ``"loss"``.

    References:
      https://arxiv.org/abs/1908.06112
        Wang et al. 2019, "Symmetric Cross Entropy for Robust Learning with
        Noisy Labels."

    """
    del context
    ce = functional.cross_entropy(logits, label)
    probs = torch.softmax(logits, dim=1).clamp_min(min_probability)
    one_hot = (
        functional.one_hot(label, num_classes=logits.shape[1])
        .float()
        .clamp_min(min_target)
    )
    rce = -(probs * torch.log(one_hot)).sum(dim=1).mean()
    return {"loss": alpha * ce + beta * rce}


def bootstrap(
    logits: Tensor,
    *,
    label: Tensor,
    context: LossContext,
    beta: float,
) -> LossOutput:
    """Return soft bootstrapping: cross-entropy against a label mixed with the prediction.

    The target is ``beta * one_hot + (1 - beta) * softmax(logits)``, the
    prediction detached, so the gradient flows through the log-probabilities
    only.

    Args:
      logits: ``(B, C)`` class scores.
      label: ``(B,)`` target classes.
      context: Unused; the loss has no schedule.
      beta: Weight of the given label; one recovers cross-entropy.

    Returns:
      loss: Scalar under ``"loss"``.

    References:
      https://arxiv.org/abs/1412.6596
        Reed et al. 2015, "Training Deep Neural Networks on Noisy Labels with
        Bootstrapping."

    """
    del context
    hard = functional.one_hot(label, num_classes=logits.shape[1]).float()
    soft = torch.softmax(logits.detach(), dim=1)
    target = beta * hard + (1.0 - beta) * soft
    log_probs = functional.log_softmax(logits, dim=1)
    return {"loss": -(target * log_probs).sum(dim=1).mean()}
