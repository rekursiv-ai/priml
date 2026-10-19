"""The published defenses of MLS-Bench ``security-membership-inference-defense``.

Each is a value for the step's ``loss`` slot, called as the vision scaffold
calls its pristine
:func:`~priml.baselines.mlsbench.dl.vision.train_step.cross_entropy`, which
is this task's pristine defense and its ``erm`` baseline. Label smoothing is
:func:`~priml.baselines.mlsbench.dl.classification_loss.losses.label_smoothing`.
A defense's settings are keywords a ``PartialConfig`` binds, so they print with
the experiment.

Every expression keeps the reference's operations in the reference's order: a
defense that must reproduce the reference's updates bit for bit follows its
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


def confidence_penalty(
    logits: Tensor,
    *,
    label: Tensor,
    context: LossContext,
    entropy_weight: float,
    min_probability: float,
) -> LossOutput:
    """Return cross-entropy minus ``entropy_weight`` times the mean predictive entropy.

    Args:
      logits: ``(B, C)`` class scores.
      label: ``(B,)`` target classes.
      context: Unused; the defense has no schedule.
      entropy_weight: Weight of the entropy bonus.
      min_probability: Floor on each probability inside the logarithm.

    Returns:
      loss: Scalar under ``"loss"``.

    References:
      https://arxiv.org/abs/1701.06548
        Pereyra et al. 2017, "Regularizing Neural Networks by Penalizing
        Confident Output Distributions."

    """
    del context
    ce = functional.cross_entropy(logits, label)
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(min_probability))).sum(dim=1).mean()
    return {"loss": ce - entropy_weight * entropy}


def relaxloss(
    logits: Tensor,
    *,
    label: Tensor,
    context: LossContext,
    alpha: float,
    upper: float,
) -> LossOutput:
    """Return RelaxLoss: hold the mean cross-entropy near ``alpha``, flattening posteriors.

    In an even epoch the loss is ``|CE - alpha|``, so descent stops at the
    target level and ascends below it. In an odd epoch a batch above ``alpha``
    descends on cross-entropy; one at or below it ascends on each example's
    cross-entropy and pulls each misclassified one toward a flattened copy of
    its own posterior, the true class's probability (clamped to ``upper``)
    kept and the rest spread evenly.

    Args:
      logits: ``(B, C)`` class scores.
      label: ``(B,)`` target classes.
      context: ``epoch`` picks the phase; ``num_classes`` spreads the rest.
      alpha: Target level of the mean cross-entropy. Upstream derives it from
        the class count, 0.5 for 100 classes and 1.0 otherwise; each setting's
        fork sets it.
      upper: Ceiling on the true class's kept probability.

    Returns:
      loss: Scalar under ``"loss"``.

    References:
      https://arxiv.org/abs/2207.05801
        Chen, Yu and Fritz 2022, "RelaxLoss: Defending Membership Inference
        Attacks without Losing Utility."

    """
    per_example = functional.cross_entropy(logits, label, reduction="none")
    mean = per_example.mean()
    if context.epoch % 2 == 0:
        return {"loss": (mean - alpha).abs()}
    if mean.item() > alpha:
        return {"loss": mean}
    probs = torch.softmax(logits, dim=1)
    kept = torch.clamp(
        probs.gather(1, label.unsqueeze(1)).squeeze(1),
        min=0.0,
        max=upper,
    )
    rest = (1.0 - kept) / (context.num_classes - 1)
    onehot = functional.one_hot(label, num_classes=context.num_classes).float()
    target = (onehot * kept.unsqueeze(1) + (1.0 - onehot) * rest.unsqueeze(1)).detach()
    flattened = -(target * functional.log_softmax(logits, dim=1)).sum(dim=1)
    wrong = 1.0 - logits.argmax(dim=1).eq(label).float()
    return {"loss": (wrong * flattened - per_example).mean()}
