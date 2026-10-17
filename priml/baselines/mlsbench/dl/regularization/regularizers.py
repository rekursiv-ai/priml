"""Regularization terms of the MLS-Bench ``dl-regularization`` baselines.

Each returns the scalar the task's editable ``compute_regularization`` adds to
cross-entropy, and fills
:attr:`~priml.baselines.mlsbench.dl.regularization.train_step.RegularizedTrainStep.Config.regularization`.

Every expression keeps the reference's operation order, including where a
running sum starts from a zero tensor: a term that must reproduce the
reference's bits follows its arithmetic, so an algebraically equal rewrite is
not a refactor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn
from torch.nn import functional

import torch


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.mlsbench.dl.vision.train_step import LossContext
    from priml.optimizers.parameter_filter import ParameterFilter


def zero(
    model: nn.Module,
    /,
    *,
    media: Tensor,
    logits: Tensor,
    label: Tensor,
    context: LossContext,
) -> Tensor:
    """Return zero, the "no regularization" the pristine script's docstring names."""
    del model, media, label, context
    return torch.tensor(0.0, device=logits.device)


def confidence_penalty(
    model: nn.Module,
    /,
    *,
    media: Tensor,
    logits: Tensor,
    label: Tensor,
    context: LossContext,
    beta: float,
    eps: float,
) -> Tensor:
    """Return ``-beta`` times the mean entropy of the predicted distribution.

    Args:
      model: Unused; the penalty reads only the output.
      media: Unused.
      logits: ``(B, C)`` class scores.
      label: Unused.
      context: Unused; the penalty has no schedule.
      beta: Weight of the penalty.
      eps: Added to each probability inside the logarithm.

    Returns:
      penalty: A scalar, lowest when the predictions are uniform.

    References:
      https://arxiv.org/abs/1701.06548
        Pereyra et al. 2017, "Regularizing Neural Networks by Penalizing
        Confident Output Distributions."

    """
    del model, media, label, context
    probs = functional.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + eps)).sum(dim=-1).mean()
    return -beta * entropy


def orthogonal(
    model: nn.Module,
    /,
    *,
    media: Tensor,
    logits: Tensor,
    label: Tensor,
    context: LossContext,
    strength: float,
    select: ParameterFilter,
) -> Tensor:
    """Return ``strength`` times the sum of ``||W W^T - I||_F^2`` over chosen weights.

    Each chosen weight is flattened to ``(channels_out, -1)``, so ``W W^T`` is
    the Gram matrix of its filters.

    Args:
      model: Network whose parameters are walked in registration order.
      media: Unused.
      logits: Scores, read only for their device.
      label: Unused.
      context: Unused; the penalty has no schedule.
      strength: Coefficient of the summed penalty.
      select: Which named parameters are penalized.

    Returns:
      penalty: A scalar, zero when every chosen weight has orthonormal rows.

    References:
      https://arxiv.org/abs/1609.07093
        Brock et al. 2017, "Neural Photo Editing with Introspective
        Adversarial Networks."

    """
    del media, label, context
    reg = torch.tensor(0.0, device=logits.device)
    for name, parameter in model.named_parameters():
        if select(name, parameter):
            w = parameter.view(parameter.size(0), -1)
            identity = torch.eye(w.size(0), device=w.device)
            reg = reg + ((w @ w.t() - identity) ** 2).sum()
    return strength * reg


def spatial_coactivation(
    model: nn.Module,
    /,
    *,
    media: Tensor,
    logits: Tensor,
    label: Tensor,
    context: LossContext,
    block_size: int,
    strength: float,
    delay: float,
) -> Tensor:
    """Penalize the energy of local spatial blocks of every convolution kernel.

    For each convolution whose kernel is at least ``block_size`` on both
    sides, the squared weight is averaged over input channels, then over every
    ``block_size`` window, and the windows' mean is taken. The term is zero
    for the first ``delay`` of training and then ramps linearly to ``strength``.

    Args:
      model: Network whose convolutions are walked in registration order.
      media: Unused.
      logits: Scores, read only for their device.
      label: Unused.
      context: The epoch, which sets the ramp.
      block_size: Side of the window, and the smallest kernel penalized.
      strength: Coefficient in the last epoch.
      delay: Fraction of training before the term turns on.

    Returns:
      penalty: The mean over penalized kernels, scaled by the ramp.

    References:
      https://arxiv.org/abs/1810.12890
        Ghiasi, Lin and Le 2018, "DropBlock: A regularization method for
        convolutional networks." The idea of penalizing contiguous regions,
        moved here from activations to weights.

    """
    del media, label
    progress = context.epoch / max(context.total_epochs - 1, 1)
    if progress < delay:
        return torch.tensor(0.0, device=logits.device)
    lam = strength * ((progress - delay) / (1 - delay))
    reg = torch.tensor(0.0, device=logits.device)
    count = 0
    for module in model.modules():
        if (
            isinstance(module, nn.Conv2d)
            and min(module.weight.shape[-2:]) >= block_size
        ):
            energy = module.weight.pow(2).mean(dim=1, keepdim=True)
            local = functional.avg_pool2d(
                energy,
                block_size,
                stride=1,
                padding=block_size // 2,
            )
            reg = reg + local.mean()
            count += 1
    if count > 0:
        reg = reg / count
    return lam * reg
