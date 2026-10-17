"""Learning-rate schedules of the MLS-Bench ``dl-lr-schedule`` baselines.

Each maps a zero-based epoch to the rate every parameter group uses during it,
as the task's editable ``get_lr`` does. They fill
:attr:`~priml.baselines.mlsbench.pytorch_vision.train_step.VisionTrainStep.Config.schedule`.

Every expression keeps the reference's operation order: a schedule that must
reproduce the reference's rates bit for bit follows its arithmetic, so an
algebraically equal rewrite is not a refactor.
"""

from __future__ import annotations

import math


def constant(epoch: int, /, *, base_lr: float, total_epochs: int) -> float:
    """Hold the rate at ``base_lr``: the task's pristine schedule.

    Args:
      epoch: Unused.
      base_lr: The rate of every epoch.
      total_epochs: Unused.

    Returns:
      lr: ``base_lr``.

    """
    del epoch, total_epochs
    return base_lr


def cosine(epoch: int, /, *, base_lr: float, total_epochs: int) -> float:
    """Anneal from ``base_lr`` to zero along half a cosine.

    Args:
      epoch: Completed epochs.
      base_lr: Rate of epoch 0.
      total_epochs: Epochs over which the rate reaches zero.

    Returns:
      lr: ``base_lr * (1 + cos(pi * epoch / total_epochs)) / 2``.

    References:
      https://arxiv.org/abs/1608.03983
        Loshchilov and Hutter 2017, "SGDR: Stochastic Gradient Descent with
        Warm Restarts."

    """
    return base_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))


def warmup_cosine(
    epoch: int,
    /,
    *,
    base_lr: float,
    total_epochs: int,
    warmup_epochs: int,
) -> float:
    """Ramp linearly to ``base_lr``, then anneal to zero along half a cosine.

    Args:
      epoch: Completed epochs.
      base_lr: Peak rate, reached in the last warmup epoch.
      total_epochs: Epochs over which the rate reaches zero.
      warmup_epochs: Epochs of the ramp, from ``base_lr / warmup_epochs``.

    Returns:
      lr: Learning rate to use during ``epoch``.

    References:
      https://arxiv.org/abs/1706.02677
        Goyal et al. 2017, "Accurate, Large Minibatch SGD: Training ImageNet in
        1 Hour."

    """
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def one_cycle(
    epoch: int,
    /,
    *,
    base_lr: float,
    total_epochs: int,
    pct_start: float,
    div_factor: float,
    final_div: float,
) -> float:
    """Rise along a cosine to ``base_lr``, then fall along one to a floor.

    Args:
      epoch: Completed epochs.
      base_lr: Peak rate.
      total_epochs: Epochs of the whole cycle.
      pct_start: Fraction of the cycle spent rising.
      div_factor: The rise starts at ``base_lr / div_factor``.
      final_div: The fall ends at ``base_lr / final_div``.

    Returns:
      lr: Learning rate to use during ``epoch``.

    References:
      https://arxiv.org/abs/1708.07120
        Smith and Topin 2019, "Super-Convergence: Very Fast Training of Neural
        Networks Using Large Learning Rates."

    """
    min_lr = base_lr / div_factor
    final_lr = base_lr / final_div
    progress = epoch / max(total_epochs - 1, 1)
    if progress <= pct_start:
        t = progress / pct_start
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * (1 - t)))
    t = (progress - pct_start) / (1 - pct_start)
    return final_lr + (base_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * t))
