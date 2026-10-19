"""The learning-rate schedule the ``dl-regularization`` script gives VGG networks.

The script anneals every other network along ``CosineAnnealingLR``, the vision
recipe's ``cosine_annealing``; for an ``--arch`` starting ``vgg`` it builds a
``LambdaLR`` whose factor warms up linearly for five epochs and then follows
half a cosine. This fills
:attr:`~priml.baselines.mlsbench.dl.vision.train_step.VisionTrainStep.Config.schedule`
at the ``vgg16bn-cifar100`` setting.
"""

from __future__ import annotations

import math


def warmup_cosine_factor(
    epoch: int,
    /,
    *,
    base_lr: float,
    total_epochs: int,
    warmup_epochs: int,
) -> float:
    """Return ``base_lr`` times a linear-warmup-then-cosine factor.

    ``LambdaLR`` multiplies the base rate by the factor last, so the factor is
    computed whole first: ``base_lr * ((epoch + 1) / warmup_epochs)`` rounds
    differently from ``base_lr * (epoch + 1) / warmup_epochs``.

    Args:
      epoch: Completed epochs.
      base_lr: Peak rate, reached in the last warmup epoch.
      total_epochs: Epochs of the run; the cosine spans those after warmup.
      warmup_epochs: Epochs of the ramp, from ``base_lr / warmup_epochs``.

    Returns:
      lr: Learning rate to use during ``epoch``.

    References:
      https://arxiv.org/abs/1706.02677
        Goyal et al. 2017, "Accurate, Large Minibatch SGD: Training ImageNet in
        1 Hour."

    """
    if epoch < warmup_epochs:
        factor = (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * factor
