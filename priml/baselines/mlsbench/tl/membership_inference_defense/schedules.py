"""The step decay of the ``security-membership-inference-defense`` recipe.

A value for
:attr:`~priml.baselines.mlsbench.dl.vision.train_step.VisionTrainStep.Config.schedule`,
mapping a zero-based epoch to the rate every parameter group uses during it.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence


def multistep(
    epoch: int,
    /,
    *,
    base_lr: float,
    total_epochs: int,
    milestones: Sequence[int],
    gamma: float,
) -> float:
    """Multiply the rate by ``gamma`` at every milestone the run has reached.

    A milestone listed ``k`` times multiplies by ``gamma**k`` once, and the
    milestones apply in ascending order, as ``torch.optim.lr_scheduler.MultiStepLR``
    updates its rate; the products round as its recurrence rounds.

    Args:
      epoch: Completed epochs.
      base_lr: Rate of epoch 0.
      total_epochs: Unused; the milestones are absolute.
      milestones: Epochs from which the next decay applies.
      gamma: Factor of each decay.

    Returns:
      lr: Learning rate to use during ``epoch``.

    """
    del total_epochs
    lr = base_lr
    for milestone, count in sorted(Counter(milestones).items()):
        if epoch >= milestone:
            lr = lr * float(gamma**count)
    return lr
