"""The vision training loop, evaluating only after the epochs a policy picks.

Some MLS-Bench scripts evaluate the test split after the first epoch, every
tenth and the last, not after every epoch. Each evaluation pass takes one draw
from the global generator, as the reference's loader does, so evaluating after
any other epoch moves every shuffle that follows. Which epochs evaluate is a
slot.
"""

from __future__ import annotations

from dataclasses import field
from typing import Generic, Self, cast, override
from typing_extensions import TypeVar

from configgle import Makeable, PartialConfig

from priml.baselines.mlsbench.dl.vision.data import Cifar100Data
from priml.baselines.mlsbench.dl.vision.experiments import check_reference_loop
from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep
from priml.train.epoch_loop import (
    EpochLoopConfig,
    EvalEpochs,
    first_periodic_and_last,
)


_StepT_co = TypeVar(
    "_StepT_co",
    bound=VisionTrainStep.Config,
    default=VisionTrainStep.Config,
    covariant=True,
)
_DataT_co = TypeVar(
    "_DataT_co",
    bound=Cifar100Data.Config,
    default=Cifar100Data.Config,
    covariant=True,
)


class VisionLoop(
    EpochLoopConfig[_StepT_co, _DataT_co],
    Generic[_StepT_co, _DataT_co],
):
    """The vision step, prepared CIFAR tensors and the evaluation policy.

    Generic over the step and dataset configs. A task narrows either by
    subclassing the subscripted config, with ``Makes`` so ``make()`` is
    typed, and redeclaring the field::

        class MyLoop(
            Makes["EpochTrainLoop"],
            VisionLoop[VisionTrainStep.Config, MyData.Config],
        ):
            dataset: MyData.Config = field(default_factory=MyData.Config)
    """

    step: _StepT_co = field(
        default_factory=lambda: cast(_StepT_co, VisionTrainStep.Config()),
    )
    """Model, optimizer, loss and schedule."""

    dataset: _DataT_co = field(
        default_factory=lambda: cast(_DataT_co, Cifar100Data.Config()),
    )
    """Prepared CIFAR tensors, shuffled and augmented per pass."""

    eval_epochs: Makeable[EvalEpochs] = field(
        default_factory=lambda: PartialConfig(first_periodic_and_last, every=10),
    )
    """Picks the completed epochs after which the test split is evaluated."""

    @override
    def finalize(self) -> Self:
        assert isinstance(self.step, VisionTrainStep.Config)
        check_reference_loop(self, budget_epochs=self.step.train_budget_epochs)
        return super().finalize()
