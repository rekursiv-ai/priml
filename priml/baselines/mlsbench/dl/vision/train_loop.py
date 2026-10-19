"""The vision training loop, evaluating only after the epochs a policy picks.

Some MLS-Bench scripts evaluate the test split after the first epoch, every
tenth and the last, not after every epoch. Each evaluation pass takes one draw
from the global generator, as the reference's loader does, so evaluating after
any other epoch moves every shuffle that follows. Which epochs evaluate is a
slot.
"""

from __future__ import annotations

from dataclasses import field
from typing import Protocol, Self, cast, override
from typing_extensions import TypeVar

from configgle import Makeable, PartialConfig

from priml.baselines.mlsbench.dl.vision.data import Cifar100Data
from priml.baselines.mlsbench.dl.vision.experiments import check_reference_loop
from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep
from priml.train.train_loop import TrainLoop


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


class EvalEpochs(Protocol):
    """Says whether the test split is evaluated after a completed epoch."""

    def __call__(self, epoch: int, /, *, total_epochs: int) -> bool:
        """Return whether to evaluate after ``epoch``, counted from 1."""
        raise NotImplementedError


def first_periodic_and_last(epoch: int, /, *, total_epochs: int, every: int) -> bool:
    """Evaluate after the first epoch, every ``every``-th and the last.

    Args:
      epoch: Completed epochs, from 1.
      total_epochs: Epochs of the run.
      every: Period of the evaluations between the first and the last.

    Returns:
      evaluate: Whether ``epoch`` is evaluated.

    """
    return epoch == 1 or epoch % every == 0 or epoch == total_epochs


class EvalEpochsTrainLoop(TrainLoop):
    """A vision training loop that evaluates after the epochs its policy picks."""

    # No ``Makes`` base here: ``Makes[...]`` hides the generic parameters from
    # ty, so ``Config[VisionTrainStep.Config, MyData.Config]`` stops being a
    # valid base. ``Fig`` still binds ``parent_class`` to this loop at runtime.
    class Config(TrainLoop.Config[_StepT_co, _DataT_co]):
        """The vision step, prepared CIFAR tensors and the evaluation policy.

        Generic over the step and dataset configs. A task narrows either by
        subclassing the subscripted config, with ``Makes`` so ``make()`` is
        typed, and redeclaring the field::

            class MyLoop(
                Makes["EvalEpochsTrainLoop"],
                EvalEpochsTrainLoop.Config[VisionTrainStep.Config, MyData.Config],
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
            check_reference_loop(self, budget_epochs=self.step.train_budget_epochs)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.eval_epochs: EvalEpochs = config.eval_epochs.make()
        super().__init__(config)

    @override
    def _on_epoch_boundary(self) -> None:
        # An evaluation pass draws from the global generator, so the base
        # evaluates only after the epochs the reference does.
        evaluate = self.eval_every_epoch
        self.eval_every_epoch = evaluate and self.eval_epochs(
            self.current_epoch,
            total_epochs=int(self.max_epochs),
        )
        try:
            super()._on_epoch_boundary()
        finally:
            self.eval_every_epoch = evaluate
