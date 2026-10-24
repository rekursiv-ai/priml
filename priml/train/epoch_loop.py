"""A TrainLoop that picks its evaluation epochs and stops early on a metric."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field
from typing import (
    TYPE_CHECKING,
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
    cast,
    override,
)
from typing_extensions import TypeVar

import math

from configgle import Fig, Makeable, PartialConfig

from priml.data.custom_types import DatasetProtocol
from priml.train.custom_types import TrainStepProtocol
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    import torch.distributed as dist
else:
    from wrapt import lazy_import

    # Torch's distributed package costs ~1 s to import and is read only when a
    # process group exists.
    dist = lazy_import("torch.distributed")

_StepT_co = TypeVar(
    "_StepT_co",
    bound=Makeable[TrainStepProtocol],
    covariant=True,
)
_DataT_co = TypeVar(
    "_DataT_co",
    bound=Makeable[DatasetProtocol],
    covariant=True,
)


class EvalEpochs(Protocol):
    """Says whether the eval split is evaluated after a completed epoch."""

    def __call__(self, epoch: int, /, *, total_epochs: float) -> bool:
        """Return whether to evaluate after ``epoch``, counted from 1."""
        raise NotImplementedError


def first_periodic_and_last(epoch: int, /, *, total_epochs: float, every: int) -> bool:
    """Evaluate after the first epoch, every ``every``-th and the last.

    Args:
      epoch: Completed epochs, from 1.
      total_epochs: Epochs of the run; ``math.inf`` has no last epoch.
      every: Period of the evaluations between the first and the last.

    Returns:
      evaluate: Whether ``epoch`` is evaluated.

    """
    return epoch == 1 or epoch % every == 0 or epoch == total_epochs


def every_epoch(epoch: int, /, *, total_epochs: float) -> bool:
    """Evaluate after every epoch."""
    del epoch, total_epochs
    return True


class EarlyStopping:
    """Stop once a metric has not strictly improved for ``patience`` epochs."""

    class Config(Fig["EarlyStopping"]):
        metric: str = "total_loss"
        """Eval scalar compared across evaluations."""

        mode: Literal["min", "max"] = "min"
        """Whether a smaller or a larger ``metric`` is the improvement."""

        patience: int = 1
        """Epochs since the last strict improvement that end training."""

        min_epochs: int = 0
        """First completed epoch at which training may stop."""

    class StateDict(TypedDict):
        """Best value so far, the epoch it was reached, and the verdict."""

        best: float
        best_epoch: int
        stop: bool

    def __init__(self, config: Config) -> None:
        if config.patience < 1:
            raise ValueError(f"patience must be positive, got {config.patience}.")
        self.metric = config.metric
        self.mode = config.mode
        self.patience = config.patience
        self.min_epochs = config.min_epochs
        self.best = math.inf if config.mode == "min" else -math.inf
        self.best_epoch = 0
        self.stop = False

    def update(self, metrics: Mapping[str, float], *, epoch: int) -> bool:
        """Record one evaluation after ``epoch`` completed epochs.

        Args:
          metrics: Eval scalars by name.
          epoch: Completed epochs.

        Returns:
          stop: Whether training should end.

        Raises:
          KeyError: ``metric`` is not among ``metrics``.

        """
        if self.metric not in metrics:
            raise KeyError(
                f"early-stopping metric {self.metric!r} is not an eval metric; "
                f"available: {sorted(metrics)}",
            )
        value = metrics[self.metric]
        if value < self.best if self.mode == "min" else value > self.best:
            self.best, self.best_epoch = value, epoch
        self.stop = (
            epoch >= self.min_epochs and epoch - self.best_epoch >= self.patience
        )
        return self.stop

    def state_dict(self) -> StateDict:
        """Get the state a resumed run continues from."""
        return {"best": self.best, "best_epoch": self.best_epoch, "stop": self.stop}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Load the record of an earlier run.

        Args:
          state_dict: State as returned by :meth:`state_dict`.

        """
        state = cast(EarlyStopping.StateDict, state_dict)
        self.best, self.best_epoch, self.stop = (
            state["best"],
            state["best_epoch"],
            state["stop"],
        )


class EpochLoopConfig(TrainLoop.Config[_StepT_co, _DataT_co]):
    """Config of an ``EpochTrainLoop``, generic over its step and dataset configs."""

    eval_epochs: Makeable[EvalEpochs] = field(
        default_factory=lambda: PartialConfig(every_epoch),
    )
    """Picks the completed epochs evaluated."""

    early_stopping: Makeable[EarlyStopping] | None = None
    """Ends training when the eval metric stops improving; ``None`` never."""


class EpochTrainLoop(TrainLoop):
    """A loop that evaluates after chosen epochs and may stop on a metric."""

    Config = EpochLoopConfig

    class StateDict(TrainLoop.StateDict):
        """The loop state plus the early-stopping record."""

        early_stopping: NotRequired[EarlyStopping.StateDict]

    def __init__(
        self,
        config: EpochLoopConfig[Makeable[TrainStepProtocol], Makeable[DatasetProtocol]],
    ) -> None:
        self.eval_epochs: EvalEpochs = config.eval_epochs.make()
        self.early_stopping: EarlyStopping | None = (
            None if config.early_stopping is None else config.early_stopping.make()
        )
        super().__init__(config)

    @override
    def _on_epoch_boundary(self) -> None:
        # An evaluation pass may draw from the global generator, so the base
        # evaluates only after the epochs the policy picks.
        evaluate = self.eval_every_epoch
        self.eval_every_epoch = evaluate and self.eval_epochs(
            self.current_epoch,
            total_epochs=self.max_epochs,
        )
        try:
            super()._on_epoch_boundary()
        finally:
            self.eval_every_epoch = evaluate

    @override
    def _publish_eval_metrics(
        self,
        eval_metrics: dict[str, object],
        *,
        eval_time: float,
        step: int,
        is_final: bool,
    ) -> dict[str, float]:
        scalars = super()._publish_eval_metrics(
            eval_metrics,
            eval_time=eval_time,
            step=step,
            is_final=is_final,
        )
        if self.early_stopping is not None:
            self.early_stopping.update(scalars, epoch=self.current_epoch)
        return scalars

    @override
    def _should_stop_early(self) -> bool:
        if self.early_stopping is None:
            return False
        stop = self.early_stopping.stop
        if not dist.is_initialized():
            return stop
        # Eval scalars are per-rank, so rank 0's verdict keeps every rank in step.
        shared = [stop]
        dist.broadcast_object_list(shared, src=0)
        return shared[0]

    @override
    def state_dict(self) -> StateDict:
        state = cast(EpochTrainLoop.StateDict, super().state_dict())
        if self.early_stopping is not None:
            state["early_stopping"] = self.early_stopping.state_dict()
        return state

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        super().load_state_dict(state_dict)
        if self.early_stopping is not None and "early_stopping" in state_dict:
            self.early_stopping.load_state_dict(
                cast(Mapping[str, object], state_dict["early_stopping"]),
            )
