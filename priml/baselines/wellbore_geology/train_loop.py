"""Drive complete folds through Priml and select original-row EMA checkpoints."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Self, cast, override

import math

from configgle import Fig, Makeable, Makes

import torch

from priml.baselines.wellbore_geology.custom_types import (
    BatchLoader,
    ScheduleHorizon,
)
from priml.baselines.wellbore_geology.data import (
    FoldObservations,
    WellLoader,
)
from priml.baselines.wellbore_geology.metrics import PooledRMSE
from priml.baselines.wellbore_geology.train_step import WellboreTrainStep
from priml.metrics.custom_types import MetricProtocol
from priml.model.custom_types import propagate_attr
from priml.runtime import runtime_initialized
from priml.timer import CheckpointableStepTimer
from priml.train.custom_types import CheckpointerProtocol
from priml.train.epoch_loop import EarlyStopping, EpochLoopConfig, EpochTrainLoop


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from torch import Tensor

    from priml.baselines.wellbore_geology.folds import RelativeSurface


def _fold_loader() -> WellLoader.Config:
    config = WellLoader.Config()
    config.dataset.observations = FoldObservations.Config()
    return config


class FoldData:
    """Construct complete fold loaders from printable observation declarations."""

    class Config(Fig["FoldData"]):
        loader: WellLoader.Config = field(default_factory=_fold_loader)
        """Complete training data and owned random-stream configuration."""

        validation: FoldObservations.Config | None = field(
            default_factory=lambda: FoldObservations.Config(leave_one_out=False),
        )
        """Held-out complete observations, or None for training without validation."""

        validation_batch_size: int = 16
        """Complete held-out wells per prediction batch."""

        validation_seed_offset: int = 1
        """Held-out loader generator seed offset from the training loader's seed."""

        device: torch.device | None = None
        """Runtime placement inherited before loader construction."""

    def __init__(self, config: Config) -> None:
        train = config.loader.copy_tree()
        train.pin_memory = (config.device or torch.device("cpu")).type == "cuda"
        self.train: BatchLoader = train.make().loader
        self.validation: BatchLoader | None = None
        self._held_out: WellLoader | None = None
        if config.validation is not None:
            validation = train.copy_tree()
            validation.seed += config.validation_seed_offset
            validation.batch_size = config.validation_batch_size
            validation.shuffle = validation.drop_last = False
            validation.dataset.transform.paths = None
            validation.dataset.transform.images = None
            validation.dataset.observations = config.validation
            self._held_out = validation.make()
            self.validation = self._held_out.loader
        self.timer_epoch = CheckpointableStepTimer()

    @property
    def held_out_surfaces(self) -> Mapping[str, RelativeSurface]:
        """Return every held-out well's geographic surface from its observations."""
        if self._held_out is None:
            raise ValueError("Expected self._held_out is not None.")
        observations = self._held_out.dataset.observations
        assert isinstance(observations, FoldObservations)
        return observations.surfaces

    def train_dataloader(self) -> Iterable[object]:
        """Return the persistent training loader without consuming a draw."""
        return self.train

    def eval_dataloader(self) -> Iterable[object]:
        """Return the held-out loader, or the empty validation split."""
        return self.validation if self.validation is not None else ()

    def state_dict(self) -> dict[str, object]:
        """Expose the consumed epoch count for complete state inspection."""
        return {"timer_epoch": self.timer_epoch.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Reject resume until persistent-worker streams can be restored exactly."""
        del state_dict
        raise NotImplementedError(
            "Exact mid-fold native DataLoader resume is unavailable.",
        )


class WellboreFoldLoop(EpochTrainLoop):
    """Select complete EMA states using unsmoothed original-row RMSE."""

    class Config(  # pyrefly: ignore[bad-override] -- pyrefly reads the base's ``Config = EpochLoopConfig`` alias as an instance variable.
        Makes["WellboreFoldLoop"],
        EpochLoopConfig[WellboreTrainStep.Config, FoldData.Config],
        kw_only=True,
    ):
        step: WellboreTrainStep.Config = field(default_factory=WellboreTrainStep.Config)
        """Shared optimization slots and the complete posterior objective."""

        dataset: FoldData.Config = field(default_factory=FoldData.Config)
        """Reconstructible complete training and held-out observations."""

        metrics_eval: dict[str, Makeable[MetricProtocol]] = field(
            default_factory=lambda: {"": PooledRMSE.Config()},
        )
        """Native metric slots receiving the model bins of each held-out batch."""

        max_epochs: float = math.inf
        """Complete training passes and cosine horizon."""

        best_metric: str = "rmse"
        """Eval metric whose strict decrease selects the fold checkpoint; early
        stopping, when declared, must follow it."""

        evaluate_truncated: bool = False
        """Evaluate the last completed partial epoch when an outer budget stops it."""

        fold: int = 0
        """Flattened repeat-major fold position, selecting its random streams."""

        base_seed: int = -1
        """Family seed that fold offsets build on; -1 leaves it to the family."""

        fold_seed_stride: int = 1000
        """Loader generator seed spacing between consecutive folds."""

        loader_seed_offset: int = 1
        """Loader generator seed offset from the loop seed."""

        checkpointer: Makeable[CheckpointerProtocol] | None = None
        """Publish only selected complete fold checkpoints through the pipeline."""

        num_steps_eval: float = 0
        """Checkpoint selection follows original epoch boundaries."""

        early_train_log_steps: int = 0
        """Retain the original ordinary optimizer-step logging cadence."""

        @override
        def finalize(self) -> Self:
            if self.base_seed >= 0:
                if self.seed is None:
                    self.seed = self.base_seed + self.fold
                loader = self.dataset.loader
                if loader.seed == -1:
                    loader.seed = (
                        self.base_seed
                        + self.fold_seed_stride * self.fold
                        + self.loader_seed_offset
                    )
                if loader.dataset.streams.seed == -1:
                    loader.dataset.streams.seed = self.base_seed + self.fold
            observations = self.dataset.loader.dataset.observations
            if (
                isinstance(observations, FoldObservations.Config)
                and observations.well_ids
            ):
                batches = len(observations.well_ids) / self.dataset.loader.batch_size
                count = (
                    math.floor(batches)
                    if self.dataset.loader.drop_last
                    else math.ceil(batches)
                )
                propagate_attr(
                    self.step.lr_schedule,
                    "T_max",
                    max(1, math.ceil(count / self.step.accumulate_grad_batches))
                    * int(self.max_epochs),
                    protocol=ScheduleHorizon,
                )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if math.isnan(config.max_time) or config.max_time <= 0:
            raise ValueError(
                "Training wall cap must be positive or positive infinity.",
            )
        if min(config.max_epochs, config.step.accumulate_grad_batches) < 1:
            raise ValueError("Epoch and accumulation counts must be positive.")
        if (
            isinstance(config.early_stopping, EarlyStopping.Config)
            and config.early_stopping.metric != config.best_metric
        ):
            raise ValueError(
                f"Early stopping must follow the selection metric {config.best_metric!r}.",
            )
        self.config = config
        self.best_score = math.inf
        self.best_epoch: int | str = "NA"
        self.best_state: dict[str, Tensor] | None = None
        self.val_history: list[dict[str, object]] = []
        super().__init__(config)

    @property
    def native_step(self) -> WellboreTrainStep:
        """Return the actual shared-loop step with its concrete domain type."""
        return cast("WellboreTrainStep", self.step)

    @property
    def fold_data(self) -> FoldData:
        """Return the actual shared-loop dataset with its concrete domain type."""
        return cast("FoldData", self.dataset)

    @override
    def _maybe_eval(self, *, is_final: bool = False, force: bool = False) -> None:
        if (
            is_final
            and self.config.evaluate_truncated
            and not self._terminal_epoch_evaluated
        ):
            force = True
        super()._maybe_eval(is_final=is_final, force=force)

    @override
    def eval(self) -> dict[str, object]:
        """Score every held-out target row and retain the strict best complete state."""
        results = super().eval() if self.fold_data.validation is not None else {}
        if self.fold_data.validation is not None:
            score = results[self.config.best_metric]
            assert isinstance(score, float)
            self.val_history.append(
                {
                    "phase": "train",
                    "epoch": self.current_epoch,
                    "rmse": score,
                    "rmse_raw": score,
                },
            )
            if score >= self.best_score:
                return results
            self.best_score = score
        step = self.native_step
        self.best_epoch = self.current_epoch
        with step.ema.apply_to(step.model):
            self.best_state = {
                name: value.detach().cpu().clone()
                for name, value in step.model.state_dict().items()
            }
        return results


def require_fresh_runtime() -> None:
    """Reject a caller-owned runtime before initializing numerical state."""
    if runtime_initialized():
        raise ValueError(
            "Wellbore native folds do not support a caller-owned runtime; launch in a fresh process.",
        )
