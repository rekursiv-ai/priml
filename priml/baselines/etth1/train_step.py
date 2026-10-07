"""Training step for the ETTh1 DLinear baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import TYPE_CHECKING, NotRequired, Self, cast, override

import math

from configgle import Makeable, Makes, PartialConfig
from torch import Tensor

import torch

from priml.baselines.etth1.data import Etth1Data
from priml.baselines.etth1.model import DLinear
from priml.loss.custom_types import LossOutput
from priml.loss.simple_loss import SimpleLoss, mse
from priml.train.ema import NoEMA
from priml.train.train_loop import TrainLoop
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from collections.abc import Mapping

    from priml.train.custom_types import TrainStepOutput


class Etth1TrainStep(TrainStep):
    """DLinear model plus the reference ETTh1 optimization recipe."""

    class Config(
        Makes["Etth1TrainStep"],
        TrainStep.Config[DLinear.Config],
        kw_only=True,
    ):
        """Model and optimizer for the canonical DLinear experiment."""

        model: DLinear.Config = field(default_factory=DLinear.Config)
        """DLinear forecasting model."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=lambda: PartialConfig(
                torch.optim.Adam,
                lr=1e-4,
            ),
        )
        """Adam optimizer matching the reference DLinear recipe."""

        loss: Makeable[Callable[..., LossOutput]] = field(
            default_factory=lambda: SimpleLoss.Config(
                loss_fn=mse,
                kwargs={"reduction": "mean"},
            ),
        )
        """Mean squared error, reduced in the same order as the reference."""

        seed: int | None = None
        """Unsalted reference Torch seed; inherited from the loop when specified."""

        compile: (
            Makeable[Callable[[Callable[..., object]], Callable[..., object]]] | None
        ) = None
        """The canonical reference uses eager forward and backward."""

    def __init__(self, config: Config) -> None:
        """Initialize with the reference DLinear Torch RNG stream."""
        unsupported = {
            "accumulate_grad_batches": config.accumulate_grad_batches != 1,
            "gradient_clip_norm": config.gradient_clip_norm != math.inf,
            "compile": config.compile is not None,
            "dtype_autocast": config.dtype_autocast is not None,
            "autocast_cache_enabled": config.autocast_cache_enabled,
            "ema": type(config.ema) is not NoEMA.Config,
            "skip_step_on_nonfinite_grad": config.skip_step_on_nonfinite_grad,
        }
        for name, enabled in unsupported.items():
            if enabled:
                raise ValueError(
                    f"ETTh1's reference train step does not support {name}.",
                )
        if config.seed is not None:
            # The reference seeds Torch directly.
            torch.manual_seed(config.seed)
        super().__init__(config)

    @property
    def net(self) -> DLinear:
        """Return the forecasting model under its concrete type."""
        return cast(DLinear, self.model)

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """Run one DLinear optimization step."""
        media = batch["media"]
        assert isinstance(media, Tensor)

        label = batch["label"]
        assert isinstance(label, Tensor)

        self.model.train()
        self.optimizer.zero_grad()

        with self.timer_forward:
            prediction = self.net(media)
            loss = self.loss(prediction, **batch)["loss"]

        loss.backward()

        with self.timer_step:
            self.apply_learning_rate()
            self.optimizer.step()

        return {
            "loss": loss.detach().reshape(1),
            "model": prediction.detach(),
        }

    @override
    def train_loss(self, **batch: object) -> TrainStepOutput:
        """Compute training loss without updating parameters."""
        media = batch["media"]
        assert isinstance(media, Tensor)

        label = batch["label"]
        assert isinstance(label, Tensor)

        self.model.train()

        with self.timer_forward, torch.no_grad():
            prediction = self.net(media)
            loss = self.loss(prediction, **batch)["loss"]

        return {
            "loss": loss.detach().reshape(1),
            "model": prediction.detach(),
        }

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Compute forecasting MSE without updating parameters."""
        media = batch["media"]
        assert isinstance(media, Tensor)

        label = batch["label"]
        assert isinstance(label, Tensor)

        self.model.eval()

        with self.timer_eval, torch.no_grad():
            prediction = self.net(media)
            loss = self.loss(prediction, **batch)["loss"]

        return {
            "loss": loss.detach().reshape(1),
            "model": prediction.detach(),
        }


class Etth1TrainLoop(TrainLoop):
    """Training loop reproducing the reference DLinear stopping rule."""

    class Config(
        Makes["Etth1TrainLoop"],
        TrainLoop.Config[Etth1TrainStep.Config, Etth1Data.Config],
    ):
        """Training loop with ETTh1 and DLinear installed."""

        step: Etth1TrainStep.Config = field(
            default_factory=Etth1TrainStep.Config,
        )
        """DLinear model and optimization recipe."""

        dataset: Etth1Data.Config = field(
            default_factory=Etth1Data.Config,
        )
        """ETTh1 forecasting dataset."""

        patience: int = 3
        """Reference early-stopping patience in validation epochs."""

        @override
        def finalize(self) -> Self:
            self.step.model.seq_len = self.dataset.seq_len
            self.step.model.pred_len = self.dataset.pred_len
            self.step.model.channels = self.dataset.channels
            if self.step.seed is None:
                self.step.seed = self.seed
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.patience <= 0:
            raise ValueError("Early-stopping patience must be positive.")
        self.patience = config.patience
        self.best_validation_loss = float("inf")
        self.bad_validation_epochs = 0
        self.stop_early = False
        self.validation_losses: list[float] = []
        self._pending_epoch_completion = False
        super().__init__(config)

    @override
    def train(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot train a closed TrainLoop.")
        try:
            if not self.eval_only:
                self._complete_pending_epoch()
            super().train()
        finally:
            self.close()

    @override
    def _do_train_step(self, batch: dict[str, object]) -> None:
        super()._do_train_step(batch)
        self._pending_epoch_completion = cast(
            Etth1Data,
            self.dataset,
        ).train_epoch_complete

    def _complete_pending_epoch(self) -> bool:
        if not self._pending_epoch_completion:
            return False
        cast(Etth1Data, self.dataset).finish_train_epoch()
        self.train_loader = None
        self.train_iter = None
        self._pending_epoch_completion = False
        self._on_epoch_boundary()
        return True

    @override
    def _get_next_batch(self) -> dict[str, object]:
        self._complete_pending_epoch()
        if self._training and (
            self.current_epoch >= self.max_epochs or self.stop_early
        ):
            raise StopIteration
        return super()._get_next_batch()

    @override
    def _maybe_eval(self, *, is_final: bool = False, force: bool = False) -> None:
        if not force and self._complete_pending_epoch() and self.eval_every_epoch:
            return
        super()._maybe_eval(is_final=is_final, force=force)

    @override
    def _publish_eval_metrics(
        self,
        eval_metrics: dict[str, object],
        *,
        eval_time: float,
        step: int,
        is_final: bool,
    ) -> dict[str, float]:
        scalar_metrics = super()._publish_eval_metrics(
            eval_metrics,
            eval_time=eval_time,
            step=step,
            is_final=is_final,
        )

        validation_loss = scalar_metrics["total_loss"]
        self.validation_losses.append(validation_loss)

        if validation_loss <= self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.bad_validation_epochs = 0
        else:
            self.bad_validation_epochs += 1

        if self.bad_validation_epochs >= self.patience:
            self.stop_early = True

        return scalar_metrics

    @override
    def _should_stop_early(self) -> bool:
        return self.stop_early

    @override
    def _on_epoch_boundary(self) -> None:
        self._last_boundary_epoch = self.current_epoch
        self.step.on_epoch_end()
        is_final = (
            self.current_epoch >= self.max_epochs
            or self.step.global_step >= self.max_steps
        )
        if self.eval_every_epoch:
            try:
                self._maybe_eval(is_final=is_final, force=True)
            except BaseException:
                self._save_after_evaluation_error(is_final=is_final)
                raise
            self._terminal_epoch_evaluated = is_final or self.stop_early

        # Save after validation so a resume starts after the full epoch.
        if self.checkpointer is not None:
            self.checkpointer.save(self, step=self.step.global_step)

    class StateDict(TrainLoop.StateDict):
        """Training state plus the reference early-stopping decision."""

        best_validation_loss: float
        bad_validation_epochs: int
        stop_early: bool
        validation_losses: list[float]
        pending_epoch_completion: NotRequired[bool]

    @override
    def state_dict(self) -> StateDict:
        """Save stopping history together with model, loader, optimizer, and RNG."""
        return {
            **super().state_dict(),
            "best_validation_loss": self.best_validation_loss,
            "bad_validation_epochs": self.bad_validation_epochs,
            "stop_early": self.stop_early,
            "validation_losses": list(self.validation_losses),
            "pending_epoch_completion": self._pending_epoch_completion,
        }

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore stopping history before continuing a saved run."""
        if "best_validation_loss" not in state_dict:
            raise ValueError(
                "This legacy ETTh1 checkpoint lacks resume state. "
                "Evaluate it with scripts.evaluate or start in a new run directory.",
            )
        super().load_state_dict(state_dict)
        state = cast(Etth1TrainLoop.StateDict, state_dict)
        self.best_validation_loss = state["best_validation_loss"]
        self.bad_validation_epochs = state["bad_validation_epochs"]
        self.stop_early = state["stop_early"]
        self.validation_losses = list(state["validation_losses"])
        self._pending_epoch_completion = state.get(
            "pending_epoch_completion",
            cast(Etth1Data, self.dataset).train_epoch_complete,
        )
        self.train_loader = None
        self.train_iter = None
        self._terminal_epoch_evaluated = False
        self._last_boundary_epoch = self.current_epoch
        self._last_eval_step = -1
        self._last_cadence_step = self.step.global_step
