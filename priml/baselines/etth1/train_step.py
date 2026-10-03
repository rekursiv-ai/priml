"""Training step for the ETTh1 DLinear baseline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import TYPE_CHECKING, Self, cast, override

from configgle import Makeable, Makes, PartialConfig
from torch import Tensor

import torch

from priml.baselines.etth1.data import Etth1Data
from priml.baselines.etth1.model import DLinear
from priml.loss.custom_types import LossOutput
from priml.loss.simple_loss import SimpleLoss, mse
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

        with torch.no_grad():
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

        with torch.no_grad():
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
        super().__init__(config)

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
        super()._on_epoch_boundary()

        # Save after validation so a resume starts after the full epoch.
        if self.checkpointer is not None and self.local_step > 0:
            self.checkpointer.save(self, step=self.step.global_step)

        if self.stop_early:
            self._terminal_epoch_evaluated = True
            raise StopIteration

    class StateDict(TrainLoop.StateDict):
        """Training state plus the reference early-stopping decision."""

        best_validation_loss: float
        bad_validation_epochs: int
        stop_early: bool
        validation_losses: list[float]

    @override
    def state_dict(self) -> StateDict:
        """Save stopping history together with model, loader, optimizer, and RNG."""
        return {
            **super().state_dict(),
            "best_validation_loss": self.best_validation_loss,
            "bad_validation_epochs": self.bad_validation_epochs,
            "stop_early": self.stop_early,
            "validation_losses": list(self.validation_losses),
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
