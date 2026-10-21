"""Fit existing DLinear and a linear simulator-correction baseline."""

from __future__ import annotations

from dataclasses import field
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypedDict,
    runtime_checkable,
)

import math

from configgle import Fig
from torch import Tensor

import torch

from priml.baselines.etth1.model import DLinear
from priml.baselines.etth1.train_step import Etth1TrainStep
from priml.baselines.oscillator.data import STATE_CHANNELS


if TYPE_CHECKING:
    from priml.baselines.oscillator.data import Oscillator, Windows


class _FitCounts(TypedDict):
    updates: int
    best_update: int
    pretrain_updates: int


class FitReport(_FitCounts, total=False):
    """What a forecasting method reports after fitting."""

    validation_scaled_mse: float


class ForecastMethod(Protocol):
    """Data a baseline can see before the final evaluation."""

    def fit(
        self,
        train: Windows,
        *,
        validation: Windows,
        synthetic: Windows,
        simulator: Oscillator,
    ) -> FitReport:
        """Fit using only the supplied development data.

        Args:
          train: Reference trajectories available for fitting.
          validation: Reference trajectories available for model selection.
          synthetic: Cheap-simulator trajectories available for pretraining.
          simulator: The cheap simulator.

        Returns:
          report: Update counts; a method that selects weights on validation
            adds its best score.

        """
        ...

    def predict(self, media: Tensor) -> Tensor:
        """Forecast from the observed history.

        Args:
          media: Histories shaped (trajectories, seq_len, channels).

        Returns:
          prediction: Forecasts shaped (trajectories, pred_len, channels).

        """
        ...


@runtime_checkable
class ForecastGeometry(Protocol):
    """A method config that must match the study's history and horizon."""

    seq_len: int
    pred_len: int


class DLinearForecast:
    """Train DLinear within a fixed total update budget."""

    class Config(Fig["DLinearForecast"]):
        step: Etth1TrainStep.Config = field(
            default_factory=lambda: Etth1TrainStep.Config(
                model=DLinear.Config(channels=STATE_CHANNELS),
                train_budget_steps=300,
            ),
        )
        """DLinear training code; ``train_budget_steps`` is the total update count."""

        pretrain_updates: int = 0
        """Initial updates using cheap trajectories instead of reference data."""

        batch_size: int = 8
        """Trajectories sampled per training update."""

        validation_every: int = 25
        """Reference-training updates between validation checks."""

        sample_seed: int = 606
        """Random stream for training-batch selection."""

        @property
        def seq_len(self) -> int:
            """Observed timesteps, read from and written to the model."""
            return int(self.step.model.seq_len)  # pyrefly: ignore[unnecessary-type-conversion] -- The stub's type hides the runtime value (numpy/torch scalar or non-str key); the conversion is load-bearing.

        @seq_len.setter
        def seq_len(self, value: int) -> None:
            self.step.model.seq_len = value

        @property
        def pred_len(self) -> int:
            """Forecast timesteps, read from and written to the model."""
            return int(self.step.model.pred_len)  # pyrefly: ignore[unnecessary-type-conversion] -- The stub's type hides the runtime value (numpy/torch scalar or non-str key); the conversion is load-bearing.

        @pred_len.setter
        def pred_len(self, value: int) -> None:
            self.step.model.pred_len = value

        @property
        def updates(self) -> int:
            """Total updates, including pretraining; the step's schedule horizon."""
            return int(self.step.train_budget_steps)

        @updates.setter
        def updates(self, value: int) -> None:
            self.step.train_budget_steps = value

    def __init__(self, config: Config) -> None:
        if config.step.seed is not None:
            raise ValueError(
                "Set the study seed; the nested DLinear step seed must be None.",
            )
        budget = config.step.train_budget_steps
        if not math.isfinite(budget) or budget != int(budget):
            raise ValueError("The step budget must be a finite whole update count.")
        if min(config.updates, config.batch_size, config.validation_every) <= 0:
            raise ValueError("Training counts must be positive.")
        if config.pretrain_updates < 0 or config.pretrain_updates >= config.updates:
            raise ValueError("Pretraining must leave at least one reference update.")
        # ``fit`` counts updates itself, so time and epoch budgets would be ignored.
        for name, value in (
            ("train_budget_sec", config.step.train_budget_sec),
            ("train_budget_epochs", config.step.train_budget_epochs),
        ):
            if value != math.inf:
                raise ValueError(f"DLinear training does not support {name}.")
        self.config = config
        self.step = config.step.make()
        self.mean: Tensor | None = None
        self.scale: Tensor | None = None

    def fit(
        self,
        train: Windows,
        *,
        validation: Windows,
        synthetic: Windows,
        simulator: Oscillator,
    ) -> FitReport:
        """Select weights using validation, never test trajectories.

        Args:
          train: Reference trajectories; they alone set the normalization.
          validation: Reference trajectories scored to choose the weights.
          synthetic: Cheap trajectories for the first ``pretrain_updates``.
          simulator: Unused; DLinear sees only trajectories.

        Returns:
          report: Update counts, the chosen update, and its validation score.

        Raises:
          RuntimeError: The forecaster was already fitted.
          ValueError: The data does not match DLinear's geometry, or no
            validation score was finite.

        """
        del simulator
        if self.step.global_step:
            raise RuntimeError("Build a fresh forecaster for each run.")
        cfg = self.config
        geometry = (cfg.step.model.seq_len, cfg.step.model.channels)
        for windows in (train, validation, synthetic):
            if windows.media.shape[1:] != geometry or windows.label.shape[1:] != (
                cfg.step.model.pred_len,
                cfg.step.model.channels,
            ):
                raise ValueError("Data dimensions do not match DLinear.")
        # Keep validation and test data out of normalization.
        self.mean, self.scale = train.channel_stats()
        train_generator = torch.Generator().manual_seed(cfg.sample_seed)
        cheap_generator = torch.Generator().manual_seed(cfg.sample_seed + 1)
        best_loss = math.inf
        best_update = 0
        best_weights: dict[str, Tensor] = {}
        for update in range(cfg.updates):
            pretraining = update < cfg.pretrain_updates
            source = synthetic if pretraining else train
            indices = torch.randint(
                len(source.media),
                (cfg.batch_size,),
                generator=cheap_generator if pretraining else train_generator,
            )
            self.step.train_step(
                **self.step.preprocess_batch(
                    {
                        "media": (source.media[indices] - self.mean) / self.scale,
                        "label": (source.label[indices] - self.mean) / self.scale,
                    },
                ),
            )
            reference_updates = update + 1 - cfg.pretrain_updates
            if reference_updates > 0 and (
                reference_updates % cfg.validation_every == 0
                or update + 1 == cfg.updates
            ):
                loss = float(
                    ((self.predict(validation.media) - validation.label) / self.scale)
                    .square()
                    .mean(),
                )
                if loss < best_loss:
                    best_loss = loss
                    best_update = update + 1
                    best_weights = {
                        key: value.clone()
                        for key, value in self.step.model.state_dict().items()
                    }
        if not best_weights:
            self.mean = self.scale = None
            raise ValueError("No finite validation checkpoint was found.")
        self.step.model.load_state_dict(best_weights)
        return {
            "updates": self.step.global_step,
            "best_update": best_update,
            "validation_scaled_mse": best_loss,
            "pretrain_updates": cfg.pretrain_updates,
        }

    def predict(self, media: Tensor) -> Tensor:
        """Return forecasts in the original units.

        Args:
          media: Histories shaped (trajectories, seq_len, channels).

        Returns:
          prediction: Forecasts shaped (trajectories, pred_len, channels).

        Raises:
          RuntimeError: ``fit`` has not run.

        """
        if self.mean is None or self.scale is None:
            raise RuntimeError("Fit the model before predicting.")
        self.step.model.eval()
        with torch.no_grad():
            prediction = self.step.net(
                ((media - self.mean) / self.scale).to(self.step.device),
            )
            assert isinstance(prediction, Tensor)
            return prediction.to(media.device) * self.scale + self.mean


class RidgeCorrection:
    """Fit a regularized linear correction to the cheap simulator's forecast."""

    class Config(Fig["RidgeCorrection"]):
        penalty: float = 0.01
        """Ridge penalty on standardized history features."""

    def __init__(self, config: Config) -> None:
        if not math.isfinite(config.penalty) or config.penalty <= 0:
            raise ValueError("Ridge penalty must be finite and positive.")
        self.config = config
        self.simulator: Oscillator | None = None
        self.horizon = 0
        self.weights: Tensor | None = None
        self.feature_mean: Tensor | None = None
        self.feature_scale: Tensor | None = None
        self.target_mean: Tensor | None = None

    def fit(
        self,
        train: Windows,
        *,
        validation: Windows,
        synthetic: Windows,
        simulator: Oscillator,
    ) -> FitReport:
        """Use reference training residuals without tuning on validation or test.

        Args:
          train: Reference trajectories whose residuals the correction fits.
          validation: Unused; the penalty is fixed rather than tuned.
          synthetic: Unused; the correction learns from reference data only.
          simulator: The cheap simulator whose forecasts are corrected.

        Returns:
          report: Zero update counts; a direct solve takes no updates.

        """
        del validation, synthetic
        self.simulator = simulator
        self.horizon = train.label.shape[1]
        features = train.media.flatten(start_dim=1).double()
        self.feature_mean = features.mean(dim=0)
        self.feature_scale = features.std(dim=0, correction=0).clamp_min(1e-6)
        features = (features - self.feature_mean) / self.feature_scale
        targets = (
            (
                train.label
                - simulator.forecast_history(
                    train.media,
                    steps=self.horizon,
                )
            )
            .flatten(start_dim=1)
            .double()
        )
        self.target_mean = targets.mean(dim=0)
        targets = targets - self.target_mean
        gram = features.T @ features
        self.weights = torch.linalg.solve(
            gram + self.config.penalty * torch.eye(gram.shape[0], dtype=gram.dtype),
            features.T @ targets,
        )
        return {"updates": 0, "best_update": 0, "pretrain_updates": 0}

    def predict(self, media: Tensor) -> Tensor:
        """Add the learned correction to the cheap forecast.

        Args:
          media: Histories shaped (trajectories, seq_len, channels).

        Returns:
          prediction: Corrected forecasts shaped (trajectories, horizon, channels).

        Raises:
          RuntimeError: ``fit`` has not run.

        """
        if (
            self.simulator is None
            or self.weights is None
            or self.feature_mean is None
            or self.feature_scale is None
            or self.target_mean is None
        ):
            raise RuntimeError("Fit the correction before predicting.")
        features = (
            media.flatten(start_dim=1).double() - self.feature_mean
        ) / self.feature_scale
        correction = (features @ self.weights + self.target_mean).to(media.dtype)
        return self.simulator.forecast_history(
            media,
            steps=self.horizon,
        ) + correction.reshape(
            media.shape[0],
            self.horizon,
            media.shape[-1],
        )
