"""Fit existing DLinear and a linear simulator-correction baseline."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Protocol

import math

from configgle import Fig
from torch import Tensor

import torch

from priml.baselines.etth1.train_step import Etth1TrainStep


if TYPE_CHECKING:
    from priml.baselines.oscillator.data import Oscillator, Windows


class ForecastMethod(Protocol):
    """Data a baseline can see before the final evaluation."""

    def fit(
        self,
        train: Windows,
        *,
        validation: Windows,
        synthetic: Windows,
        simulator: Oscillator,
    ) -> dict[str, float | int]:
        """Fit using only the supplied development data."""
        ...

    def predict(self, media: Tensor) -> Tensor:
        """Forecast from the observed history."""
        ...


class DLinearForecast:
    """Train DLinear within a fixed total update budget."""

    class Config(Fig["DLinearForecast"]):
        step: Etth1TrainStep.Config = field(default_factory=Etth1TrainStep.Config)
        """Existing DLinear training code and optimizer."""
        updates: int = 300
        """Total updates, including any simulator pretraining."""
        pretrain_updates: int = 0
        """Initial updates using cheap trajectories instead of reference data."""
        batch_size: int = 8
        """Trajectories sampled per training update."""
        validation_every: int = 25
        """Reference-training updates between validation checks."""
        sample_seed: int = 606
        """Random stream for training-batch selection."""

    def __init__(self, config: Config) -> None:
        if min(config.updates, config.batch_size, config.validation_every) <= 0:
            raise ValueError("Training counts must be positive.")
        if not 0 <= config.pretrain_updates < config.updates:
            raise ValueError("Pretraining must leave at least one reference update.")
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
    ) -> dict[str, float | int]:
        """Select weights using validation, never test trajectories."""
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
        values = torch.cat((train.media, train.label), dim=1)
        self.mean = values.mean(dim=(0, 1), keepdim=True)
        self.scale = values.std(dim=(0, 1), correction=0, keepdim=True).clamp_min(1e-6)
        train_generator = torch.Generator().manual_seed(cfg.sample_seed)
        cheap_generator = torch.Generator().manual_seed(cfg.sample_seed + 1)
        best_loss = math.inf
        best_update = 0
        best_weights: dict[str, Tensor] = {}
        for update in range(cfg.updates):
            source = synthetic if update < cfg.pretrain_updates else train
            generator = (
                cheap_generator if update < cfg.pretrain_updates else train_generator
            )
            indices = torch.randint(
                len(source.media),
                (cfg.batch_size,),
                generator=generator,
            )
            result = self.step.train_step(
                media=(source.media[indices] - self.mean) / self.scale,
                label=(source.label[indices] - self.mean) / self.scale,
            )
            if not torch.isfinite(result["loss"]).all():
                raise ValueError("Training produced a nonfinite loss.")
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
            raise ValueError("No finite validation checkpoint was found.")
        self.step.model.load_state_dict(best_weights)
        return {
            "updates": self.step.global_step,
            "best_update": best_update,
            "validation_scaled_mse": best_loss,
            "pretrain_updates": cfg.pretrain_updates,
        }

    def predict(self, media: Tensor) -> Tensor:
        """Return forecasts in the original units."""
        if self.mean is None or self.scale is None:
            raise RuntimeError("Fit the model before predicting.")
        self.step.model.eval()
        with torch.no_grad():
            prediction = self.step.net((media - self.mean) / self.scale)
            assert isinstance(prediction, Tensor)
            return prediction * self.scale + self.mean


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
    ) -> dict[str, float | int]:
        """Use reference training residuals without tuning on validation or test."""
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
                - simulator.forecast(
                    train.media[:, -1],
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
        return {"updates": 0}

    def predict(self, media: Tensor) -> Tensor:
        """Add the learned correction to the cheap forecast."""
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
        return self.simulator.forecast(
            media[:, -1],
            steps=self.horizon,
        ) + correction.reshape(
            media.shape[0],
            self.horizon,
            media.shape[-1],
        )
