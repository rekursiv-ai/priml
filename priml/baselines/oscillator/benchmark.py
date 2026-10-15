"""Compare forecasters on familiar and shifted simulated trajectories."""

from __future__ import annotations

from dataclasses import field
from typing import Self, override

from configgle import Fig, Makeable
from torch import Tensor

import torch

from priml.baselines.oscillator.data import ForecastData
from priml.baselines.oscillator.train_step import DLinearForecast, ForecastMethod


class ForecastStudy:
    """Fit a baseline before exposing either test split."""

    class Config(Fig["ForecastStudy"]):
        experiment_name: str = "exp000"
        """Name of the baseline recipe."""
        seed: int = 2021
        """Model seed; comparison runs vary this to measure training variance."""
        data: ForecastData.Config = field(default_factory=ForecastData.Config)
        """Simulators, independent splits, and forecasting geometry."""
        method: Makeable[ForecastMethod] = field(default_factory=DLinearForecast.Config)
        """Forecasting method given the same reference-data budget."""

        @override
        def finalize(self) -> Self:
            if isinstance(self.method, DLinearForecast.Config):
                self.method.step.seed = self.seed
                self.method.step.train_budget_steps = self.method.updates
                self.method.step.model.seq_len = self.data.seq_len
                self.method.step.model.pred_len = self.data.pred_len
                # Oscillator states contain position and velocity.
                self.method.step.model.channels = 2
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.data = config.data.make()

    def run(self) -> dict[str, object]:
        """Return separate scores without using held-out targets during fitting."""
        cfg = self.config
        train = self.data.windows(cfg.data.train, simulator=self.data.reference)
        validation = self.data.windows(
            cfg.data.validation,
            simulator=self.data.reference,
        )
        synthetic = self.data.windows(cfg.data.synthetic, simulator=self.data.cheap)
        values = torch.cat((train.media, train.label), dim=1)
        scale = values.std(dim=(0, 1), correction=0, keepdim=True).clamp_min(1e-6)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed)
            method = cfg.method.make()
            training = method.fit(
                train,
                validation=validation,
                synthetic=synthetic,
                simulator=self.data.cheap,
            )
        scores = {}
        for name, states in (
            ("familiar", cfg.data.test),
            ("shifted", cfg.data.shifted),
        ):
            test = self.data.windows(states, simulator=self.data.reference)
            predictions = {
                "method": method.predict(test.media),
                "cheap_simulator": self.data.cheap.forecast(
                    test.media[:, -1],
                    steps=cfg.data.pred_len,
                ),
                "last_value": test.media[:, -1:].expand_as(test.label),
            }
            scores[name] = {
                method_name: forecast_scores(prediction, target=test.label, scale=scale)
                for method_name, prediction in predictions.items()
            }
        return {
            "experiment": cfg.experiment_name,
            "seed": cfg.seed,
            "reference_fit_trajectories": cfg.data.train.count,
            "reference_validation_trajectories": cfg.data.validation.count,
            "reference_budget": cfg.data.train.count + cfg.data.validation.count,
            "cheap_training_trajectories": cfg.data.synthetic.count,
            "timesteps_per_query": cfg.data.seq_len + cfg.data.pred_len,
            "training": training,
            "scores": scores,
        }


def forecast_scores(
    prediction: Tensor,
    *,
    target: Tensor,
    scale: Tensor,
) -> dict[str, float]:
    """Score every forecast cell using a scale fitted on reference training data."""
    if target.ndim != 3 or prediction.shape != target.shape or target.numel() == 0:
        raise ValueError("Predictions and targets must have the same nonempty shape.")
    if scale.shape != (1, 1, target.shape[-1]) or not (scale > 0).all():
        raise ValueError("Expected a positive training scale for each channel.")
    if not all(torch.isfinite(value).all() for value in (prediction, target, scale)):
        raise ValueError("Scoring requires finite predictions, targets, and scales.")
    error = prediction.double() - target.double()
    return {
        "mse": float(error.square().mean()),
        "mae": float(error.abs().mean()),
        "scaled_mse": float((error / scale.double()).square().mean()),
    }
