"""Compare forecasters on familiar and shifted simulated trajectories."""

from __future__ import annotations

from dataclasses import field
from typing import Self, override

from configgle import Fig, Makeable
from torch import Tensor

import torch

from priml.baselines.oscillator.data import ForecastData
from priml.baselines.oscillator.train_step import (
    DLinearForecast,
    ForecastGeometry,
    ForecastMethod,
)
from priml.model.custom_types import propagate_attr


class ForecastStudy:
    """Fit a baseline before exposing either test split."""

    class Config(Fig["ForecastStudy"]):
        experiment_name: str = ""
        """Name of the recipe; each experiment factory sets its own."""

        seed: int = 2021
        """Model seed; the comparison varies it to measure training variance."""

        data: ForecastData.Config = field(default_factory=ForecastData.Config)
        """Simulators, independent splits, and forecasting geometry."""

        method: Makeable[ForecastMethod] = field(default_factory=DLinearForecast.Config)
        """Forecasting method given the same reference-data budget."""

        @override
        def finalize(self) -> Self:
            propagate_attr(
                self.method,
                "seq_len",
                self.data.seq_len,
                protocol=ForecastGeometry,
            )
            propagate_attr(
                self.method,
                "pred_len",
                self.data.pred_len,
                protocol=ForecastGeometry,
            )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.data = config.data.make()

    def run(self) -> dict[str, object]:
        """Fit the method, then score it on each held-out split.

        Returns:
          record: The recipe, its reference-data budget, the method's fit
            report, and per-split scores for the method and both controls.

        """
        cfg = self.config
        train = self.data.windows(cfg.data.train, simulator=self.data.reference)
        validation = self.data.windows(
            cfg.data.validation,
            simulator=self.data.reference,
        )
        synthetic = self.data.windows(cfg.data.synthetic, simulator=self.data.cheap)
        _, scale = train.channel_stats()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.seed)
            method = cfg.method.make()
            training = method.fit(
                train,
                validation=validation,
                synthetic=synthetic,
                simulator=self.data.cheap,
            )
        scores: dict[str, dict[str, dict[str, float]]] = {}
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
    """Score every forecast cell using a scale fitted on reference training data.

    Nonfinite predictions propagate into the scores rather than being checked
    here, so a diverged forecast reports NaN or inf.

    Args:
      prediction: Forecasts shaped (trajectories, pred_len, channels).
      target: Reference values with the same shape.
      scale: Per-channel training scale shaped (1, 1, channels).

    Returns:
      scores: ``mse`` and ``mae`` in original units, and ``scaled_mse``.

    Raises:
      ValueError: The shapes differ, ``target`` is empty, or ``scale`` is not
        one value per channel.

    """
    if target.ndim != 3 or prediction.shape != target.shape or target.numel() == 0:
        raise ValueError("Predictions and targets must have the same nonempty shape.")
    if scale.shape != (1, 1, target.shape[-1]):
        raise ValueError("Expected one training scale per channel.")
    error = prediction.double() - target.double()
    return {
        "mse": float(error.square().mean()),
        "mae": float(error.abs().mean()),
        "scaled_mse": float((error / scale.double()).square().mean()),
    }
