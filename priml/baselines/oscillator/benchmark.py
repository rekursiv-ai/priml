"""Compare forecasters on familiar and shifted simulated trajectories."""

from __future__ import annotations

from dataclasses import field
from typing import Self, TypedDict, override

from configgle import Fig, Makeable
from torch import Tensor

import torch

from priml.baselines.oscillator.data import ForecastData, Windows
from priml.baselines.oscillator.train_step import (
    DLinearForecast,
    FitReport,
    ForecastGeometry,
    ForecastMethod,
)
from priml.math.seed import make_seed
from priml.model.custom_types import propagate_attr


class ForecastScores(TypedDict):
    """Errors of one forecaster on one held-out split."""

    mse: float
    mae: float
    scaled_mse: float


class StudyRecord(TypedDict):
    """One fitted method's budget, fit report, and held-out scores."""

    experiment: str
    seed: int
    reference_fit_trajectories: int
    reference_validation_trajectories: int
    reference_budget: int
    cheap_training_trajectories: int
    timesteps_per_query: int
    training: FitReport
    scores: dict[str, dict[str, ForecastScores]]


class BenchmarkRun(TypedDict):
    """One attempted fit, including failed attempts."""

    run_id: str
    family: str
    system_id: str
    budget: str
    method: str
    access: str
    model_seed: int
    elapsed_seconds: float
    status: str
    error: str | None
    result: StudyRecord | None


class ForecastStudy:
    """Fit a baseline before exposing either test split."""

    class Config(Fig["ForecastStudy"]):
        experiment_name: str = ""
        """Name of the recipe; each experiment factory sets its own."""

        seed: int | None = None
        """Model seed; ``None`` draws one from OS entropy and records it."""

        data: ForecastData.Config = field(default_factory=ForecastData.Config)
        """Simulators, independent splits, and forecasting geometry."""

        method: Makeable[ForecastMethod] = field(default_factory=DLinearForecast.Config)
        """Forecasting method given the same reference-data budget."""

        @override
        def finalize(self) -> Self:
            self.data = self.data.finalized()
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

    def run(self) -> StudyRecord:
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
        scale = self.score_scale(train)
        seed = make_seed() if cfg.seed is None else cfg.seed
        # ``torch.manual_seed`` would also seed CUDA, which this CPU-only fork
        # does not restore, so a study would reseed its caller's GPU stream.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            method = cfg.method.make()
            training = method.fit(
                train,
                validation=validation,
                synthetic=synthetic,
                simulator=self.data.cheap,
            )
        scores: dict[str, dict[str, ForecastScores]] = {}
        for name, states in (
            ("familiar", cfg.data.test),
            ("shifted", cfg.data.shifted),
        ):
            test = self.data.windows(states, simulator=self.data.reference)
            predictions = {
                "method": method.predict(test.media),
                "cheap_simulator": self.data.cheap.forecast_history(
                    test.media,
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
            "seed": seed,
            "reference_fit_trajectories": cfg.data.train.count,
            "reference_validation_trajectories": cfg.data.validation.count,
            "reference_budget": cfg.data.train.count + cfg.data.validation.count,
            "cheap_training_trajectories": cfg.data.synthetic.count,
            "timesteps_per_query": cfg.data.seq_len + cfg.data.pred_len,
            "training": training,
            "scores": scores,
        }

    def score_scale(self, train: Windows) -> Tensor:
        """Return the training-only channel scale used to report forecast errors.

        Args:
          train: Reference windows already included in the query budget.

        Returns:
          scale: Positive per-channel scales shaped (1, 1, channels).

        """
        return train.channel_stats()[1]


def forecast_scores(
    prediction: Tensor,
    *,
    target: Tensor,
    scale: Tensor,
) -> ForecastScores:
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
