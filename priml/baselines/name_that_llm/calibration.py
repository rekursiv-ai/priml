"""Fit smoothed sigmoid probabilities from held-out affine scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self, cast

import math

from configgle import Fig
from numpy.typing import NDArray

import numpy as np

from priml.baselines.name_that_llm.logistic import LBFGS


class SigmoidCalibration:
    """Fit a scalar sigmoid using class-count-smoothed calibration targets."""

    class Config(Fig["SigmoidCalibration"]):
        """Score rescaling and likelihood minimization."""

        max_magnitude: float = 30.0
        """Largest absolute score before fitting on rescaled inputs."""

        optimizer: LBFGS.Config = field(
            default_factory=lambda: LBFGS.Config(
                max_iter=15_000,
                tolerance_gradient=1e-6,
            ),
        )
        """Quasi-Newton solver for the smoothed binary likelihood."""

    def __init__(self, config: Config) -> None:
        if not math.isfinite(config.max_magnitude) or config.max_magnitude <= 0:
            raise ValueError(
                "Expected a positive finite calibration rescaling threshold.",
            )
        self.config = config
        self.a_ = 0.0
        self.b_ = 0.0

    def fit(self, scores: NDArray[np.float64], targets: NDArray[np.int64]) -> Self:
        """Fit sigmoid slope and offset from complete held-out scores.

        Args:
          scores: One complete out-of-fold score per calibration row.
          targets: Binary integer calibration labels in the same row order.

        Returns:
          calibration: This fitted scalar probability transform.

        """
        if len(scores) != len(targets) or len(scores) == 0:
            raise ValueError(
                "Expected nonempty calibration scores aligned with targets.",
            )
        scores = np.ascontiguousarray(scores)
        maximum = float(abs(scores).max())
        scale = maximum if maximum >= self.config.max_magnitude else 1.0
        inputs = scores if scale == 1 else scores / scale
        negative = float(np.sum(targets <= 0))
        positive = len(targets) - negative
        smoothed = np.zeros_like(scores)
        smoothed[targets > 0] = (positive + 1) / (positive + 2)
        smoothed[targets <= 0] = 1 / (negative + 2)
        fitted, _ = self.config.optimizer.make().minimize(
            _Objective(scores=inputs, targets=smoothed),
            initial=np.array([0.0, math.log((negative + 1) / (positive + 1))]),
        )
        slope, offset = cast("list[float]", fitted.tolist())
        self.a_ = slope / scale
        self.b_ = offset
        return self

    def predict(self, scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return calibrated probabilities for the supplied scores."""
        return _expit(-(self.a_ * scores + self.b_))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Objective:
    scores: NDArray[np.float64]
    targets: NDArray[np.float64]

    def __call__(
        self,
        parameters: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        slope, offset = cast("list[float]", parameters.tolist())
        logits = -(slope * self.scores + offset)
        losses, gradient = _binary_points(logits, targets=self.targets)
        return float(losses.sum()), np.asarray(
            [-(gradient * self.scores).sum(), -gradient.sum()],
            dtype=np.float64,
        )


def _binary_points(
    logits: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return each row's binary log loss and its gradient in the logit."""
    low = logits <= -2
    exponential = np.exp(np.where(low, logits, -logits))
    softplus = np.where(logits > 18, exponential, np.log1p(exponential))
    losses = np.where(
        logits <= -37,
        exponential - targets * logits,
        np.where(low, softplus - targets * logits, softplus + (1 - targets) * logits),
    )
    gradient = np.where(
        logits <= -37,
        exponential - targets,
        np.where(
            low,
            ((1 - targets) * exponential - targets) / (1 + exponential),
            ((1 - targets) - targets * exponential) / (1 + exponential),
        ),
    )
    return losses, gradient


def _expit(values: NDArray[np.float64]) -> NDArray[np.float64]:
    exponential = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + exponential), exponential / (1 + exponential))
