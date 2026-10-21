"""Fit smoothed sigmoid probabilities from held-out affine scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self, cast

import math

from configgle import Fig
from numpy.typing import NDArray

import numpy as np

from priml.baselines.name_that_llm.logistic import LBFGS
from priml.math.fused import fma64


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


def dot_product(left: NDArray[np.float64], *, right: NDArray[np.float64]) -> float:
    """Reduce calibration vectors in the captured binary64 accumulation order.

    Args:
      left: First complete vector.
      right: Second vector of the same length.

    Returns:
      result: Binary64 scalar independent of the selected BLAS kernel.

    """
    if left.ndim != 1 or left.shape != right.shape:
        raise ValueError("Expected equal-length vectors for a dot product.")
    first, second = (cast("list[float]", values.tolist()) for values in (left, right))
    boundary = len(first) // 16 * 16
    result = _dot_blocks(first, right=second, boundary=boundary) if boundary else 0.0
    for index in range(boundary, len(first)):
        result = fma64(first[index], y=second[index], addend=result)
    return result


def _dot_blocks(left: list[float], *, right: list[float], boundary: int) -> float:
    # The source capture's accumulators and folding order determine its bits.
    accumulators = [0.0] * 32
    boundary32 = boundary // 32 * 32
    for start in range(0, boundary32, 32):
        for lane in range(32):
            accumulators[lane] = fma64(
                left[start + lane],
                y=right[start + lane],
                addend=accumulators[lane],
            )
    folded = [
        accumulators[group * 8 + lane] + accumulators[group * 8 + lane + 4]
        for group in range(4)
        for lane in range(4)
    ]
    if boundary32 < boundary:
        for lane in range(16):
            folded[lane] = fma64(
                left[boundary32 + lane],
                y=right[boundary32 + lane],
                addend=folded[lane],
            )
    reduced = [
        ((folded[lane] + folded[4 + lane]) + folded[8 + lane]) + folded[12 + lane]
        for lane in range(4)
    ]
    return (reduced[0] + reduced[2]) + (reduced[1] + reduced[3])


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
            [dot_product(-gradient, right=self.scores), -gradient.sum()],
            dtype=np.float64,
        )


def _binary_points(
    logits: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    losses = np.empty_like(logits)
    gradient = np.empty_like(logits)
    for row, (score, target) in enumerate(
        zip(
            cast("list[float]", logits.tolist()),
            cast("list[float]", targets.tolist()),
            strict=True,
        ),
    ):
        if score <= -37:
            exponential = math.exp(score)
            losses[row] = exponential - target * score
            gradient[row] = exponential - target
        elif score <= -2:
            exponential = math.exp(score)
            losses[row] = math.log1p(exponential) - target * score
            gradient[row] = ((1 - target) * exponential - target) / (1 + exponential)
        else:
            exponential = math.exp(-score)
            losses[row] = (math.log1p(exponential) if score <= 18 else exponential) + (
                1 - target
            ) * score
            gradient[row] = ((1 - target) - target * exponential) / (1 + exponential)
    return losses, gradient


def _expit(values: NDArray[np.float64]) -> NDArray[np.float64]:
    # NumPy's float64 exp and torch.sigmoid are SIMD kernels that round per ISA; the C
    # library's scalar exp rounds alike on x86 and aarch64. Each branch exponentiates
    # a nonpositive value: Python's exp raises OverflowError above 709.
    probabilities = [
        1 / (1 + math.exp(-value))
        if value >= 0
        else (exponential := math.exp(value)) / (1 + exponential)
        for value in values.reshape(-1).tolist()
    ]
    return np.array(probabilities, dtype=np.float64).reshape(values.shape)
