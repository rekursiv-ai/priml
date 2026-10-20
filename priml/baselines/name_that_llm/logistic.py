"""Fit regularized multinomial heads with a quasi-Newton solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self, cast

import math

from configgle import Fig
from numpy.typing import NDArray

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal, Protocol

    import torch

    from priml.math.fused import fma32

    class _DenseResult(Protocol):
        @property
        def x(self) -> NDArray[np.float64]: ...

        @property
        def nit(self) -> int: ...

    class _DenseMinimize(Protocol):
        def __call__[Float: np.float32 | np.float64](
            self,
            objective: Callable[[NDArray[Float]], tuple[float, NDArray[Float]]],
            *,
            x0: NDArray[Float],
            method: Literal["L-BFGS-B"],
            jac: Literal[True],
            options: dict[str, float | int],
        ) -> _DenseResult: ...

    # SciPy retains x0.dtype at callbacks; scipy-stubs incorrectly requires F64.
    minimize: _DenseMinimize
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")
    fma32 = lazy_import("priml.math.fused", "fma32")
    minimize = lazy_import("scipy.optimize", "minimize")


class LBFGS:
    """Configure the dense quasi-Newton solve used by the embedding heads."""

    class Config(Fig["LBFGS"]):
        """Gradient tolerance and update budget."""

        max_iter: int = 5000
        """Accepted-update limit checked after a step; zero still allows one."""

        tolerance_gradient: float = 1e-4
        """Stop when the maximum absolute gradient reaches this bound."""

        tolerance_relative: float = 64 * np.finfo(np.float64).eps
        """Relative objective improvement required to continue updating."""

        max_line_search: int = 50
        """Maximum trial steps per line search."""

    def __init__(self, config: Config) -> None:
        if config.max_iter < 0 or config.max_line_search < 1:
            raise ValueError("Expected nonnegative updates and a positive line search.")
        if (
            math.isnan(config.tolerance_gradient)
            or config.tolerance_gradient < 0
            or math.isnan(config.tolerance_relative)
            or config.tolerance_relative < 0
        ):
            raise ValueError("Expected nonnegative stopping tolerances.")
        self.config = config

    def minimize[Float: np.float32 | np.float64](
        self,
        objective: Callable[[NDArray[Float]], tuple[float, NDArray[Float]]],
        initial: NDArray[Float],
    ) -> tuple[NDArray[np.float64], int]:
        """Minimize an objective supplying its complete analytical gradient.

        Args:
          objective: Value and same-dtype gradient; every call uses initial.dtype.
          initial: Complete initial vector; its dtype controls objective arithmetic.

        Returns:
          parameters: Complete fitted parameters at solver precision.
          updates: Number of accepted quasi-Newton updates.

        """
        result = minimize(
            objective,
            x0=initial,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": self.config.max_iter,
                "gtol": self.config.tolerance_gradient,
                "ftol": self.config.tolerance_relative,
                "maxls": self.config.max_line_search,
            },
        )
        return result.x, result.nit


class LogisticRegression:
    """Fit a dense multinomial affine classifier with L2 regularization."""

    class Config(Fig["LogisticRegression"]):
        """Objective regularization and injectable solver."""

        inverse_regularization: float = 1.0
        """Inverse penalty strength before dividing by the training row count."""

        optimizer: LBFGS.Config = field(default_factory=LBFGS.Config)
        """Quasi-Newton update budget and stopping policy."""

    def __init__(self, config: Config) -> None:
        if (
            math.isnan(config.inverse_regularization)
            or config.inverse_regularization <= 0
        ):
            raise ValueError("Expected positive inverse regularization.")
        self.config = config
        self.n_features_in_ = 0
        self.classes_ = np.empty(0, dtype=np.int64)
        self.coef_ = np.empty((0, 0), dtype=np.float32)
        self.intercept_ = np.empty(0, dtype=np.float32)
        self.n_iter_ = np.empty(0, dtype=np.int32)

    def fit(
        self,
        features: np.ndarray[tuple[int, int], np.dtype[np.float32]],
        targets: NDArray[np.int64],
    ) -> Self:
        """Fit every class jointly from the complete dense training matrix.

        Args:
          features: Dense float32 embedding rows.
          targets: Integer class identities aligned with the training rows.

        Returns:
          classifier: This fitted multinomial classifier.

        """
        if len(features) != len(targets):
            raise ValueError("Expected training features aligned with targets.")
        self.n_features_in_ = features.shape[1]
        self.classes_, encoded = np.unique(targets, return_inverse=True)
        if len(self.classes_) < 3:
            raise ValueError("Expected at least three classes for a multinomial head.")
        objective = _Objective(
            features=np.ascontiguousarray(features),
            targets=encoded,
            penalty=1 / (self.config.inverse_regularization * len(features)),
        )
        parameters, updates = self.config.optimizer.make().minimize(
            objective,
            initial=np.zeros(
                len(self.classes_) * (self.n_features_in_ + 1),
                dtype=np.float32,
            ),
        )
        fitted = parameters.reshape(
            (self.n_features_in_ + 1, len(self.classes_)),
        ).T.astype(np.float32)
        self.coef_ = fitted[:, :-1]
        self.intercept_ = fitted[:, -1]
        self.n_iter_ = np.array([updates], dtype=np.int32)
        return self

    def predict_proba(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return class probabilities in sorted fitted-label order.

        Args:
          features: Dense rows with the fitted embedding columns.

        Returns:
          probabilities: Softmax probabilities in sorted fitted-label order.

        """
        probabilities = features @ self.coef_.T + self.intercept_
        probabilities -= probabilities.max(axis=1, keepdims=True)
        np.exp(probabilities, out=probabilities)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities


@dataclass(frozen=True, slots=True, kw_only=True)
class _Objective:
    features: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    targets: NDArray[np.int64]
    penalty: float

    def __call__[Float: np.float32 | np.float64](
        self,
        parameters: np.ndarray[tuple[int], np.dtype[Float]],
    ) -> tuple[float, np.ndarray[tuple[int], np.dtype[Float]]]:
        columns = self.features.shape[1]
        weights = parameters.reshape((-1, columns + 1), order="F")
        logits = self.features @ weights[:, :-1].astype(np.float32).T + weights[
            :,
            -1,
        ].astype(np.float32)
        losses, point_gradient = _multinomial_points(logits, targets=self.targets)
        flattened = weights[:, :-1].ravel(order="K")
        norm_squared = (
            _norm_squared32(cast("NDArray[np.float32]", flattened))
            if flattened.dtype == np.float32
            else flattened @ flattened
        )
        loss = float(losses.sum() / len(self.features)) + float(
            0.5 * self.penalty * norm_squared,
        )
        point_gradient /= len(self.features)
        gradient = np.empty_like(weights, order="F")
        gradient[:, :-1] = (
            point_gradient.T @ self.features + self.penalty * weights[:, :-1]
        )
        gradient[:, -1] = point_gradient.sum(axis=0)
        return loss, gradient.ravel(order="F")


def _multinomial_points(
    logits: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    targets: NDArray[np.int64],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    losses = np.empty(len(logits), dtype=np.float32)
    gradient = np.empty_like(logits)
    target_ids = cast("list[int]", targets.tolist())
    for row, values in enumerate(logits.tolist()):
        maximum = max(values)
        exponentials = np.array(
            [math.exp(value - maximum) for value in values],
            dtype=np.float32,
        )
        total = np.float32(sum(cast("list[float]", exponentials.tolist())))
        target = target_ids[row]
        losses[row] = np.float32(math.log(float(total)) + maximum) - np.float32(
            values[target],
        )
        gradient[row] = exponentials / total - (np.arange(len(values)) == target)
    return losses, gradient


def _norm_squared32(values: NDArray[np.float32]) -> np.float32:
    # Fixed source accumulation avoids host-selected BLAS rounding changes.
    terms = torch.from_numpy(values)
    boundary32 = len(values) // 32 * 32
    boundary64 = boundary32 // 64 * 64
    accumulators = terms.new_zeros((4, 16))
    for start in range(0, boundary64, 64):
        chunk = terms[start : start + 64].reshape(4, 16)
        accumulators = fma32(chunk, y=chunk, addend=accumulators)
    accumulators = accumulators[:, :8] + accumulators[:, 8:]
    if boundary64 < boundary32:
        chunk = terms[boundary64:boundary32].reshape(4, 8)
        accumulators = fma32(chunk, y=chunk, addend=accumulators)
    reduced = ((accumulators[0] + accumulators[1]) + accumulators[2]) + accumulators[3]
    reduced = reduced[:4] + reduced[4:]
    combined = (reduced[0] + reduced[1]) + (reduced[2] + reduced[3])
    tail = 0.0
    for value in terms[boundary32:] * terms[boundary32:]:
        tail += float(value)
    return np.float32(float(combined) + tail)
