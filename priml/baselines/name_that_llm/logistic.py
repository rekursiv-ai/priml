"""Fit regularized multinomial heads with a quasi-Newton solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self, cast

import math

from configgle import Fig
from numpy.typing import NDArray

import numpy as np


if TYPE_CHECKING:
    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


class LBFGS:
    """Configure the dense quasi-Newton solve used by the embedding heads."""

    class Config(Fig["LBFGS"]):
        """Iteration budget, stopping tolerances, and curvature history."""

        max_iter: int = 5000
        """Iteration limit; torch also caps objective calls at max_iter * 5 // 4."""

        tolerance_gradient: float = 1e-4
        """Stop when the maximum absolute gradient reaches this bound."""

        tolerance_change: float = 64 * np.finfo(np.float64).eps
        """Stop once a step entry, objective change, or descent slope falls below."""

        history_size: int = 10
        """Curvature pairs kept to approximate the inverse Hessian."""

    def __init__(self, config: Config) -> None:
        # Torch checks none of these: a zero history fails at the first curvature
        # update, and a negative one never discards a pair.
        if config.max_iter < 0 or config.history_size < 1:
            raise ValueError("Expected nonnegative updates and a positive history.")
        if (
            math.isnan(config.tolerance_gradient)
            or config.tolerance_gradient < 0
            or math.isnan(config.tolerance_change)
            or config.tolerance_change < 0
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
          initial: Complete initial vector; its dtype controls solver arithmetic.

        Returns:
          parameters: Complete fitted parameters, widened exactly to float64.
          updates: Iterations run, counting a final one that stops before a step.

        """
        parameters = torch.tensor(initial)
        optimizer = torch.optim.LBFGS(
            [parameters],
            lr=1,
            max_iter=self.config.max_iter,
            tolerance_grad=self.config.tolerance_gradient,
            tolerance_change=self.config.tolerance_change,
            history_size=self.config.history_size,
            line_search_fn="strong_wolfe",
        )
        optimizer.step(_Evaluation(objective=objective, parameters=parameters))
        return (
            parameters.numpy().astype(np.float64),
            cast("int", optimizer.state[parameters]["n_iter"]),
        )


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
            features=np.ascontiguousarray(features, dtype=np.float64),
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
        probabilities = _logits(features, coef=self.coef_, intercept=self.intercept_)
        probabilities -= probabilities.max(axis=1, keepdims=True)
        # NumPy's float32 exp differs per ISA; float64 rounded once differs only at
        # float32 near-ties.
        probabilities = np.exp(probabilities.astype(np.float64)).astype(np.float32)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities


@dataclass(frozen=True, slots=True, kw_only=True)
class _Evaluation[Float: np.float32 | np.float64]:
    objective: Callable[[NDArray[Float]], tuple[float, NDArray[Float]]]
    parameters: torch.Tensor

    def __call__(self) -> float:
        loss, gradient = self.objective(
            cast("NDArray[Float]", self.parameters.numpy()),
        )
        self.parameters.grad = torch.from_numpy(gradient)
        return loss


@dataclass(frozen=True, slots=True, kw_only=True)
class _Objective:
    features: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    targets: NDArray[np.int64]
    penalty: float

    def __call__[Float: np.float32 | np.float64](
        self,
        parameters: np.ndarray[tuple[int], np.dtype[Float]],
    ) -> tuple[float, np.ndarray[tuple[int], np.dtype[Float]]]:
        columns = self.features.shape[1]
        weights = parameters.reshape((-1, columns + 1), order="F")
        logits = _logits(
            self.features,
            coef=weights[:, :-1].astype(np.float32),
            intercept=weights[:, -1].astype(np.float32),
        )
        losses, point_gradient = _multinomial_points(logits, targets=self.targets)
        flattened = weights[:, :-1].ravel(order="K")
        # A BLAS dot product here would round in the host kernel's lane order.
        norm_squared = (
            np.square(flattened.astype(np.float64)).sum().astype(flattened.dtype)
        )
        loss = float(losses.sum() / len(self.features)) + float(
            0.5 * self.penalty * norm_squared,
        )
        point_gradient /= len(self.features)
        gradient = np.empty_like(weights, order="F")
        # Summed in float64 and rounded once on assignment, as the logits are.
        gradient[:, :-1] = point_gradient.T @ self.features + self.penalty * weights[
            :,
            :-1,
        ].astype(np.float64)
        gradient[:, -1] = point_gradient.sum(axis=0)
        return loss, gradient.ravel(order="F")


def _logits(
    features: NDArray[np.float32] | NDArray[np.float64],
    coef: NDArray[np.float32],
    intercept: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Return affine class scores, summed in float64 and rounded once to float32."""
    # A float32 BLAS product rounds per kernel (FMA, blocking); float64 rounded once
    # differs only at float32 near-ties.
    return (features.astype(np.float64, copy=False) @ coef.T + intercept).astype(
        np.float32,
    )


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
