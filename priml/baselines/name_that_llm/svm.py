"""Fit sparse squared-hinge classifiers with native dual coordinate updates."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Self, cast

import math

from configgle import Fig

import numpy as np


if TYPE_CHECKING:
    from numpy.random import RandomState
    from numpy.typing import NDArray

    from priml.baselines.name_that_llm.sparse import CsrMatrix


class DualCoordinate:
    """Configure sparse dual coordinate minimization."""

    class Config(Fig["DualCoordinate"]):
        """Sweep budget and projected-gradient stopping rule."""

        max_iter: int = 5000
        """Maximum complete shuffled coordinate sweeps."""

        tolerance: float = 1e-4
        """Maximum projected-gradient range permitting termination."""

    def __init__(self, config: Config) -> None:
        if config.max_iter < 0:
            raise ValueError("Expected a nonnegative sweep budget.")
        if not math.isfinite(config.tolerance) or config.tolerance <= 0:
            raise ValueError("Expected a positive finite projected-gradient tolerance.")
        self.config = config

    def minimize(
        self,
        rows: list[list[tuple[int, float]]],
        signs: list[float],
        *,
        width: int,
        inverse_regularization: float,
        random: RandomState,
    ) -> tuple[list[float], int]:
        """Minimize a binary squared-hinge dual over sparse augmented rows.

        Args:
          rows: All sparse training rows, including the regularized intercept.
          signs: One positive or negative unit label per row.
          width: Complete augmented parameter width.
          inverse_regularization: Squared-hinge objective multiplier.
          random: Continuing MT19937 coordinate-shuffle stream.

        Returns:
          weights: Complete fitted augmented affine coefficients.
          sweeps: Number of completed shuffled coordinate sweeps.

        """
        diagonal = 0.5 / inverse_regularization
        curvature = [diagonal for _ in rows]
        for row, terms in enumerate(rows):
            for _, value in terms:
                curvature[row] += value * value
        weights = [0.0] * width
        dual = [0.0] * len(rows)
        order = list(range(len(rows)))
        active = len(rows)
        previous_maximum = math.inf
        sweeps = 0
        while sweeps < self.config.max_iter:
            maximum, minimum = -math.inf, math.inf
            for position in range(active):
                chosen = position + _bounded(random, limit=active - position)
                order[position], order[chosen] = order[chosen], order[position]
            position = 0
            while position < active:
                row = order[position]
                gradient = 0.0
                for column, value in rows[row]:
                    gradient += weights[column] * value
                gradient = gradient * signs[row] - 1
                gradient += dual[row] * diagonal
                if dual[row] == 0 and gradient > previous_maximum:
                    active -= 1
                    order[position], order[active] = order[active], order[position]
                    continue
                projected = min(gradient, 0.0) if dual[row] == 0 else gradient
                maximum, minimum = max(maximum, projected), min(minimum, projected)
                if abs(projected) > 1e-12:
                    previous = dual[row]
                    dual[row] = max(dual[row] - gradient / curvature[row], 0.0)
                    increment = (dual[row] - previous) * signs[row]
                    for column, value in rows[row]:
                        weights[column] += increment * value
                position += 1
            sweeps += 1
            if maximum - minimum <= self.config.tolerance:
                if active == len(rows):
                    break
                active = len(rows)
                previous_maximum = math.inf
                continue
            previous_maximum = maximum
            if previous_maximum <= 0:
                previous_maximum = math.inf
        return weights, sweeps


class LinearSVC:
    """Fit regularized squared-hinge one-versus-rest sparse classifiers."""

    class Config(Fig["LinearSVC"]):
        """Squared-hinge penalty, initialization seed, and injected optimizer."""

        inverse_regularization: float = 1.0
        """Multiplier of the squared-hinge data objective."""

        optimizer: DualCoordinate.Config = field(default_factory=DualCoordinate.Config)
        """Native dual coordinate minimization and stopping policy."""

        seed: int = 42
        """Seed for the continuing coordinate-shuffle stream."""

    def __init__(self, config: Config) -> None:
        if (
            not math.isfinite(config.inverse_regularization)
            or config.inverse_regularization <= 0
        ):
            raise ValueError("Expected positive finite inverse regularization.")
        self.config = config
        self.n_features_in_ = 0
        self.classes_ = np.empty(0, dtype=np.int64)
        self.coef_ = np.empty((0, 0), dtype=np.float64)
        self.intercept_ = np.empty(0, dtype=np.float64)
        self.n_iter_ = 0

    def fit(self, features: CsrMatrix, targets: NDArray[np.int64]) -> Self:
        """Fit every class with a continuing shuffle stream.

        Args:
          features: Sparse float64 document features in their stored order.
          targets: Class identities aligned with every training row.

        Returns:
          classifier: This fitted sparse classifier.

        """
        if features.shape[0] != len(targets):
            raise ValueError("Expected feature rows aligned with targets.")
        self.n_features_in_ = features.shape[1]
        self.classes_ = np.unique(targets)
        if len(self.classes_) < 3:
            raise ValueError(
                "Expected at least three classes for the multiclass branch.",
            )
        grouped = features.take_rows(np.argsort(targets, kind="stable"))
        data = grouped.data.tolist()
        indices = grouped.indices.tolist()
        boundaries = grouped.indptr.tolist()
        rows = [
            [
                (indices[position], data[position])
                for position in range(boundaries[row], boundaries[row + 1])
            ]
            + [(self.n_features_in_, 1.0)]
            for row in range(len(targets))
        ]
        labels = cast("list[int]", targets[np.argsort(targets, kind="stable")].tolist())
        seed_solver = int(  # pyrefly: ignore[unnecessary-type-conversion] -- The stub's type hides the runtime value (numpy/torch scalar or non-str key); the conversion is load-bearing.
            np.random.RandomState(self.config.seed).randint(np.iinfo(np.int32).max),
        )
        random = np.random.RandomState(seed_solver)
        optimizer = self.config.optimizer.make()
        parameters: list[list[float]] = []
        self.n_iter_ = 0
        for label in cast("list[int]", self.classes_.tolist()):
            weights, sweeps = optimizer.minimize(
                rows,
                signs=[1.0 if value == label else -1.0 for value in labels],
                width=self.n_features_in_ + 1,
                inverse_regularization=self.config.inverse_regularization,
                random=random,
            )
            parameters.append(weights)
            self.n_iter_ = max(self.n_iter_, sweeps)
        fitted = np.array(parameters, dtype=np.float64)
        self.coef_ = fitted[:, :-1]
        self.intercept_ = fitted[:, -1]
        return self

    def decision_function(self, features: CsrMatrix) -> NDArray[np.float64]:
        """Return one score per fitted class for every document.

        Args:
          features: Sparse float64 document features in their stored order.

        Returns:
          scores: Dense binary64 scores in fitted-class order.

        """
        if len(self.classes_) == 0 or features.shape[1] != self.n_features_in_:
            raise ValueError("Expected fitted classifier feature columns.")
        # Products summed in stored order; a BLAS product would round per host.
        rows = np.repeat(np.arange(features.shape[0]), repeats=np.diff(features.indptr))
        scores = np.zeros((features.shape[0], len(self.classes_)))
        np.add.at(scores, rows, features.data[:, None] * self.coef_.T[features.indices])
        return scores + self.intercept_


def _bounded(random: RandomState, *, limit: int) -> int:
    product = int(random.randint(0, 1 << 32, dtype=np.uint32)) * limit
    remainder = product & 0xFFFFFFFF
    threshold = ((1 << 32) - limit) % limit
    while remainder < threshold:
        product = int(random.randint(0, 1 << 32, dtype=np.uint32)) * limit
        remainder = product & 0xFFFFFFFF
    return product >> 32
