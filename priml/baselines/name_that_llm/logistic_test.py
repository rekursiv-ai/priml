"""Pin the logistic objective, its probabilities and the quasi-Newton solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import math

from numpy.typing import NDArray

import numpy as np
import pytest
import torch

from priml.baselines.name_that_llm import logistic
from priml.baselines.name_that_llm.logistic import LBFGS, LogisticRegression
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize("target_rows", argvalues=[3, 5])
def test_logistic_rejects_misaligned_targets_before_mutating_fit_state(
    target_rows: int,
) -> None:
    model = LogisticRegression.Config().make()
    with pytest.raises(
        ValueError,
        match=r"^Expected training features aligned with targets\.$",
    ):
        model.fit(
            np.zeros((4, 2), dtype=np.float32),
            targets=np.arange(target_rows, dtype=np.int64),
        )
    assert model.n_features_in_ == 0
    assert model.classes_.size == 0


def test_logistic_probabilities_keep_underflow_subnormal_and_uniform_rows() -> None:
    model = LogisticRegression.Config().make()
    model.coef_ = np.eye(3, M=5, dtype=np.float32)
    model.intercept_ = np.zeros(3, dtype=np.float32)
    probabilities = model.predict_proba(
        np.array(
            [
                [-150, -104, 0, 0, 0],
                [-103, -100, 0, 0, 0],
                [0, -150, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )
    # Logits at or below -104 underflow to zero; -103 and -100 stay subnormal.
    assert probabilities[0].tolist() == [0, 0, 1]
    assert probabilities[1].view(np.uint32).tolist()[:2] == [1, 27]
    assert probabilities[2].tolist() == [0.5, 0, 0.5]
    assert probabilities[3].tolist() == [np.float32(1) / np.float32(3)] * 3


def test_logistic_probabilities_add_each_class_intercept() -> None:
    model = LogisticRegression.Config().make()
    model.coef_ = np.zeros((3, 2), dtype=np.float32)
    model.intercept_ = np.array([0, 0, 1], dtype=np.float32)
    probabilities = model.predict_proba(np.ones((4, 2), dtype=np.float32))
    exponential = np.float32(np.exp(-1.0))
    expected = np.array([exponential, exponential, 1], dtype=np.float32) / (
        exponential + exponential + 1
    )
    assert np.array_equal(probabilities, a2=np.tile(expected, reps=(4, 1)))


def test_probabilities_round_each_float64_score_once() -> None:
    model = LogisticRegression.Config().make()
    # (1 + 2**-12)**2 needs 25 bits: float32 rounds each product, so every float32
    # order, fused or not, scores the first class 0 or 2**-24, not the exact 2**-23.
    model.coef_ = np.array(
        [[1 + 2**-12, -2 - 2**-10, 1 + 2**-12, 0], [0] * 4, [0] * 4],
        dtype=np.float32,
    )
    model.intercept_ = np.array([0, 2**-23, 0], dtype=np.float32)
    probabilities = model.predict_proba(
        np.array([[1 + 2**-12, 1, 1 + 2**-12, 0], [0] * 4], dtype=np.float32),
    )
    # The first row ties its first two classes; the second scores the intercept alone.
    assert probabilities[0, 0] == probabilities[0, 1] > probabilities[0, 2]
    assert probabilities[1, 1] > probabilities[1, 0] == probabilities[1, 2]


def test_fit_sums_gradients_in_float64_and_unpacks_the_solver_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _FirstGradient()
    monkeypatch.setattr(LBFGS, "minimize", solver)
    features = np.zeros((8, 5), dtype=np.float32)
    # A row gives each class it is not labelled 1/4 / 8 = 1/32, so the last class sums
    # 1 + 2**-24 + 2**-50 in column 1: every float32 order rounds 1 + 2**-24 down at
    # the tie, while the exact sum lies above it and rounds to 1 + 2**-23.
    features[:3, 1] = [32, 32 * 2**-24, 32 * 2**-50]
    model = (
        LogisticRegression.Config()
        .make()
        .fit(features, targets=np.tile(np.arange(4, dtype=np.int64), reps=2))
    )
    assert solver.gradients[0].reshape((4, 6), order="F")[3, 1] == 1 + 2**-23
    # The solver returned 0, 1, ... laid out column-major by class, then column.
    assert model.coef_.tolist() == [[k + 4 * j for j in range(5)] for k in range(4)]
    assert model.intercept_.tolist() == [k + 4 * 5 for k in range(4)]
    assert model.n_iter_.tolist() == [7]


@pytest.mark.parametrize(
    ("coefficients", "expected"),
    argvalues=[
        # 2**24 + 3 is a float32 tie that rounds to 2**24 + 4; a float32 sum
        # absorbs each 1 into 2**24.
        ([2**12, 1, 1, 1], 2**24 + 4),
        # (1 + 2**-23)**2 rounded alone leaves 5 + 2**-22, a tie that rounds to 5.
        ([2, 1 + 2**-23], 5 + 2**-21),
    ],
)
def test_regularizer_rounds_the_float64_square_sum_once(
    coefficients: list[float],
    expected: float,
) -> None:
    weights = np.zeros((3, 5), dtype=np.float32)
    weights[0, : len(coefficients)] = coefficients
    objective = logistic._Objective(
        features=np.zeros((2, 4)),
        targets=np.array([0, 2], dtype=np.int64),
        penalty=2,
    )
    loss, _ = objective(weights.ravel(order="F"))
    # Zero features give each row the uniform loss log(3).
    assert loss == float(np.float32(math.log(3))) + expected


def test_objective_rounds_the_penalty_gradient_once() -> None:
    weights = np.zeros((3, 5), dtype=np.float32)
    weights[1, 2] = 1 + 2**-15
    objective = logistic._Objective(
        features=np.zeros((2, 4)),
        targets=np.array([0, 2], dtype=np.int64),
        penalty=1 + 2**-24 - 2**-40,
    )
    _, gradient = objective(weights.ravel(order="F"))
    # A float32 penalty rounds to 1; the exact product lies just above the midpoint
    # 1 + 2**-15 + 2**-24, so one rounding gives 1 + 2**-15 + 2**-23.
    assert gradient.reshape((3, 5), order="F")[1, 2] == 1 + 2**-15 + 2**-23


def test_public_width_objective_returns_the_closed_form_loss_and_gradient() -> None:
    features = np.zeros((2, 4096))
    targets = np.array([0, 17], dtype=np.int64)
    weights = np.zeros((18, 4097), dtype=np.float32, order="F")
    weights[:, :-1] = (
        ((np.arange(73_728, dtype=np.float32) % 509 - 254) / 257)
        .reshape((18, 4096), order="F")
        .astype(np.float32)
    )
    objective = logistic._Objective(features=features, targets=targets, penalty=0.125)
    loss, gradient = objective(weights.ravel(order="F"))
    squares = np.square(weights.astype(np.float64)).ravel().tolist()
    # Zero features leave each row the loss log(18); half the penalty is 1 / 16.
    regularizer = float(np.float32(math.fsum(squares))) / 16
    assert loss == float(np.float32(math.log(18))) + regularizer
    expected = weights * np.float32(0.125)
    probability = np.float32(1 / 18)
    expected[:, -1] = probability
    expected[[0, 17], -1] = np.float32(
        np.float32((probability - np.float32(1)) / 2) + probability / 2,
    )
    report = mismatches(
        {"gradient": torch.from_numpy(expected.ravel(order="F"))},
        actual={"gradient": torch.from_numpy(gradient)},
    )
    assert not report, "\n".join(report)


@pytest.mark.parametrize("classes", argvalues=[1, 2])
def test_multinomial_head_rejects_nonmultinomial_targets(classes: int) -> None:
    model = LogisticRegression.Config().make()
    with pytest.raises(
        ValueError,
        match=r"^Expected at least three classes for a multinomial head\.$",
    ):
        model.fit(
            np.zeros((4, 5), dtype=np.float32),
            targets=np.arange(4, dtype=np.int64) % classes,
        )


@pytest.mark.parametrize("inverse", argvalues=[0, -1, np.nan])
def test_head_rejects_invalid_inverse_regularization(inverse: float) -> None:
    config = LogisticRegression.Config()
    config.inverse_regularization = inverse
    with pytest.raises(
        ValueError,
        match=r"^Expected positive inverse regularization\.$",
    ):
        config.make()


@pytest.mark.parametrize(("updates", "history"), argvalues=[(-1, 10), (5000, 0)])
def test_solver_rejects_invalid_update_and_history_counts(
    updates: int,
    history: int,
) -> None:
    config = LBFGS.Config()
    config.max_iter = updates
    config.history_size = history
    with pytest.raises(
        ValueError,
        match=r"^Expected nonnegative updates and a positive history\.$",
    ):
        config.make()


@pytest.mark.parametrize(
    ("gradient", "change"),
    argvalues=[(-1, 0), (np.nan, 0), (0, -1), (0, np.nan)],
)
def test_solver_rejects_invalid_stopping_tolerances(
    gradient: float,
    change: float,
) -> None:
    config = LBFGS.Config()
    config.tolerance_gradient = gradient
    config.tolerance_change = change
    with pytest.raises(
        ValueError,
        match=r"^Expected nonnegative stopping tolerances\.$",
    ):
        config.make()


@pytest.mark.parametrize("dtype", argvalues=[np.float32, np.float64])
def test_solver_preserves_initial_dtype_at_every_objective_evaluation[
    Float: np.float32 | np.float64,
](dtype: type[Float]) -> None:
    objective = _DtypeObjective(dtype=dtype)
    fitted, updates = (
        LBFGS.Config(max_iter=3)
        .make()
        .minimize(
            objective,
            initial=np.array([2.0, -3.0], dtype=dtype),
        )
    )
    assert updates > 0
    assert len(objective.calls) > 1
    assert all(actual == np.dtype(dtype) for actual in objective.calls)
    assert fitted.dtype == np.float64


def test_solver_returns_an_initial_stationary_point() -> None:
    initial = np.array([2.0, -3.0])
    result, updates = LBFGS.Config().make().minimize(_quadratic, initial=initial)
    np.testing.assert_array_equal(result, initial)
    assert result is not initial
    assert updates == 0


def test_solver_keeps_a_point_with_no_representable_descent() -> None:
    initial = np.full(2, 1e20)
    result, updates = LBFGS.Config().make().minimize(_linear, initial=initial)
    np.testing.assert_array_equal(result, initial)
    assert updates == 1


# Torch also stops once objective calls reach max_iter * 5 // 4: two iterations
# allow two calls, the initial one and the first step's.
@pytest.mark.parametrize(("budget", "expected"), [(0, 0), (1, 1), (2, 1), (3, 2)])
def test_solver_stops_at_the_iteration_and_call_budgets(
    budget: int,
    expected: int,
) -> None:
    result, updates = (
        LBFGS.Config(max_iter=budget, tolerance_gradient=0, tolerance_change=0)
        .make()
        .minimize(_quadratic, initial=np.zeros(2))
    )
    assert updates == expected
    if budget in {1, 2}:
        # The first step is the steepest descent scaled by the gradient's L1 norm.
        np.testing.assert_array_equal(result, np.array([4.0, -6.0]) * (1 / (4 + 6)))
    assert (_quadratic(result)[0] < _quadratic(np.zeros(2))[0]) == bool(budget)


# From zero the quadratic's descent slope is -(4**2 + 6**2) = -52; a change
# tolerance above it ends the first iteration before its step.
@pytest.mark.parametrize(("change", "stepped"), [(52, True), (53, False)])
def test_solver_stops_before_a_slope_within_the_change_tolerance(
    change: float,
    stepped: bool,
) -> None:
    result, updates = (
        LBFGS.Config(tolerance_change=change)
        .make()
        .minimize(_quadratic, initial=np.zeros(2))
    )
    assert updates == 1
    expected = np.array([4.0, -6.0]) * (1 / (4 + 6)) if stepped else np.zeros(2)
    np.testing.assert_array_equal(result, expected)


def test_solver_keeps_the_initial_point_when_search_fails() -> None:
    initial = np.array([2.0, -3.0])
    result, updates = (
        LBFGS.Config().make().minimize(_inconsistent_objective, initial=initial)
    )
    np.testing.assert_array_equal(result, initial)
    assert updates == 1


def test_solver_history_size_changes_the_quasi_newton_path() -> None:
    fits = [
        LBFGS.Config(max_iter=100, tolerance_gradient=1e-8, history_size=history)
        .make()
        .minimize(_rosenbrock, initial=np.array([-1.2, 1.0]))
        for history in (1, 10)
    ]
    (single, single_updates), (_, full_updates) = fits
    assert _rosenbrock(single)[0] < 1e-12
    assert single_updates != full_updates


def test_solver_converges_through_a_full_correction_history() -> None:
    result, updates = (
        LBFGS.Config(max_iter=100, tolerance_gradient=1e-8)
        .make()
        .minimize(_rosenbrock, initial=np.array([-1.2, 1.0]))
    )
    assert updates > 10
    assert _rosenbrock(result)[0] < 1e-12
    np.testing.assert_allclose(result, [1.0, 1.0], rtol=0, atol=1e-6)


@dataclass(slots=True, kw_only=True)
class _FirstGradient:
    """Record the objective's gradient at the start; a non-function stays unbound."""

    gradients: list[NDArray[np.float32]] = field(default_factory=list)

    def __call__(
        self,
        objective: Callable[[NDArray[np.float32]], tuple[float, NDArray[np.float32]]],
        initial: NDArray[np.float32],
    ) -> tuple[NDArray[np.float64], int]:
        self.gradients.append(objective(initial)[1])
        return np.arange(initial.size, dtype=np.float64), 7


@dataclass(slots=True, kw_only=True)
class _DtypeObjective[Float: np.float32 | np.float64]:
    dtype: type[Float]
    calls: list[np.dtype[Float]] = field(default_factory=list)

    def __call__(self, parameters: NDArray[Float]) -> tuple[float, NDArray[Float]]:
        self.calls.append(parameters.dtype)
        assert parameters.dtype == np.dtype(self.dtype)
        return float((parameters * parameters).sum()), 2 * parameters


def _quadratic(parameters: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    delta = parameters - np.array([2.0, -3.0])
    return float((delta * delta).sum()), 2 * delta


def _linear(parameters: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    return float(parameters.sum()), np.ones_like(parameters)


def _inconsistent_objective(
    parameters: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    return 1.0, np.ones_like(parameters)


def _rosenbrock(parameters: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    x, y = cast("list[float]", parameters.tolist())
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2, np.array(
        [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)],
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
