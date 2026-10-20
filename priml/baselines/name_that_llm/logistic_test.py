"""Replay whole fitted heads after three updates and after convergence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import pytest
import torch

from priml.baselines.name_that_llm import logistic
from priml.baselines.name_that_llm.logistic import LBFGS, LogisticRegression
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from numpy.typing import NDArray


_CWD: Final = Path(__file__).resolve().parent


@pytest.mark.parametrize("target_rows", argvalues=[3, 5])
def test_logistic_rejects_misaligned_targets_before_mutating_fit_state(
    target_rows: int,
) -> None:
    model = LogisticRegression.Config().make()
    with pytest.raises(ValueError, match="aligned"):
        model.fit(
            np.zeros((4, 2), dtype=np.float32),
            targets=np.arange(target_rows, dtype=np.int64),
        )
    assert model.n_features_in_ == 0
    assert model.classes_.size == 0


@pytest.mark.parametrize("max_iter", argvalues=[3, 5000])
@pytest.mark.parametrize("corrupt", argvalues=[False, True])
def test_logistic_replays_complete_fitted_state_and_probabilities(
    max_iter: int,
    corrupt: bool,
) -> None:
    fit = np.sin(np.arange(45).reshape(9, 5) / 7).astype(np.float32)
    predict = np.cos(np.arange(20).reshape(4, 5) / 11).astype(np.float32)
    targets = np.tile(np.array([2, 5, 9], dtype=np.int64), reps=3)
    config = LogisticRegression.Config()
    config.inverse_regularization = 0.002  # The C of the minted source records.
    config.optimizer.max_iter = max_iter
    model = config.make().fit(fit, targets=targets)
    if corrupt:
        model.coef_[:1, :1] = np.nextafter(model.coef_[:1, :1], np.inf)
    actual = {"predict": torch.from_numpy(model.predict_proba(predict))}
    for name, value in cast("dict[str, object]", vars(model)).items():
        if name.endswith("_"):
            assert isinstance(value, (int, np.generic, np.ndarray))
            actual[name] = torch.from_numpy(np.asarray(value))
    prefix = f"{max_iter}/"
    expected = {
        name.removeprefix(prefix): value
        for name, value in read_tensors(_CWD / "testdata" / "logistic.pt").items()
        if name.startswith(prefix)
    }
    report = mismatches(expected, actual=actual)
    if corrupt:
        assert any(item.startswith("coef_:") for item in report), report
    else:
        assert not report, "\n".join(report)


@pytest.mark.parametrize("corrupt", argvalues=[False, True])
def test_logistic_probabilities_keep_source_underflow_and_normal_tail_bits(
    corrupt: bool,
) -> None:
    model = LogisticRegression.Config().make()
    model.coef_ = np.eye(3, M=5, dtype=np.float32)
    model.intercept_ = np.zeros(3, dtype=np.float32)
    if corrupt:
        model.coef_[:1, :1] = np.nextafter(model.coef_[:1, :1], np.inf)
    actual = model.predict_proba(
        np.array(
            [
                [-150, -104, 0, 0, 0],
                [-103, -100, 0, 0, 0],
                [-90, -88, 0, 0, 0],
                [-37, -18, 0, 0, 0],
                [-2, -0.5, 0, 0, 0],
                [-0.0001, -0.00001, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )
    # These are all 21 binary32 values from the unchanged x86 public classifier.
    expected = (
        np.array(
            [
                0x00000000,
                0x00000000,
                0x3F800000,
                0x00000001,
                0x0000001B,
                0x3F800000,
                0x0008EC28,
                0x0041EDC4,
                0x3F800000,
                0x24C4C239,
                0x3282D315,
                0x3F800000,
                0x3D9F1EDC,
                0x3EB2483F,
                0x3F12F806,
                0x3EAAA7E7,
                0x3EAAABD5,
                0x3EAAAC45,
                0x3EAAAAAB,
                0x3EAAAAAB,
                0x3EAAAAAB,
            ],
            dtype=np.uint32,
        )
        .view(np.float32)
        .reshape((7, 3))
    )
    report = mismatches(
        {"probabilities": torch.from_numpy(expected)},
        actual={"probabilities": torch.from_numpy(actual)},
    )
    assert bool(report) is corrupt, report


@pytest.mark.parametrize(
    ("length", "expected"),
    argvalues=[
        (6, "0x1.6fc1cc0000000p+2"),
        (15, "0x1.bb82200000000p+3"),
        (16, "0x1.d72ecc0000000p+3"),
        (17, "0x1.f2a04c0000000p+3"),
        (73_727, "0x1.7817580000000p+14"),
        (73_728, "0x1.78193e0000000p+14"),
        (73_729, "0x1.781b2a0000000p+14"),
    ],
)
def test_regularizer_replays_complete_captured_source_scalars(
    length: int,
    expected: str,
) -> None:
    coefficients = ((np.arange(length, dtype=np.float32) % 509 - 254) / 257).astype(
        np.float32,
    )
    # Unchanged x86 source captures cover short vectors, tails and public width.
    assert float(logistic._norm_squared32(coefficients)).hex() == expected


def test_regularizer_fuses_each_binary32_accumulation() -> None:
    coefficients = np.zeros(128, dtype=np.float32)
    coefficients[0], coefficients[64] = 2, 1 + 2**-23
    expected = np.float32(5 + 2**-21)
    assert logistic._norm_squared32(coefficients) == expected
    unfused = np.float32(4 + float(np.float32((1 + 2**-23) * (1 + 2**-23))))
    assert unfused != expected


@pytest.mark.parametrize("dtype", argvalues=[torch.float32, torch.float64])
def test_regularizer_keeps_the_source_block_boundary_and_input_precision(
    dtype: torch.dtype,
) -> None:
    coefficients = np.zeros(32, dtype=np.float32)
    coefficients[0] = 2**12
    coefficients[[4, 8, 12]] = 1
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        actual = logistic._norm_squared32(coefficients)
    finally:
        torch.set_default_dtype(previous)
    assert actual == np.float32(2**24 + 2)
    assert actual.dtype == np.dtype(np.float32)


def test_public_width_objective_keeps_captured_source_loss_and_complete_gradient() -> (
    None
):
    features = np.zeros((2, 4096), dtype=np.float32)
    targets = np.array([0, 17], dtype=np.int64)
    weights = np.zeros((18, 4097), dtype=np.float32, order="F")
    weights[:, :-1] = (
        ((np.arange(73_728, dtype=np.float32) % 509 - 254) / 257)
        .reshape((18, 4096), order="F")
        .astype(np.float32)
    )
    objective = logistic._Objective(features=features, targets=targets, penalty=0.125)
    loss, gradient = objective(weights.ravel(order="F"))
    # The unchanged source under its declared thread contract supplies this scalar.
    assert float(loss).hex() == "0x1.78d239da00000p+10"
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
    with pytest.raises(ValueError, match="positive inverse regularization"):
        config.make()


@pytest.mark.parametrize(("updates", "line_search"), argvalues=[(-1, 50), (5000, 0)])
def test_solver_rejects_invalid_update_and_line_search_counts(
    updates: int,
    line_search: int,
) -> None:
    config = LBFGS.Config()
    config.max_iter = updates
    config.max_line_search = line_search
    with pytest.raises(
        ValueError,
        match="nonnegative updates and a positive line search",
    ):
        config.make()


@pytest.mark.parametrize(
    ("gradient", "relative"),
    argvalues=[(-1, 0), (np.nan, 0), (0, -1), (0, np.nan)],
)
def test_solver_rejects_invalid_stopping_tolerances(
    gradient: float,
    relative: float,
) -> None:
    config = LBFGS.Config()
    config.tolerance_gradient = gradient
    config.tolerance_relative = relative
    with pytest.raises(ValueError, match="nonnegative stopping tolerances"):
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


def test_solver_accepts_a_valid_wolfe_search_warning_step() -> None:
    result, updates = (
        LBFGS.Config(max_iter=1, max_line_search=50)
        .make()
        .minimize(_linear, initial=np.zeros(2))
    )
    np.testing.assert_array_equal(result, [-1e10, -1e10])
    assert updates == 1


def test_solver_restores_a_point_with_no_representable_descent() -> None:
    initial = np.full(2, 1e20)
    result, updates = LBFGS.Config().make().minimize(_linear, initial=initial)
    np.testing.assert_array_equal(result, initial)
    assert updates == 0


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_solver_checks_update_budgets_after_accepting_a_step(budget: int) -> None:
    result, updates = (
        LBFGS.Config(max_iter=budget, tolerance_gradient=0, tolerance_relative=0)
        .make()
        .minimize(_quadratic, initial=np.zeros(2))
    )
    assert updates == max(budget, 1)
    assert _quadratic(result)[0] < _quadratic(np.zeros(2))[0]


def test_solver_restores_the_initial_point_when_search_fails() -> None:
    initial = np.array([2.0, -3.0])
    result, updates = (
        LBFGS.Config(max_line_search=1)
        .make()
        .minimize(_inconsistent_objective, initial=initial)
    )
    np.testing.assert_array_equal(result, initial)
    assert updates == 0


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
