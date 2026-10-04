"""PDLP must follow cuOpt's Stable3 trajectory and stop where cuOpt stops.

cuOpt's iteration-limited solves expose its state at each termination check: the
returned (unscaled) iterate, the scaled current iterate, the step size and the primal
weight. The port repeats cuOpt's operations in a different floating-point order, so
iterates are compared at that noise floor, while the stop iteration and the primal
weight's first restart update must agree.

The LP (``pdlp_program.npz``) is the fixture's first 20 pretokens with the candidates
that occur in them: a third of the fixture's columns, so the goldens stay small, while
both solves still restart at the 200 check and run past 210.
"""

from dataclasses import replace
from pathlib import Path
from typing import Final, cast, override

import math

import numpy as np
import pytest
import torch

from priml.baselines.convextok import pdlp
from priml.baselines.convextok.pdlp import (
    Pdlp,
    _Check,
    _check_interval,
    _dual_reflection,
    _dual_update,
    _halpern_update,
    _primal_reflection,
    _primal_update,
    _PrimalWeight,
    _Termination,
    original_to_scaled,
    scaled_to_original,
)
from priml.baselines.convextok.program import LinearProgram
from priml.baselines.convextok.scaling import ScaledProgram, scale_program


_CWD: Final = Path(__file__).resolve().parent
# Measured on this LP: iterates within 7e-15 of cuOpt's (relative to their largest
# entry) at every check; the bound leaves room for other devices' reduction orders.
_ITERATE_TOLERANCE: Final = 1e-12


@pytest.mark.parametrize("presolve", [0, 2])
@pytest.mark.parametrize("check", [10, 20, 200, 210])
def test_iterates_match_cuopt_at_checks(presolve: int, check: int) -> None:
    config = Pdlp.Config()
    config.iteration_limit = check
    result = config.make()(_program(presolve))
    with _npz("cuopt_warmstart.npz") as golden:
        tag = f"presolve{presolve}_{check}"
        assert result.iterations == check
        _assert_near(result.current_primal, golden[f"{tag}_current_primal_solution"])
        _assert_near(result.current_dual, golden[f"{tag}_current_dual_solution"])
        if presolve == 0:
            _assert_near(result.primal, golden[f"{tag}_x"])
            _assert_near(result.dual, golden[f"{tag}_y"])


@pytest.mark.parametrize(
    ("presolve", "weight"),
    [(0, 0.6133764514088671), (2, 0.4982866580710819)],
)
def test_first_restart_updates_the_primal_weight_as_cuopt(
    presolve: int,
    weight: float,
) -> None:
    config = Pdlp.Config()
    config.iteration_limit = 20
    assert config.make()(_program(presolve)).primal_weight == 1.0
    config.iteration_limit = 210
    assert config.make()(_program(presolve)).primal_weight == pytest.approx(
        weight,
        rel=_ITERATE_TOLERANCE,
    )


@pytest.mark.parametrize(("presolve", "iterations"), [(0, 250), (2, 230)])
def test_stops_at_cuopt_iteration(presolve: int, iterations: int) -> None:
    result = Pdlp.Config().make()(_program(presolve))
    assert result.optimal
    assert result.iterations == iterations
    with _npz("cuopt_warmstart.npz") as golden:
        _assert_near(
            result.current_primal,
            golden[f"presolve{presolve}_final_current_primal_solution"],
        )
        if presolve == 0:
            _assert_near(result.primal, golden["presolve0_final_x"])


def test_iteration_limit_stops_unconverged() -> None:
    config = Pdlp.Config()
    config.iteration_limit = 10
    result = config.make()(_program(0))
    assert not result.optimal


def test_restart_state_is_used_before_a_negative_period_restarts() -> None:
    config = Pdlp.Config()
    config.restart_period = -1
    config.iteration_limit = 2

    result = config.make()(_small_program())

    assert result.iterations == 2
    assert not result.optimal


def test_restarts_follow_configured_period() -> None:
    config = Pdlp.Config()
    config.tolerance = 1e-12
    config.iteration_limit = 20
    config.restart_period = 4
    config.artificial_restart = 1.0
    config.sufficient_reduction = 100.0
    result = config.make()(_small_program())

    assert result.iterations == 20
    assert not result.optimal
    assert result.primal_weight == pytest.approx(0.016997546553600227, rel=1e-12)
    torch.testing.assert_close(
        result.primal,
        torch.tensor([2.0, -0.5443797027864222, 0.0], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        result.reduced_cost,
        torch.tensor([-1.250981871785863, 0.0, 2.0], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        result.dual,
        torch.tensor([0.2511950431560914, 0.0], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.step_size == pytest.approx(0.9980000073060245, rel=1e-12)
    torch.testing.assert_close(
        result.current_primal,
        torch.tensor(
            [0.5419548105551106, -0.3651401768854216, 0.0],
            dtype=torch.float64,
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_short_restart_period_updates_controller() -> None:
    config = Pdlp.Config()
    config.tolerance = 1e-12
    config.iteration_limit = 12
    config.restart_period = 2
    config.artificial_restart = 1.0
    config.sufficient_reduction = 100.0

    result = config.make()(_small_program())

    assert result.iterations == 12
    assert result.primal_weight == pytest.approx(0.36305443549956556, rel=1e-12)
    torch.testing.assert_close(
        result.current_primal,
        torch.tensor(
            [0.5419548105551106, -0.1590093757758583, 0.0],
            dtype=torch.float64,
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_artificial_restart_accepts_exact_threshold() -> None:
    config = Pdlp.Config()
    config.tolerance = 1e-12
    config.iteration_limit = 20
    config.restart_period = 4
    config.sufficient_reduction = 0.0
    config.necessary_reduction = 0.0
    config.artificial_restart = 0.5

    result = config.make()(_small_program())

    assert result.iterations == 20
    assert result.primal_weight == pytest.approx(0.5297168317386274, rel=1e-12)
    torch.testing.assert_close(
        result.current_primal,
        torch.tensor(
            [0.5419548105551106, -0.35458037803628933, 0.0],
            dtype=torch.float64,
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_necessary_reduction_tracks_rising_error() -> None:
    config = Pdlp.Config()
    config.tolerance = 1e-12
    config.iteration_limit = 20
    config.restart_period = 4
    config.sufficient_reduction = 0.0
    config.necessary_reduction = 100.0
    config.artificial_restart = 1.0

    result = config.make()(_small_program())

    assert result.iterations == 20
    assert result.primal_weight == pytest.approx(0.2082765327409026, rel=1e-12)


def test_sufficient_reduction_thresholds_select_restarts() -> None:
    weights: list[float] = []
    for reduction in (
        0.1,
        0.2,
        0.21,
        0.22,
        0.23,
        0.24,
        0.25,
        0.3,
        0.35,
        0.36,
        0.37,
        0.38,
        0.39,
        0.4,
        0.45,
        0.5,
        1.0,
        2.0,
        100.0,
    ):
        config = Pdlp.Config()
        config.tolerance = 1e-12
        config.iteration_limit = 20
        config.restart_period = 4
        config.sufficient_reduction = reduction
        config.necessary_reduction = 0.0
        config.artificial_restart = 100.0
        weights.append(config.make()(_small_program()).primal_weight)

    assert weights == pytest.approx(
        [
            *([0.086306313778881] * 4),
            *([0.05797185827273946] * 5),
            *([0.01699754655360023] * 10),
        ],
        rel=1e-12,
    )


def test_sufficient_reduction_includes_its_exact_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, updates, _ = _solve_with_fixed_point_errors(
        monkeypatch,
        [4.0, 100.0, 4.0, 100.0, 100.0],
        sufficient_reduction=0.2,
        necessary_reduction=0.0,
    )

    assert result.iterations == 10
    assert updates == 2


def test_necessary_reduction_includes_its_exact_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, updates, _ = _solve_with_fixed_point_errors(
        monkeypatch,
        [4.0, 100.0, 1.0, 4.0, 100.0],
        sufficient_reduction=0.0,
        necessary_reduction=0.2,
    )

    assert result.iterations == 10
    assert updates == 2


def test_necessary_reduction_requires_error_to_rise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, updates, _ = _solve_with_fixed_point_errors(
        monkeypatch,
        [4.0, 100.0, 4.0, 4.0, 100.0],
        sufficient_reduction=0.0,
        necessary_reduction=0.2,
    )

    assert result.iterations == 10
    assert updates == 1


def test_compiles_updates_when_the_program_is_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = _small_program()
    scaled = scale_program(program, ruiz_iterations=0)
    cuda_program = replace(scaled.program)
    object.__setattr__(cuda_program, "values", _CudaMarker())
    scaled = replace(scaled, program=cuda_program)
    monkeypatch.setattr(pdlp, "scale_program", _ScaledProgramFactory(scaled))
    monkeypatch.setattr(pdlp, "initial_step_size", _FixedStepSize())
    compile_calls: list[bool] = []
    monkeypatch.setattr(pdlp, "_compiled_updates", _CompiledUpdates(compile_calls))
    config = Pdlp.Config()
    config.iteration_limit = 1
    config.restart_period = 1

    result = config.make()(program)

    assert result.iterations == 1
    assert compile_calls == [True]


def test_store_uses_the_check_interval_for_the_potential_next_iterate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    updates = pdlp._UPDATES._replace(
        primal=_PrimalRecorder(calls),
        primal_reflection=_PrimalReflectionRecorder(calls),
    )
    monkeypatch.setattr(pdlp, "_UPDATES", updates)
    monkeypatch.setattr(pdlp, "_check_interval", _CheckIntervals())
    config = Pdlp.Config()
    config.tolerance = 1e-12
    config.iteration_limit = 9
    config.restart_period = 8

    result = config.make()(_small_program())

    assert result.iterations == 9
    assert calls[-1] == "primal"


def test_solver_honors_ruiz_iteration_config() -> None:
    config = Pdlp.Config()
    config.ruiz_iterations = 0
    config.iteration_limit = 1
    config.restart_period = 1
    result = config.make()(_small_program())

    assert result.iterations == 1
    assert result.step_size == pytest.approx(0.998000001867216, rel=1e-12)


def test_optimality_requires_two_completed_iterations() -> None:
    config = Pdlp.Config()
    config.restart_period = 1
    result = config.make()(_feasible_zero_program())

    assert result.optimal
    assert result.iterations == 2
    torch.testing.assert_close(result.primal, torch.zeros(3, dtype=torch.float64))


def test_check_interval_changes_at_each_decade() -> None:
    assert [_check_interval(n) for n in (0, 999, 1000, 9999, 10_000, 99_999)] == [
        10,
        10,
        100,
        100,
        1000,
        1000,
    ]


def test_iterate_scaling_round_trips() -> None:
    scaled = scale_program(_program(0))
    primal = torch.linspace(0.2, 0.7, _program(0).num_columns, dtype=torch.float64)
    dual = torch.linspace(0.3, -0.4, _program(0).num_rows, dtype=torch.float64)
    reduced_cost = torch.linspace(
        0.6,
        -0.1,
        _program(0).num_columns,
        dtype=torch.float64,
    )

    converted = scaled_to_original(scaled, primal, dual, reduced_cost)
    round_trip = original_to_scaled(scaled, *converted)

    for actual, expected in zip(round_trip, (primal, dual, reduced_cost), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)


def test_termination_requires_all_three_tolerances() -> None:
    check = _Check(
        gap=0.2,
        absolute_objective=1.0,
        primal_residual=0.2,
        dual_residual=0.2,
        bound_norm=1.0,
        objective_norm=1.0,
    )
    assert check.optimal(0.1)
    assert not check.optimal(0.09)


def test_projected_primal_and_dual_updates() -> None:
    x = torch.tensor([2.0, 2.0])
    objective = torch.tensor([1.0, -1.0])
    aty = torch.tensor([0.5, 2.0])
    bounds = (torch.tensor([0.0, -1.0]), torch.tensor([1.0, 3.0]))
    primal, reduced_cost, reflected_x = _primal_update(
        x,
        objective,
        aty,
        bounds,
        0.5,
    )
    torch.testing.assert_close(primal, torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(reduced_cost, torch.tensor([-1.5, -1.0]))
    torch.testing.assert_close(reflected_x, torch.tensor([0.0, 4.0]))
    torch.testing.assert_close(
        _primal_reflection(x, objective, aty, bounds, 0.5),
        reflected_x,
    )

    y = torch.tensor([2.0, -1.0])
    activity = torch.tensor([0.5, 2.0])
    row_lower = torch.tensor([0.0, -2.0])
    row_upper = torch.tensor([1.0, 3.0])
    dual, reflected_y = _dual_update(y, activity, row_lower, row_upper, 0.5)
    torch.testing.assert_close(dual, torch.tensor([1.75, -0.5]))
    torch.testing.assert_close(reflected_y, torch.tensor([1.5, 0.0]))
    torch.testing.assert_close(
        _dual_reflection(y, activity, row_lower, row_upper, 0.5),
        reflected_y,
    )


def test_primal_weight_tracks_pid_state() -> None:
    check = _Check(
        gap=0.0,
        absolute_objective=0.0,
        primal_residual=1.0,
        dual_residual=2.0,
        bound_norm=1.0,
        objective_norm=1.0,
    )
    config = Pdlp.Config()
    config.pid_proportional = 0.4
    config.pid_integral = 0.2
    config.pid_derivative = 0.3
    config.pid_integral_smoothing = 0.5
    weight = _PrimalWeight(
        value=3.0,
        best_gap=float("inf"),
        error_sum=0.5,
        last_error=0.25,
    )
    error = 0.6931471805599453 - 1.0986122886681098
    error_sum = 0.25 + error
    delta = error - 0.25
    expected = 3.0 * math.exp(0.4 * error + 0.2 * error_sum + 0.3 * delta)

    weight.update(4.0, 16.0, check, config)

    assert weight.value == pytest.approx(expected)
    assert weight.best == pytest.approx(expected)
    assert weight.error_sum == pytest.approx(error_sum)
    assert weight.last_error == pytest.approx(error)
    assert weight.best_gap == pytest.approx(0.3010299956639812)


@pytest.mark.parametrize(
    ("primal_distance", "dual_distance", "primal_residual", "dual_residual"),
    [
        (1e-32, 4.0, 1.0, 2.0),
        (4.0, 1e-32, 1.0, 2.0),
        (1e24, 4.0, 1.0, 2.0),
        (4.0, 1e24, 1.0, 2.0),
        (4.0, 4.0, 1.0, 1e-8),
        (4.0, 4.0, 1.0, 1e8),
        (4.0, 4.0, 0.0, 1.0),
    ],
)
def test_primal_weight_resets_out_of_range_movement(
    primal_distance: float,
    dual_distance: float,
    primal_residual: float,
    dual_residual: float,
) -> None:
    check = _Check(
        gap=0.0,
        absolute_objective=0.0,
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        bound_norm=1.0,
        objective_norm=1.0,
    )
    weight = _PrimalWeight(value=7.0, best=3.0, error_sum=9.0, last_error=2.0)

    weight.update(primal_distance, dual_distance, check, Pdlp.Config())

    assert weight.value == 3.0
    assert weight.error_sum == 0.0
    assert weight.last_error == 0.0


def test_primal_weight_keeps_best_on_equal_residual_gap() -> None:
    check = _Check(
        gap=0.0,
        absolute_objective=0.0,
        primal_residual=1.0,
        dual_residual=1.0,
        bound_norm=1.0,
        objective_norm=1.0,
    )
    weight = _PrimalWeight(value=3.0, best=7.0, best_gap=0.0)

    weight.update(4.0, 4.0, check, Pdlp.Config())

    assert weight.best == 7.0
    assert weight.best_gap == 0.0


def test_primal_weight_handles_zero_and_unit_residual_ratios() -> None:
    zero_ratio = _PrimalWeight()
    zero_ratio.update(
        4.0,
        4.0,
        _Check(
            gap=0.0,
            absolute_objective=0.0,
            primal_residual=1.0,
            dual_residual=0.0,
            bound_norm=1.0,
            objective_norm=1.0,
        ),
        Pdlp.Config(),
    )
    assert zero_ratio.best_gap == float("inf")

    unit_ratio = _PrimalWeight()
    unit_ratio.update(
        4.0,
        4.0,
        _Check(
            gap=0.0,
            absolute_objective=0.0,
            primal_residual=1.0,
            dual_residual=1.0,
            bound_norm=1.0,
            objective_norm=1.0,
        ),
        Pdlp.Config(),
    )
    assert unit_ratio.best_gap == 0.0


def test_termination_masks_infinite_row_bounds() -> None:
    base = _small_program()
    program = replace(
        base,
        row_lower=torch.tensor([-float("inf"), 0.0], dtype=torch.float64),
        row_upper=torch.tensor([1.0, float("inf")], dtype=torch.float64),
    )
    termination = _Termination(program, scale_program(program))
    torch.testing.assert_close(
        termination.finite_lower,
        torch.where(torch.isfinite(program.row_lower), program.row_lower, 0.0),
    )
    torch.testing.assert_close(
        termination.finite_upper,
        torch.where(torch.isfinite(program.row_upper), program.row_upper, 0.0),
    )


def test_halpern_update_interpolates_anchor() -> None:
    result = _halpern_update(
        torch.tensor([2.0, -1.0]),
        torch.tensor([-2.0, 3.0]),
        0.75,
    )
    torch.testing.assert_close(result, torch.tensor([1.0, 0.0]))


class _FixedPointMax:
    def __init__(self, squares: list[float]) -> None:
        self.squares = squares
        self.calls = 0

    def __call__(self, zero: float, value: float) -> float:
        del value
        del zero
        index = min(self.calls, len(self.squares) - 1)
        self.calls += 1
        return self.squares[index]


class _RecordingWeight(pdlp._PrimalWeight):
    calls = 0

    @override
    def update(
        self,
        primal_distance: float,
        dual_distance: float,
        check: _Check,
        config: Pdlp.Config,
    ) -> None:
        type(self).calls += 1
        super().update(primal_distance, dual_distance, check, config)


class _CudaMarker:
    is_cuda: bool = True


class _ScaledProgramFactory:
    def __init__(self, scaled: ScaledProgram) -> None:
        self.scaled = scaled

    def __call__(
        self,
        program: LinearProgram,
        *,
        ruiz_iterations: int,
    ) -> ScaledProgram:
        del program, ruiz_iterations
        return self.scaled


class _FixedStepSize:
    def __call__(self, scaled: ScaledProgram) -> float:
        del scaled
        return 1.0


class _CompiledUpdates:
    def __init__(self, calls: list[bool]) -> None:
        self.calls = calls

    def __call__(self) -> pdlp._Updates:
        self.calls.append(True)
        return pdlp._UPDATES


class _CheckIntervals:
    def __call__(self, iteration: int) -> int:
        return {6: 9, 9: 1, 10: 10, 11: 9}.get(iteration, 100)


class _PrimalRecorder:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def __call__(
        self,
        x: torch.Tensor,
        objective: torch.Tensor,
        aty: torch.Tensor,
        bounds: tuple[torch.Tensor, torch.Tensor],
        step: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.calls.append("primal")
        return pdlp._primal_update(x, objective, aty, bounds, step)


class _PrimalReflectionRecorder:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def __call__(
        self,
        x: torch.Tensor,
        objective: torch.Tensor,
        aty: torch.Tensor,
        bounds: tuple[torch.Tensor, torch.Tensor],
        step: float,
    ) -> torch.Tensor:
        self.calls.append("reflection")
        return pdlp._primal_reflection(x, objective, aty, bounds, step)


def _solve_with_fixed_point_errors(
    monkeypatch: pytest.MonkeyPatch,
    squares: list[float],
    *,
    sufficient_reduction: float,
    necessary_reduction: float,
) -> tuple[pdlp.PdlpResult, int, int]:
    fixed_max = _FixedPointMax(squares)
    monkeypatch.setattr(pdlp, "max", fixed_max, raising=False)
    _RecordingWeight.calls = 0
    monkeypatch.setattr(pdlp, "_PrimalWeight", _RecordingWeight)
    config = Pdlp.Config()
    config.tolerance = 1e-12
    config.iteration_limit = 10
    config.restart_period = 3
    config.ruiz_iterations = 0
    config.sufficient_reduction = sufficient_reduction
    config.necessary_reduction = necessary_reduction
    config.artificial_restart = 100.0
    result = config.make()(_small_program())
    return result, _RecordingWeight.calls, fixed_max.calls


def _assert_near(ours: torch.Tensor, golden: np.ndarray) -> None:
    theirs = torch.from_numpy(golden)
    scale = float(theirs.abs().max().clamp(min=1.0))
    assert float((ours - theirs).abs().max()) <= _ITERATE_TOLERANCE * scale


def _feasible_zero_program() -> LinearProgram:
    return LinearProgram(
        crow_indices=torch.tensor([0, 1, 2]),
        col_indices=torch.tensor([0, 1]),
        values=torch.ones(2, dtype=torch.float64),
        num_columns=3,
        row_lower=torch.zeros(2, dtype=torch.float64),
        row_upper=torch.zeros(2, dtype=torch.float64),
        objective=torch.zeros(3, dtype=torch.float64),
        lower=torch.zeros(3, dtype=torch.float64),
        upper=torch.ones(3, dtype=torch.float64),
    )


def _small_program() -> LinearProgram:
    return LinearProgram(
        crow_indices=torch.tensor([0, 2, 4]),
        col_indices=torch.tensor([0, 1, 1, 2]),
        values=torch.tensor([1.0, 2.0, -1.0, 1.0], dtype=torch.float64),
        num_columns=3,
        row_lower=torch.tensor([1.0, 0.0], dtype=torch.float64),
        row_upper=torch.tensor([1.0, 2.0], dtype=torch.float64),
        objective=torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64),
        lower=torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64),
        upper=torch.tensor([2.0, 2.0, 3.0], dtype=torch.float64),
        objective_offset=0.3,
    )


def _program(presolve: int) -> LinearProgram:
    """Load the LP as built (``presolve=0``) or as PSLP reduces it (``presolve=2``)."""
    prefix = "presolved" if presolve else "original"
    with _npz("pdlp_program.npz") as program:

        def tensor(field: str) -> torch.Tensor:
            return torch.from_numpy(program[f"{prefix}_{field}"])

        return LinearProgram(
            crow_indices=tensor("crow_indices"),
            col_indices=tensor("col_indices"),
            values=tensor("values"),
            num_columns=len(tensor("objective")),
            row_lower=tensor("row_lower"),
            row_upper=tensor("row_upper"),
            objective=tensor("objective"),
            lower=tensor("lower"),
            upper=tensor("upper"),
            objective_offset=float(tensor("objective_offset")),
        )


def _npz(name: str) -> np.lib.npyio.NpzFile:
    loaded = cast(object, np.load(_CWD / "testdata" / name))
    assert isinstance(loaded, np.lib.npyio.NpzFile)
    return loaded


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
