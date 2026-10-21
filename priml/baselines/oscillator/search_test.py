"""Check the bounded searches against reference iterates and closed-form fits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

from torch import Tensor

import pytest
import torch

from priml.baselines.oscillator.search import (
    _GOLDEN,
    BoundedLeastSquares,
    BoundedScalarSearch,
    _Bracket,
)


if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("fun", "bounds", "xatol", "expected"),
    [
        # Recorded from the reference fminbound implementation of Brent's
        # method with the same bounds, xatol, and evaluation budget.
        (
            "quartic",
            (0.0, 1.0),
            1e-6,
            [
                0.3819660112501051,
                0.6180339887498948,
                0.7639320225002102,
                0.6118033988749895,
                0.6280723896948777,
                0.630108372093311,
                0.6299714576226553,
                0.6299603669895831,
                0.6299607096667387,
                0.6299600243124276,
            ],
        ),
        (
            "kink",
            (0.0, 1.0),
            1e-6,
            [
                0.3819660112501051,
                0.6180339887498948,
                0.7639320225002102,
                0.7179193142241742,
                0.6964771452336834,
                0.6665145256417029,
                0.696873518840036,
                0.7029711575288475,
                0.708680845317303,
                0.7000072102890345,
                0.6999612586440378,
                0.7005966634898977,
                0.7002323613769867,
                0.7000393961807081,
                0.7000001332382565,
                0.6999921443434729,
                0.6999997895222433,
                0.7000006421841584,
            ],
        ),
        (
            "plateau",
            (0.0, 1.0),
            1e-6,
            [
                0.3819660112501051,
                0.6180339887498948,
                0.7639320225002102,
                0.6017819747993336,
                0.6099079817746142,
                0.6068041233028758,
                0.6048858332710721,
                0.6037002648311374,
                0.6029675432392685,
                0.6025146963912026,
                0.6022348216473996,
                0.6020618495431367,
                0.6019549469035965,
                0.6018888774388736,
                0.6018480442640565,
                0.6018228079741509,
                0.6018072110892393,
                0.6017975716842453,
                0.6017916142043277,
                0.6017879322792513,
                0.6017856567244101,
                0.6017842503541749,
                0.6017833811695688,
                0.6017828439839397,
                0.6017825017247243,
            ],
        ),
        (
            "cubic",
            (0.5, 3.0),
            1e-6,
            [
                1.4549150281252627,
                2.045084971874737,
                1.0901699437494743,
                1.1754488172263853,
                1.152754179982163,
                1.1545171641957965,
                1.1546942456868226,
                1.154700591862623,
                1.154700241402312,
                1.154700942322934,
            ],
        ),
        (
            "double_well",
            (-4.0, 6.0),
            1e-6,
            [
                -0.18033988749894903,
                2.1803398874989477,
                3.6393202250021024,
                1.3291700046837491,
                1.7287278767613803,
                0.7525885322241477,
                0.8732306874914149,
                1.0326974356909608,
                1.1459398803142078,
                0.9647837947655779,
                0.9614559215211943,
                0.9548827819301415,
                0.9559549172306053,
                0.9559883116157148,
                0.9559830213624483,
                0.9559833688753013,
                0.9559826738495952,
            ],
        ),
        (
            "sextic",
            (-4.0, 6.0),
            1e-6,
            [
                -0.18033988749894903,
                2.1803398874989477,
                -1.639320225002103,
                -0.9324388148069788,
                -0.9712531203806736,
                -1.2145074080017066,
                -1.3767714652392689,
                -1.450379732313319,
                -1.3486300980133916,
                -1.3613603235731686,
                -1.3611343655700618,
                -1.361309645841831,
                -1.361308251604901,
                -1.3613086051296988,
                -1.3613078980801034,
            ],
        ),
        # The parabola's vertex lands exactly on the best point.
        (
            "centered",
            (0.0, 1.0),
            1e-6,
            [
                0.3819660112501051,
                0.6180339887498948,
                0.7639320225002102,
                0.5,
                0.5000003407495318,
                0.4999996592504682,
            ],
        ),
        # Integer levels make later points tie with the runner-up.
        (
            "terraces",
            (-1.0, 2.0),
            0.05,
            [
                0.1458980337503153,
                0.8541019662496844,
                1.2917960675006308,
                0.718847050625473,
                0.4999999999999999,
                0.6770509831248421,
                0.6603843064158865,
                0.5991229526270957,
                0.5612613537887908,
                0.5378615988383049,
                0.5211949241938615,
            ],
        ),
        (
            "stairs",
            (0.0, 1.0),
            0.05,
            [
                0.3819660112501051,
                0.6180339887498948,
                0.2360679774997897,
                0.14589803375031546,
                0.19098300562505258,
                0.17376207875073607,
            ],
        ),
    ],
)
def test_scalar_search_repeats_the_reference_iterates(
    fun: str,
    bounds: tuple[float, float],
    xatol: float,
    expected: list[float],
) -> None:
    objective = {
        "quartic": _quartic,
        "kink": _kink,
        "plateau": _plateau,
        "cubic": _cubic,
        "double_well": _double_well,
        "sextic": _sextic,
        "stairs": _stairs,
        "centered": _centered,
        "terraces": _terraces,
    }[fun]
    recorded = _Recorded(objective)
    cfg = BoundedScalarSearch.Config(bounds=bounds, xatol=xatol)
    result = cfg.make()(recorded)
    assert recorded.points == expected
    # A tie moves the best point to the newer evaluation.
    lowest = min(map(objective, expected))
    assert result.x == [x for x in expected if objective(x) == lowest][-1]
    assert result.success


def test_scalar_search_approaches_an_endpoint_without_evaluating_it() -> None:
    cfg = BoundedScalarSearch.Config()
    recorded = _Recorded(_square)
    result = cfg.make()(recorded)
    assert all(0 < point < 1 for point in recorded.points)
    assert result.x == min(recorded.points)
    assert result.x < cfg.xatol
    assert result.success


@pytest.mark.parametrize("max_evaluations", [2, 5])
def test_scalar_search_reports_an_exhausted_budget(max_evaluations: int) -> None:
    recorded = _Recorded(_quartic)
    cfg = BoundedScalarSearch.Config(max_evaluations=max_evaluations)
    result = cfg.make()(recorded)
    assert len(recorded.points) == max_evaluations
    assert result.x == min(recorded.points, key=_quartic)
    assert result.success is False


def test_scalar_search_fails_when_the_objective_is_nan() -> None:
    result = BoundedScalarSearch.Config().make()(lambda x: x * math.nan)
    assert result.success is False


def test_scalar_search_accepts_a_single_point_interval() -> None:
    recorded = _Recorded(_square)
    result = BoundedScalarSearch.Config(bounds=(0.0, 0.0), xatol=0.0).make()(recorded)
    assert recorded.points == [0.0]
    assert result.success


@pytest.mark.parametrize(
    ("low", "high", "w", "v", "vertex", "previous", "xatol", "expected"),
    [
        # A previous step exactly tol long is too short to trust a parabola.
        (-1.0, 1.0, 0.5, -0.5, 2**-6, 2**-4, 3 * 2**-4, -_GOLDEN),
        # A vertex exactly half the previous step away is too far.
        (-1.0, 1.0, 0.5, -0.5, 0.125, 0.25, 3 * 2**-30, -_GOLDEN),
        # A vertex exactly on an end of the interval is outside it, whichever
        # order w and v come in.
        (-0.125, 1.0, 0.25, 0.5, -0.125, 0.5, 3 * 2**-30, _GOLDEN),
        (-0.125, 1.0, 0.5, 0.25, -0.125, 0.5, 3 * 2**-30, _GOLDEN),
        (-1.0, 0.125, -0.25, -0.5, 0.125, 0.5, 3 * 2**-30, -_GOLDEN),
        (-1.0, 0.125, -0.5, -0.25, 0.125, 0.5, 3 * 2**-30, -_GOLDEN),
        # A vertex exactly 2 * tol from an end is far enough to step to.
        (-0.125 - 2**-9, 1.0, 0.25, 0.5, -0.125, 0.5, 3 * 2**-10, -0.125),
        (-1.0, 0.125 + 2**-9, -0.25, -0.5, 0.125, 0.5, 3 * 2**-10, 0.125),
        # Nearer an end, the step is tol toward the middle, rightward when x
        # is exactly the middle.
        (
            -0.125 - 2**-11,
            0.125 + 2**-11,
            -0.0625,
            -0.125,
            0.125,
            0.5,
            3 * 2**-10,
            2**-10,
        ),
    ],
)
def test_bracket_breaks_exact_ties_as_the_reference_does(
    low: float,
    high: float,
    w: float,
    v: float,
    vertex: float,
    *,
    previous: float,
    xatol: float,
    expected: float,
) -> None:
    # Dyadic points on a parabola make every tie exact; tol is xatol / 3 at 0.
    f = {t: (t - vertex) * (t - vertex) for t in (0.0, w, v)}
    bracket = _Bracket(low=low, high=high, x=0.0, w=w, v=v, fx=f[0.0], fw=f[w], fv=f[v])
    bracket.previous = previous
    assert bracket.propose(xatol) == expected


@pytest.mark.parametrize(
    ("step", "u", "fu", "expected"),
    [
        # A point tied with v's value replaces v.
        (0.75, 0.75, 2.0, (-1.0, 0.75, 0.75, 2.0)),
        # A zero step still moved right, by tol.
        (0.0, 2**-10, 3.0, (-1.0, 2**-10, -0.5, 2.0)),
    ],
)
def test_bracket_update_keeps_the_reference_order(
    step: float,
    u: float,
    fu: float,
    expected: tuple[float, float, float, float],
) -> None:
    bracket = _Bracket(low=-1.0, high=1.0, x=0.0, w=0.5, v=-0.5, fx=0.0, fw=1.0, fv=2.0)
    bracket.step = step
    bracket.update(u, fu=fu)
    assert (bracket.low, bracket.high, bracket.v, bracket.fv) == expected


@pytest.mark.parametrize("bounds", [(1.0, 0.0), (0.0, math.inf), (math.nan, 1.0)])
def test_scalar_search_rejects_invalid_bounds(bounds: tuple[float, float]) -> None:
    with pytest.raises(
        ValueError,
        match=r"^Search bounds must be finite and ordered\.$",
    ):
        BoundedScalarSearch.Config(bounds=bounds).make()


def test_least_squares_recovers_an_exact_nonlinear_fit() -> None:
    recorded = _Recorded(_curve)
    result = BoundedLeastSquares.Config().make()(recorded, _start())
    # The default gtol stops within about 1e-11 of the exact coefficients.
    torch.testing.assert_close(
        result.x,
        torch.tensor([0.4, 1.3], dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )
    assert result.success
    # The damping schedule decides how many residual calls a fit takes.
    assert len(recorded.points) == 18


def test_least_squares_holds_parameters_on_the_bounds_they_press() -> None:
    recorded = _Recorded(_corner)
    result = BoundedLeastSquares.Config().make()(recorded, _start())
    assert torch.equal(result.x, torch.tensor([0.0, 2.0], dtype=torch.float64))
    assert result.success
    # Start, two columns, the trial clipped into the corner, two more columns.
    assert len(recorded.points) == 6
    assert all(bool(((point >= 0) & (point <= 2)).all()) for point in recorded.points)


def test_least_squares_fits_free_parameters_with_one_held_on_a_bound() -> None:
    recorded = _Recorded(_coupled)
    result = BoundedLeastSquares.Config().make()(recorded, _start())
    assert len(recorded.points) == 23
    assert result.x[0] == 0
    # The default ftol stops this nonzero-residual fit within about 1e-8.
    torch.testing.assert_close(
        result.x[1],
        torch.tensor(0.6, dtype=torch.float64),
        atol=1e-7,
        rtol=0,
    )
    assert result.success


@pytest.mark.parametrize(
    ("high", "start", "steps"),
    [
        # Float64's square-root epsilon is 2**-26, scaled by parameters above one.
        (2.0, (0.25, 1.5), (2**-26, 1.5 * 2**-26)),
        (1.0, (0.25, 1 - 2**-26), (2**-26, 2**-26)),
        (2.0, (0.25, 2.0), (2**-26, -(2**-25))),
    ],
)
def test_least_squares_differences_within_the_box(
    high: float,
    start: tuple[float, float],
    steps: tuple[float, float],
) -> None:
    recorded = _Recorded(_curve)
    cfg = BoundedLeastSquares.Config(bounds=(0.0, high), max_evaluations=1)
    cfg.make()(recorded, torch.tensor(start, dtype=torch.float64))
    x0, *columns = recorded.points
    assert torch.equal(
        torch.stack(columns) - x0,
        torch.tensor(steps, dtype=torch.float64).diag(),
    )


def test_least_squares_budget_counts_trials_but_not_jacobian_columns() -> None:
    cfg = BoundedLeastSquares.Config(max_evaluations=2, ftol=0.0, xtol=0.0, gtol=0.0)
    recorded = _Recorded(_curve)
    result = cfg.make()(recorded, _start())
    # Start, two columns, one accepted trial, then two columns at the trial.
    assert len(recorded.points) == 1 + 2 + 1 + 2
    assert torch.equal(result.x, recorded.points[3])
    assert result.success is False


def test_least_squares_without_tolerances_stalls_in_the_corner() -> None:
    cfg = BoundedLeastSquares.Config(ftol=0.0, xtol=0.0, gtol=0.0)
    recorded = _Recorded(_corner)
    result = cfg.make()(recorded, _start())
    assert torch.equal(result.x, torch.tensor([0.0, 2.0], dtype=torch.float64))
    assert result.success is False
    # Every held step after the corner is exactly zero and never accepted, so
    # only the Jacobians at the start and the corner add to the trials.
    assert len(recorded.points) == cfg.max_evaluations + 2 + 2


@pytest.mark.parametrize(
    ("ftol", "xtol", "gtol", "evaluations"),
    [(0.5, 0.0, 0.0, 7), (0.0, 1e-9, 0.0, 16), (0.0, 0.0, 1e-9, 15)],
)
def test_least_squares_stops_on_each_tolerance(
    ftol: float,
    xtol: float,
    gtol: float,
    evaluations: int,
) -> None:
    cfg = BoundedLeastSquares.Config(ftol=ftol, xtol=xtol, gtol=gtol)
    recorded = _Recorded(_line)
    result = cfg.make()(recorded, _start())
    assert result.success
    assert len(recorded.points) == evaluations
    # The least-squares line through (1, 0.5), (2, 0.4), and (3, 0.9).
    torch.testing.assert_close(
        result.x,
        torch.tensor([0.2, 0.2], dtype=torch.float64),
        atol=1e-4 if ftol else 1e-9,
        rtol=0,
    )


def test_least_squares_shortens_steps_that_reach_nonfinite_residuals() -> None:
    recorded = _Recorded(_cubes)
    result = BoundedLeastSquares.Config().make()(recorded, _start())
    assert any(bool((point > 1.5).any()) for point in recorded.points)
    assert len(recorded.points) == 21
    torch.testing.assert_close(
        result.x,
        torch.ones(2, dtype=torch.float64),
        atol=1e-9,
        rtol=0,
    )
    assert result.success


def test_least_squares_damping_settles_a_curved_valley() -> None:
    recorded = _Recorded(_valley)
    result = BoundedLeastSquares.Config().make()(recorded, _start())
    assert result.success
    # Rejected steps and poorly predicted ones exercise every damping update.
    assert len(recorded.points) == 69
    x = result.x.clone().requires_grad_()
    (gradient,) = torch.autograd.grad(_valley(x).square().sum() / 2, x)
    assert float(gradient.abs().max()) < 1e-6


def test_least_squares_leaves_a_bound_the_gradient_points_away_from() -> None:
    recorded = _Recorded(_shallow)
    start = torch.tensor([0.1, 2.0], dtype=torch.float64)
    result = BoundedLeastSquares.Config().make()(recorded, start)
    torch.testing.assert_close(
        result.x,
        torch.tensor([0.5, 1.5], dtype=torch.float64),
        atol=1e-9,
        rtol=0,
    )
    assert result.success
    assert len(recorded.points) == 12


def test_least_squares_keeps_its_parameters_when_a_short_step_fails() -> None:
    # The first trial is clipped into the undefined region and meets this xtol.
    result = BoundedLeastSquares.Config(xtol=10.0).make()(_cubes, _start())
    assert torch.equal(result.x, _start())
    assert result.success


def test_least_squares_fails_from_a_nonfinite_start() -> None:
    cfg = BoundedLeastSquares.Config()
    recorded = _Recorded(_overflow)
    result = cfg.make()(recorded, _start())
    assert torch.equal(result.x, _start())
    assert result.success is False
    assert len(recorded.points) == cfg.max_evaluations + 2


@pytest.mark.parametrize("fun", ["half_gain", "quarter_gain"])
def test_least_squares_ftol_needs_a_smaller_and_well_predicted_gain(fun: str) -> None:
    cfg = BoundedLeastSquares.Config(max_evaluations=2, ftol=0.5)
    recorded = _Recorded({"half_gain": _half_gain, "quarter_gain": _quarter_gain}[fun])
    cfg.make()(recorded, torch.tensor([0.25, 0.5], dtype=torch.float64))
    # The trial lands on the corner, misses ftol, and is differenced there.
    assert len(recorded.points) == 1 + 2 + 1 + 2


@pytest.mark.parametrize(
    ("fun", "start", "first_trial", "expected"),
    [
        ("tilted_low", (0.0, 0.5), (1 / 6, 1 / 6), (1 / 6, 1 / 6)),
        ("tilted_high", (2.0, 0.5), (2.0, 1 / 6), (2.0, 0.25)),
    ],
)
def test_least_squares_frees_a_bound_parameter_with_zero_gradient(
    fun: str,
    start: tuple[float, float],
    first_trial: tuple[float, float],
    expected: tuple[float, float],
) -> None:
    recorded = _Recorded({"tilted_low": _tilted_low, "tilted_high": _tilted_high}[fun])
    cfg = BoundedLeastSquares.Config()
    result = cfg.make()(recorded, torch.tensor(start, dtype=torch.float64))
    # The free parameter on the bound joins the coupled Gauss-Newton step,
    # which the damping shortens by about 1e-3.
    torch.testing.assert_close(
        recorded.points[3],
        torch.tensor(first_trial, dtype=torch.float64),
        atol=2e-3,
        rtol=0,
    )
    torch.testing.assert_close(
        result.x,
        torch.tensor(expected, dtype=torch.float64),
        atol=1e-9,
        rtol=0,
    )
    assert result.success


@pytest.mark.parametrize("bounds", [(2.0, 0.0), (1.0, 1.0), (math.nan, 2.0)])
def test_least_squares_rejects_invalid_bounds(bounds: tuple[float, float]) -> None:
    with pytest.raises(
        ValueError,
        match=r"^Least-squares bounds must be ordered numbers\.$",
    ):
        BoundedLeastSquares.Config(bounds=bounds).make()


class _Recorded[X, Y]:
    """Keep each point an objective is evaluated at."""

    def __init__(self, fun: Callable[[X], Y]) -> None:
        self.fun = fun
        self.points: list[X] = []

    def __call__(self, x: X) -> Y:
        self.points.append(x)
        return self.fun(x)


# The objectives multiply rather than call `**`, which rounds through the
# host's libm pow and would make the recorded iterates host-dependent.
def _quartic(x: float) -> float:
    return x * x * x * x - x


def _kink(x: float) -> float:
    return abs(x - 0.7)


def _square(x: float) -> float:
    return x * x


def _plateau(x: float) -> float:
    return max(abs(x - 0.6) - 0.1, 0.0)


def _cubic(x: float) -> float:
    return x * x * x - 4 * x


def _double_well(x: float) -> float:
    return (x - 1) * (x - 1) * (x - 2) * (x - 2) + 0.1 * x


def _sextic(x: float) -> float:
    square = x * x
    return square * square * square - 3 * square * square + square + 0.5 * x


def _stairs(x: float) -> float:
    return float(math.floor(8 * x))


def _centered(x: float) -> float:
    return (x - 0.5) * (x - 0.5)


def _terraces(x: float) -> float:
    return float(abs(math.floor(3 * x - 1.5)))


def _start() -> Tensor:
    return torch.tensor([0.1, 0.2], dtype=torch.float64)


def _curve(c: Tensor) -> Tensor:
    """Residuals of a product model whose exact fit is (0.4, 1.3)."""
    t = torch.arange(1, 4, dtype=torch.float64)
    return c[0] * t + c[1] * t**2 + c[0] * c[1] - (0.4 * t + 1.3 * t**2 + 0.4 * 1.3)


def _corner(c: Tensor) -> Tensor:
    """Linear residuals minimized at (-1, 3), outside both bounds."""
    return torch.stack((c[0] + 1, c[1] - 3, (c[0] + c[1]) / 2 - 1))


def _coupled(c: Tensor) -> Tensor:
    """Linear residuals whose box minimum (0, 0.6) holds only the first at zero."""
    return torch.stack((c[0] + 2 * c[1] - 2, 2 * c[0] + c[1] + 1, c[1] - 0.6))


def _line(c: Tensor) -> Tensor:
    """Residuals of a straight line through three points it cannot all meet."""
    t = torch.arange(1, 4, dtype=torch.float64)
    return c[0] * t + c[1] - torch.tensor([0.5, 0.4, 0.9], dtype=torch.float64)


def _valley(c: Tensor) -> Tensor:
    """Residuals along a steep cubic valley that no point fits exactly."""
    return torch.stack((20 * (c[1] - c[0] ** 3), 1.2 - c[0], 0.3 * (c[1] - 1.5)))


def _shallow(c: Tensor) -> Tensor:
    """Small residuals zero at (0.5, 1.5), with gradients below one near it."""
    return 0.1 * torch.stack((c[0] - 0.5, c[1] - 1.5, c[0] + c[1] - 2))


def _half_gain(c: Tensor) -> Tensor:
    """Corner residuals, but the corner itself has half the cost at (0.25, 0.5)."""
    return _corner_or(c, (32.0, 5.0, 1.0, 0.0))


def _quarter_gain(c: Tensor) -> Tensor:
    """Corner residuals, but the corner gains a quarter of the predicted reduction."""
    return _corner_or(c, (41.0, 3.0, 3.0, 2.0))


def _corner_or(c: Tensor, sixteenths: tuple[float, float, float, float]) -> Tensor:
    """Linear residuals, with chosen dyadic residuals at the corner (0, 2)."""
    if torch.equal(c, torch.tensor([0.0, 2.0], dtype=torch.float64)):
        return torch.tensor(sixteenths, dtype=torch.float64) / 16
    return torch.stack((c[0] + 1, c[1] - 3, (c[0] + c[1]) / 2 - 1, 0 * c[0]))


def _tilted_low(c: Tensor) -> Tensor:
    """Coupled residuals with zero first gradient component at (0, 0.5)."""
    return _tilted(c) - torch.tensor([0.25, 0.25, 0.25], dtype=torch.float64)


def _tilted_high(c: Tensor) -> Tensor:
    """Coupled residuals with zero first gradient component at (2, 0.5)."""
    return _tilted(c) - torch.tensor([2.25, 2.25, 0.25], dtype=torch.float64)


def _tilted(c: Tensor) -> Tensor:
    return torch.stack((c[0] + c[1], c[0], c[1]))


def _overflow(c: Tensor) -> Tensor:
    return torch.cat((c, c)) / torch.zeros((), dtype=c.dtype)


def _cubes(c: Tensor) -> Tensor:
    """Residuals zero at (1, 1) and undefined beyond 1.5."""
    residuals = torch.stack((c[0] ** 3 - 1, c[1] ** 3 - 1, c[0] - c[1]))
    return residuals.where((c <= 1.5).all(), math.nan)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
