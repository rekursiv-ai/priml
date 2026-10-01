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

from pathlib import Path
from typing import Final, cast

import numpy as np
import pytest
import torch

from priml.baselines.convextok.pdlp import Pdlp
from priml.baselines.convextok.program import LinearProgram


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


def _assert_near(ours: torch.Tensor, golden: np.ndarray) -> None:
    theirs = torch.from_numpy(golden)
    scale = float(theirs.abs().max().clamp(min=1.0))
    assert float((ours - theirs).abs().max()) <= _ITERATE_TOLERANCE * scale


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
