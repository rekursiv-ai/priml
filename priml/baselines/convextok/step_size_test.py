"""The initial step size must be cuOpt's: 0.998 over the largest singular value.

cuOpt estimates the scaled matrix's largest singular value by power iteration on
``A A^T``, started from libstdc++'s normal draws over ``std::mt19937(1)``, and keeps the
step constant afterwards. The start vector is compared exactly with draws minted by
libstdc++; the step size with the value cuOpt reports in its warm-start data.
"""

from pathlib import Path
from typing import Final, cast

import math

import numpy as np
import pytest
import torch

from priml.baselines.convextok.program import LinearProgram
from priml.baselines.convextok.scaling import scale_program
from priml.baselines.convextok.step_size import (
    initial_step_size,
    singular_value_probe,
)


_CWD: Final = Path(__file__).resolve().parent


def test_probe_matches_libstdcxx_normal_draws() -> None:
    golden = torch.from_numpy(np.load(_CWD / "testdata" / "normal_probe.npy"))
    assert torch.equal(singular_value_probe(len(golden)), golden)


def test_probe_prefix_does_not_depend_on_length() -> None:
    assert torch.equal(singular_value_probe(7), singular_value_probe(8)[:7])


def test_step_size_of_a_diagonal_matrix() -> None:
    # After scaling, diag(2, 8) is the identity: its largest singular value is 1.
    program = LinearProgram(
        crow_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
        col_indices=torch.tensor([0, 1], dtype=torch.int32),
        values=torch.tensor([2.0, 8.0], dtype=torch.float64),
        num_columns=2,
        row_lower=torch.zeros(2, dtype=torch.float64),
        row_upper=torch.zeros(2, dtype=torch.float64),
        objective=torch.zeros(2, dtype=torch.float64),
        lower=torch.zeros(2, dtype=torch.float64),
        upper=torch.ones(2, dtype=torch.float64),
    )
    assert math.isclose(initial_step_size(scale_program(program)), 0.998, rel_tol=1e-15)


@pytest.mark.parametrize("presolve", [0, 2])
def test_step_size_matches_cuopt(presolve: int) -> None:
    golden = _warmstart_scalar(f"presolve{presolve}_10_initial_step_size")
    assert initial_step_size(scale_program(_program(presolve))) == pytest.approx(
        golden,
        rel=1e-14,
        abs=0,
    )


def _warmstart_scalar(name: str) -> float:
    with _npz("cuopt_warmstart.npz") as warmstart:
        return float(torch.from_numpy(warmstart[name]))


def _npz(name: str) -> np.lib.npyio.NpzFile:
    loaded = cast(object, np.load(_CWD / "testdata" / name))
    assert isinstance(loaded, np.lib.npyio.NpzFile)
    return loaded


def _program(presolve: int) -> LinearProgram:
    """Load the LP cuOpt solved, as built (0) or as PSLP reduces it (2)."""
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
