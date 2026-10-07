"""The initial step size must be cuOpt's: 0.998 over the largest singular value.

cuOpt estimates the scaled matrix's largest singular value by power iteration on
``A A^T``, started from libstdc++'s normal draws over ``std::mt19937(1)``, and keeps the
step constant afterwards. The start vector is compared exactly with draws minted by
libstdc++; the step size with the value cuOpt reports in its warm-start data.
"""

from pathlib import Path
from typing import Final, cast
from unittest.mock import Mock, call

import math

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.convextok import step_size
from priml.baselines.convextok.program import LinearProgram
from priml.baselines.convextok.scaling import scale_program
from priml.baselines.convextok.step_size import (
    initial_step_size,
    singular_value_probe,
)


_CWD: Final = Path(__file__).resolve().parent


class _FixedRandomState:
    def __init__(self, words: np.ndarray) -> None:
        self.words = words

    def seed(self, value: int) -> None:
        assert value == 1

    def randint(
        self,
        *args: int,
        size: int,
    ) -> np.ndarray:
        assert args[-1] == 1 << 32
        assert size == len(self.words)
        return self.words


class _IterationOps:
    def __init__(self) -> None:
        self.norm_calls = 0
        self.dot_calls = 0

    def vector_norm(self, value: torch.Tensor) -> torch.Tensor:
        del value
        self.norm_calls += 1
        return torch.tensor(1.0 if self.norm_calls % 2 else 0.0)

    def dot(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        del left, right
        self.dot_calls += 1
        return torch.tensor(1.0, dtype=torch.float64)


def test_probe_matches_libstdcxx_normal_draws() -> None:
    golden = torch.from_numpy(np.load(_CWD / "testdata" / "normal_probe.npy"))
    assert torch.equal(singular_value_probe(len(golden)), golden)


def test_probe_prefix_does_not_depend_on_length() -> None:
    assert torch.equal(singular_value_probe(7), singular_value_probe(8)[:7])


def test_probe_accumulates_across_short_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = iter(
        [
            np.array([10.0, 11.0]),
            np.array([20.0, 21.0]),
            np.array([30.0, 31.0, 32.0]),
        ],
    )
    monkeypatch.setattr(step_size, "_polar_pairs", lambda: chunks)

    probe = singular_value_probe(5)

    assert probe.dtype == torch.float64
    assert probe.tolist() == [10.0, 11.0, 20.0, 21.0, 30.0]


def test_polar_pairs_preserve_boundary_rejection_and_clipping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    words = np.zeros(4 * 65_536, dtype=np.uint64)
    words[:16] = [
        0xFFFFFFFF,
        0xFFFFFFFF,
        0,
        0x80000000,
        0,
        0,
        0,
        0x80000000,
        0,
        0x80000000,
        0,
        0x80000000,
        0,
        0xC0000000,
        0,
        0x80000000,
    ]
    monkeypatch.setattr(
        np.random,
        "RandomState",
        lambda: _FixedRandomState(words),
    )

    pairs = next(step_size._polar_pairs())

    assert pairs.dtype == np.float64
    assert len(pairs) == 6
    assert np.isfinite(pairs).all()
    assert pairs[0] == 0.0
    assert pairs[1] > 0.0
    assert pairs[2:4].tolist() == [0.0, 0.0]
    assert pairs[4] == 0.0
    assert pairs[5] == pytest.approx(0.5 * (16 * math.log(2)) ** 0.5, rel=1e-15)


def _diagonal_program() -> LinearProgram:
    return LinearProgram(
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


def test_step_size_of_a_diagonal_matrix() -> None:
    # After scaling, diag(2, 8) is the identity: its largest singular value is 1.
    assert math.isclose(
        initial_step_size(scale_program(_diagonal_program())),
        0.998,
        rel_tol=1e-15,
    )


def test_step_size_of_a_zero_matrix_is_the_unit_operator_step() -> None:
    # A zero operator couples nothing, so any step is stable; the unit
    # operator's keeps the step on the scale every scaled program shares.
    zero = _diagonal_program()
    zero = LinearProgram(
        crow_indices=zero.crow_indices,
        col_indices=zero.col_indices,
        values=torch.zeros(2, dtype=torch.float64),
        num_columns=zero.num_columns,
        row_lower=zero.row_lower,
        row_upper=zero.row_upper,
        objective=zero.objective,
        lower=zero.lower,
        upper=zero.upper,
    )
    assert initial_step_size(scale_program(zero)) == 0.998


def test_initial_step_size_places_probe_and_accumulator_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaled = scale_program(_diagonal_program())
    destinations: list[object] = []
    original_to = Tensor.to
    zeros = Mock(wraps=torch.zeros)

    def record_to(tensor: Tensor, device: torch.device | str | None = None) -> Tensor:
        destinations.append(device)
        return original_to(tensor, device)

    monkeypatch.setattr(Tensor, "to", record_to)
    monkeypatch.setattr(torch, "zeros", zeros)

    initial_step_size(scaled)

    assert destinations == [scaled.program.values.device]
    assert zeros.call_args_list == [
        call((), dtype=torch.float64, device=scaled.program.values.device),
    ]


def test_initial_step_size_observes_iteration_limit_and_strict_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaled = scale_program(_diagonal_program())
    operations = _IterationOps()
    monkeypatch.setattr(torch.linalg, "vector_norm", operations.vector_norm)
    monkeypatch.setattr(torch, "dot", operations.dot)

    assert initial_step_size(scaled, tolerance=0.0) == 0.998
    assert operations.dot_calls == 5000
    assert operations.norm_calls == 10_000


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
