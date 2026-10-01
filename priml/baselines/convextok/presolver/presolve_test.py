"""The presolver must reduce a ConvexTok program exactly as PSLP 0.0.8 does.

cuOpt runs PDLP on the problem PSLP returns, so the reduced program, its objective
offset and the postsolved primal point are compared bit for bit with PSLP's own,
recorded while the C library was vendored: the fixture program, small ConvexTok
programs, and general random LPs (infinite bounds, inequalities, parallel rows and
columns). The fuzz programs were chosen so that, with the fixture, they execute
every reachable line of the port; those PSLP proved infeasible or unbounded must
raise likewise.

The kernels run as the Python Numba compiles (``conftest.py``'s ``kernels``), so
each test takes milliseconds rather than paying a JIT compile. Only what needs
the compiled code asks for it: that every kernel compiles and still matches PSLP,
and that a kernel keeps a single specialization.
"""

from dataclasses import replace
from pathlib import Path
from typing import Final, cast

import json

import numpy as np
import pytest
import torch

from priml.baselines.convextok.presolver import core
from priml.baselines.convextok.presolver.numba_api import Dispatcher
from priml.baselines.convextok.presolver.presolve import (
    Presolved,
    presolve,
)
from priml.baselines.convextok.program import LinearProgram, build_program
from priml.lib.custom_json import DictCodec, IntCodec, ListCodec


_CWD: Final = Path(__file__).resolve().parent
_TESTDATA: Final = _CWD.parent / "testdata"
_CPU: Final = torch.device("cpu")
_ERRORS: Final = {"2": "infeasible", "6": "unbounded"}
_FUZZ: Final = sorted((_TESTDATA / "presolve_fuzz").glob("*.npz"))


def test_presolve_matches_pslp_on_fixture() -> None:
    with _npz(_TESTDATA / "presolve.npz") as golden:
        _assert_reduced_like(presolve(fixture_program(), _CPU), golden)


def test_postsolve_matches_pslp_on_fixture() -> None:
    presolved = presolve(fixture_program(), _CPU)
    with _npz(_TESTDATA / "presolve.npz") as golden:
        primal = presolved.postsolve(torch.from_numpy(golden["point_x"]))
        assert torch.equal(primal, torch.from_numpy(golden["post_x"]))


@pytest.mark.parametrize("path", [pytest.param(path, id=path.stem) for path in _FUZZ])
def test_presolve_and_postsolve_match_pslp_on_fuzz_program(path: Path) -> None:
    _assert_matches_pslp(path)


# Compiled: the first presolve in a process JIT-compiles every kernel it reaches,
# about 45 s on x86 with a cold Numba cache, so this is one test, not one per program.
@pytest.mark.compute_large_fixture
@pytest.mark.parametrize("kernels", ["compiled"], indirect=True)
def test_compiled_kernels_match_pslp_on_every_program() -> None:
    """Every kernel compiles, and the compiled presolve still matches PSLP.

    The fuzz programs with the fixture execute every reachable kernel line, so
    together they compile every kernel the presolve can call.
    """
    with _npz(_TESTDATA / "presolve.npz") as golden:
        _assert_reduced_like(presolve(fixture_program(), _CPU), golden)
    for path in _FUZZ:
        _assert_matches_pslp(path)


@pytest.mark.compute_large_fixture
@pytest.mark.parametrize("kernels", ["compiled"], indirect=True)
def test_index_dtypes_share_one_kernel_specialization() -> None:
    # A second specialization of a cached kernel can segfault a later process.
    program = fixture_program()
    wide = replace(
        program,
        crow_indices=program.crow_indices.long(),
        col_indices=program.col_indices.long(),
    )
    narrow_result, wide_result = presolve(program, _CPU), presolve(wide, _CPU)
    assert torch.equal(narrow_result.program.values, wide_result.program.values)
    kernel = core.new_core
    assert isinstance(kernel, Dispatcher)
    assert len(kernel.signatures) == 1


def test_snapshot_names_the_reduced_program() -> None:
    presolved = presolve(fixture_program(), _CPU)
    values = core.snapshot(presolved.state)
    fields = dict(zip(core.SNAPSHOT_FIELDS, values, strict=True))
    reduced = presolved.program
    assert (fields["m"], fields["n"]) == (reduced.num_rows, reduced.num_columns)
    # Compaction leaves the arrays their full size; the live entries lead.
    a_x = cast(np.ndarray, fields["A_x"])[: len(reduced.values)]
    c = cast(np.ndarray, fields["c"])[: reduced.num_columns]
    assert np.array_equal(a_x, reduced.values.numpy())
    assert np.array_equal(c, reduced.objective.numpy())


def test_postsolve_rejects_a_point_of_the_wrong_size() -> None:
    presolved = presolve(fixture_program(), _CPU)
    wrong = torch.zeros(presolved.program.num_columns + 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="reduced columns"):
        presolved.postsolve(wrong)


def test_infeasible_program_raises() -> None:
    # One variable in [0, 1] that the only row pins to 2.
    one = torch.ones(1, dtype=torch.float64)
    program = LinearProgram(
        crow_indices=torch.tensor([0, 1], dtype=torch.int64),
        col_indices=torch.tensor([0], dtype=torch.int32),
        values=one,
        num_columns=1,
        row_lower=2 * one,
        row_upper=2 * one,
        objective=one,
        lower=0 * one,
        upper=one,
    )
    with pytest.raises(ValueError, match="infeasible"):
        presolve(program, _CPU)


def test_program_beyond_int32_positions_raises() -> None:
    # Four reserved slots per column overflow an int32 from 2**29 columns on;
    # expanded tensors describe that many without allocating them.
    columns = 2**29
    zeros = torch.zeros(1, dtype=torch.float64).expand(columns)
    empty = torch.zeros(0, dtype=torch.float64)
    program = LinearProgram(
        crow_indices=torch.zeros(1, dtype=torch.int64),
        col_indices=torch.zeros(0, dtype=torch.int32),
        values=empty,
        num_columns=columns,
        row_lower=empty,
        row_upper=empty,
        objective=zeros,
        lower=zeros,
        upper=zeros,
    )
    with pytest.raises(ValueError, match="32-bit"):
        presolve(program, _CPU)


def fixture_program() -> LinearProgram:
    """Build the eight-document fixture's program from upstream's pretokens."""
    pretokens = _read_json("pretokens.json")
    return build_program(
        dict(
            zip(
                ListCodec.coerce(pretokens.get("pretokens"), str, default=None),
                ListCodec.coerce(pretokens.get("frequencies"), int, default=None),
                strict=True,
            ),
        ),
        ListCodec.coerce(
            _read_json("candidates.json").get("tokens"),
            str,
            default=None,
        ),
        budget=IntCodec.coerce(_read_json("corpus.json").get("budget"), default=None),
    ).program


def _assert_matches_pslp(path: Path) -> None:
    """Presolve a fuzz program and postsolve its point as PSLP did, or raise as it did."""
    with _npz(path) as golden:
        if "error" in golden.files:
            # PSLP's status 2 is INFEASIBLE; 6 (UNBNDORINFEAS) comes from its
            # unboundedness exits.
            status = str(golden["error"]).removesuffix(").").rsplit("status ", 1)[-1]
            with pytest.raises(ValueError, match=_ERRORS[status]):
                presolve(_input_program(golden), _CPU)
            return
        presolved = presolve(_input_program(golden), _CPU)
        _assert_reduced_like(presolved, golden)
        primal = presolved.postsolve(torch.from_numpy(golden["point_x"]))
        assert torch.equal(primal, torch.from_numpy(golden["post_x"]))


def _assert_reduced_like(presolved: Presolved, golden: np.lib.npyio.NpzFile) -> None:
    """Assert the reduced program equals PSLP's, array for array and bit for bit."""
    reduced = presolved.program
    for got, key in (
        (reduced.crow_indices, "Ap"),
        (reduced.col_indices, "Ai"),
        (reduced.values, "Ax"),
        (reduced.row_lower, "lhs"),
        (reduced.row_upper, "rhs"),
        (reduced.objective, "c"),
        (reduced.lower, "lbs"),
        (reduced.upper, "ubs"),
    ):
        assert np.array_equal(got.numpy(), golden[key]), key
    assert reduced.num_columns == len(golden["c"])
    assert reduced.objective_offset == float(golden["offset"])


def _input_program(golden: np.lib.npyio.NpzFile) -> LinearProgram:
    def tensor(key: str) -> torch.Tensor:
        return torch.from_numpy(golden[f"in_{key}"])

    return LinearProgram(
        crow_indices=tensor("crow_indices"),
        col_indices=tensor("col_indices"),
        values=tensor("values"),
        num_columns=int(golden["in_num_columns"]),
        row_lower=tensor("row_lower"),
        row_upper=tensor("row_upper"),
        objective=tensor("objective"),
        lower=tensor("lower"),
        upper=tensor("upper"),
    )


def _npz(path: Path) -> np.lib.npyio.NpzFile:
    loaded = cast(object, np.load(path))
    assert isinstance(loaded, np.lib.npyio.NpzFile)
    return loaded


def _read_json(name: str) -> dict[str, object]:
    raw = cast(object, json.loads((_TESTDATA / name).read_text()))
    return dict(DictCodec.coerce(raw, default=None))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
