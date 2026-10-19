"""The kernels must reduce a ConvexTok program exactly as PSLP 0.0.8 does.

cuOpt runs PDLP on the problem PSLP returns, so the reduced program, its objective
offset and the postsolved primal point are compared bit for bit with PSLP's own,
recorded while the C library was vendored: the fixture program, small ConvexTok
programs, and general random LPs (infinite bounds, inequalities, parallel rows and
columns). The fuzz programs were chosen so that, with the fixture, they execute
every reachable line of the port; those PSLP proved infeasible or unbounded must
raise likewise. ``presolve`` only sequences the kernels, so it drives them here.

The kernels run as the Python Numba compiles (``conftest.py``'s ``kernels``), so
each test takes milliseconds rather than paying a JIT compile. Only what needs
the compiled code asks for it: that every kernel compiles and still matches PSLP.

``_sorted_find`` is PSLP's search within a sorted row; its one caller always
searches for an entry the row holds, so its not-found results are tested here.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pytest
import torch

from priml.baselines.convextok.presolver import core
from priml.baselines.convextok.presolver.custom_typings import FBuf, IBuf
from priml.baselines.convextok.presolver.numba_api import (
    get_num_threads,
    set_num_threads,
)
from priml.baselines.convextok.presolver.presolve import new_state, presolve
from priml.baselines.convextok.program import LinearProgram
from priml.baselines.convextok.testing import (
    TESTDATA,
    assert_reduced_like,
    fixture_program,
    input_program,
    npz,
)


_CPU: Final = torch.device("cpu")
_ERRORS: Final = {"2": "infeasible", "6": "unbounded"}
_FUZZ: Final = sorted((TESTDATA / "presolve_fuzz").glob("*.npz"))


def test_presolve_matches_pslp_on_fixture() -> None:
    with npz(TESTDATA / "presolve.npz") as golden:
        assert_reduced_like(presolve(fixture_program(), _CPU), golden)


def test_postsolve_matches_pslp_on_fixture() -> None:
    presolved = presolve(fixture_program(), _CPU)
    with npz(TESTDATA / "presolve.npz") as golden:
        primal = presolved.postsolve(torch.from_numpy(golden["point_x"]))
        assert torch.equal(primal, torch.from_numpy(golden["post_x"]))


@pytest.mark.parametrize("path", [pytest.param(path, id=path.stem) for path in _FUZZ])
def test_presolve_and_postsolve_match_pslp_on_fuzz_program(path: Path) -> None:
    _assert_matches_pslp(path)


@pytest.fixture
def two_numba_threads() -> Iterator[None]:
    """Run the parallel kernels on two threads, leaving torch's thread count alone.

    Numba sizes its pool to every core, and on these few-row programs the
    parallel kernels then spend their time synchronizing threads: the compiled
    sweep below took 2.2-4.4 s at 128 threads and 0.13 s at two (measured on a
    loaded 128-core host). Two still run each ``prange`` concurrently, and the
    kernels' results do not depend on the count -- the goldens check that.
    """
    # Read first: launching Numba's OpenMP pool, which ``get_num_threads`` does,
    # raises the process's OpenMP thread count to every core, and torch reads
    # that count as its own. Unrestored, every later test in the worker ran
    # torch on 128 threads (measured: a 0.6 s CPU rollout test took 42 s).
    torch_threads = torch.get_num_threads()
    numba_threads = get_num_threads()
    set_num_threads(min(2, numba_threads))
    yield
    set_num_threads(numba_threads)
    torch.set_num_threads(torch_threads)


# Compiled, so marked: the first case in a process JIT-compiles every kernel it
# reaches, about 45 s on x86 with a cold Numba cache. Later cases, and every case
# under a warm cache, take milliseconds. The fuzz programs with the fixture execute
# every reachable kernel line, so together these cases compile every kernel the
# presolve can call.
@pytest.mark.compute_large_fixture
@pytest.mark.parametrize("kernels", ["compiled"], indirect=True)
@pytest.mark.usefixtures("two_numba_threads")
def test_compiled_kernels_match_pslp_on_fixture() -> None:
    with npz(TESTDATA / "presolve.npz") as golden:
        assert_reduced_like(presolve(fixture_program(), _CPU), golden)


@pytest.mark.compute_large_fixture
@pytest.mark.parametrize("kernels", ["compiled"], indirect=True)
@pytest.mark.parametrize("path", [pytest.param(path, id=path.stem) for path in _FUZZ])
@pytest.mark.usefixtures("two_numba_threads")
def test_compiled_kernels_match_pslp_on_fuzz_program(path: Path) -> None:
    _assert_matches_pslp(path)


@pytest.mark.parametrize("length", [5, 20], ids=["linear", "binary"])
@pytest.mark.parametrize("target", [4, 9, 10, 18, 100])
def test_sorted_find_matches_a_linear_search(length: int, target: int) -> None:
    values = list(range(0, 3 * (length + 2), 3))
    first = 2
    window = values[first : first + length]
    expected = window.index(target) if target in window else -1
    arr = np.array(values, np.int32)
    assert core._sorted_find(arr, first, length, target) == expected


def test_new_core_retains_input_arrays_and_builds_state() -> None:
    coefficients = np.array([2.0, 3.0], np.float64)
    columns = np.array([0, 1], np.int32)
    starts = np.array([0, 2], np.int32)
    state = core.new_core(
        (coefficients, columns, starts),
        2,
        (np.array([-np.inf]), np.array([4.0])),
        (np.array([0.0, -1.0]), np.array([1.0, 2.0])),
        np.array([0.25, -0.5]),
    )
    core.attach_transpose(
        state,
        np.array([0, 1], np.int32),
        np.array([0, 1, 2], np.int32),
        np.array([1, 2, 2], np.int32),
        2,
    )
    fields: dict[str, object] = dict(
        zip(core.SNAPSHOT_FIELDS, core.snapshot(state), strict=True),
    )

    assert (fields["m"], fields["n"]) == (1, 2)
    expected_arrays = (
        ("A_x", coefficients),
        ("A_i", columns),
        ("A_start", starts),
        ("A_end", np.array([2, 2], np.int32)),
        ("lhs", np.array([-np.inf])),
        ("rhs", np.array([4.0])),
        ("lb", np.array([0.0, -1.0])),
        ("ub", np.array([1.0, 2.0])),
        ("c", np.array([0.25, -0.5])),
    )
    for name, expected in expected_arrays:
        actual = fields[name]
        assert isinstance(actual, np.ndarray)
        assert np.array_equal(actual, expected), name


def test_shift_row_steps_past_an_empty_left_neighbour() -> None:
    # Row 1 is empty with no slack to its left; the shift skips over it, then
    # moves row 2 one slot right to open the third slot.
    a = _matrix(
        x=[1.0, 2.0, 0.0, 0.0, 3.0, 0.0],
        i=[0, 1, -1, -1, 2, -1],
        start=[0, 2, 4, 6],
        end=[2, 2, 5, 6],
    )

    assert core._shift_row(a, 1, 3, 10)

    assert np.array_equal(np.asarray(a.start), [0, 2, 5, 6])
    assert np.array_equal(np.asarray(a.end), [2, 2, 6, 6])
    assert (a.x[5], a.i[5]) == (3.0, 2)


def test_shift_row_refuses_when_the_row_has_no_neighbours() -> None:
    # One full row: nothing to either side can give up a slot.
    a = _matrix(x=[1.0, 2.0], i=[0, 1], start=[0, 2], end=[2, 2])

    assert not core._shift_row(a, 0, 1, 10)

    assert np.array_equal(np.asarray(a.start), [0, 2])
    assert np.array_equal(np.asarray(a.end), [2, 2])


def test_parallel_bin_skips_a_row_of_another_length() -> None:
    # Row 1 matches row 0's first entry, so only its length tells them apart.
    mat = _matrix(
        x=[1.0, 2.0, 1.0, 2.0, 4.0],
        i=[0, 1, 0, 0, 1],
        start=[0, 2, 3],
        end=[2, 3, 5],
    )
    members, starts = core._ivec_new(4), core._ivec_new(4)
    core._ivec_append(starts, 0)

    core._parallel_in_bin(mat, np.array([0, 1, 2], np.int32), members, starts)

    assert np.array_equal(np.asarray(members.data[: members.len]), [2, 0])
    assert np.array_equal(np.asarray(starts.data[: starts.len]), [0, 2])


def test_find_substitution_rejects_a_row_on_an_inactive_column() -> None:
    # x0 + 2 x1 == 1; x0 was fixed, so the row's doubleton is stale.
    one = torch.ones(1, dtype=torch.float64)
    state = new_state(
        LinearProgram(
            crow_indices=torch.tensor([0, 2]),
            col_indices=torch.tensor([0, 1], dtype=torch.int32),
            values=torch.tensor([1.0, 2.0], dtype=torch.float64),
            num_columns=2,
            row_lower=one,
            row_upper=one,
            objective=torch.zeros(2, dtype=torch.float64),
            lower=torch.zeros(2, dtype=torch.float64),
            upper=torch.full((2,), 4.0, dtype=torch.float64),
        ),
        _CPU,
    )
    assert core._find_substitution(state, 0, 10) == (1, 0)
    state.col_tags[0] |= core.C_FIXED

    assert core._find_substitution(state, 0, 10) == (-1, -1)
    assert state.int_vec.len == 0


def test_deleting_an_empty_row_only_deactivates_it() -> None:
    # Row 0 holds x0; row 1 holds nothing.
    state = new_state(
        LinearProgram(
            crow_indices=torch.tensor([0, 1, 1]),
            col_indices=torch.tensor([0], dtype=torch.int32),
            values=torch.tensor([1.0], dtype=torch.float64),
            num_columns=1,
            row_lower=torch.zeros(2, dtype=torch.float64),
            row_upper=torch.ones(2, dtype=torch.float64),
            objective=torch.zeros(1, dtype=torch.float64),
            lower=torch.zeros(1, dtype=torch.float64),
            upper=torch.ones(1, dtype=torch.float64),
        ),
        _CPU,
    )
    core._set_row_to_inactive(state, 1)

    core._delete_inactive_rows(state)

    assert np.array_equal(np.asarray(state.row_sizes), [1, core.SIZE_INACTIVE])
    assert (state.A.nnz, state.col_sizes[0]) == (1, 1)
    assert state.rows_to_delete.len == 0


@dataclass(slots=True, kw_only=True)
class _Matrix:
    """A ``core.Matrix`` the interpreted kernels mutate in place."""

    x: FBuf
    i: IBuf
    start: IBuf
    end: IBuf
    m: int
    n: int
    nnz: int
    n_alloc: int


def _matrix(
    *,
    x: list[float],
    i: list[int],
    start: list[int],
    end: list[int],
) -> _Matrix:
    """Build a row-major matrix whose last ``start`` is the allocation's end."""
    return _Matrix(
        x=np.array(x, np.float64),
        i=np.array(i, np.int32),
        start=np.array(start, np.int32),
        end=np.array(end, np.int32),
        m=len(start) - 1,
        n=max(i) + 1,
        nnz=sum(stop - first for first, stop in zip(start, end, strict=False)),
        n_alloc=len(x),
    )


def _assert_matches_pslp(path: Path) -> None:
    """Presolve a fuzz program and postsolve its point as PSLP did, or raise as it did."""
    with npz(path) as golden:
        if "error" in golden.files:
            # PSLP's status 2 is INFEASIBLE; 6 (UNBNDORINFEAS) comes from its
            # unboundedness exits.
            status = str(golden["error"]).removesuffix(").").rsplit("status ", 1)[-1]
            with pytest.raises(ValueError, match=_ERRORS[status]):
                presolve(input_program(golden), _CPU)
            return
        presolved = presolve(input_program(golden), _CPU)
        assert_reduced_like(presolved, golden)
        primal = presolved.postsolve(torch.from_numpy(golden["point_x"]))
        assert torch.equal(primal, torch.from_numpy(golden["post_x"]))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
