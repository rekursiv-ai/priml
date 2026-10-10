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

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Final, cast

import json

from numpy.typing import NDArray

import numpy as np
import pytest
import torch

from priml.baselines.convextok.presolver import bulk, core
from priml.baselines.convextok.presolver.custom_typings import IBuf
from priml.baselines.convextok.presolver.numba_api import (
    Dispatcher,
    get_num_threads,
    set_num_threads,
)
from priml.baselines.convextok.presolver.presolve import (
    Presolved,
    _fast,
    _kernel_array,
    _positions_overflow,
    _trivial,
    new_state,
    presolve,
)
from priml.baselines.convextok.program import LinearProgram, build_program
from priml.lib.codec import from_plain


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


# Compiled: the first presolve in a process JIT-compiles every kernel it reaches,
# about 45 s on x86 with a cold Numba cache, so this is one test, not one per program.
@pytest.mark.compute_large_fixture
@pytest.mark.parametrize("kernels", ["compiled"], indirect=True)
@pytest.mark.usefixtures("two_numba_threads")
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
@pytest.mark.usefixtures("two_numba_threads")
def test_index_dtypes_share_one_kernel_specialization() -> None:
    # A second specialization of a cached kernel can segfault a later process.
    # ``new_state`` fixes every kernel argument's dtype, so building the state
    # compiles only what the assertion reads, not the whole presolve.
    program = fixture_program()
    new_state(program, _CPU)
    new_state(_wide_indices(program), _CPU)
    # ``new_core`` is the Python entry that registers the struct proxies first;
    # the compiled kernel behind it is ``_new_core``.
    kernel = core._new_core
    assert isinstance(kernel, Dispatcher)
    assert len(kernel.signatures) == 1


def test_index_dtypes_reduce_to_the_same_program() -> None:
    program = fixture_program()
    narrow_result = presolve(program, _CPU)
    wide_result = presolve(_wide_indices(program), _CPU)
    assert torch.equal(narrow_result.program.values, wide_result.program.values)


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


def test_postsolve_converts_the_primal_to_float64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    presolved = presolve(fixture_program(), _CPU)
    seen_dtypes: list[np.dtype[np.float64]] = []
    postsolve_primal = core.postsolve_primal

    def record_dtype(
        state: core.Core,
        primal: NDArray[np.float64],
        num_columns: int,
    ) -> NDArray[np.float64]:
        seen_dtypes.append(primal.dtype)
        return postsolve_primal(state, primal, num_columns)

    monkeypatch.setattr(core, "postsolve_primal", record_dtype)
    result = presolved.postsolve(
        torch.zeros(presolved.program.num_columns, dtype=torch.float32),
    )
    assert seen_dtypes == [np.dtype(np.float64)]
    assert result.dtype == torch.float64


def test_kernel_arrays_have_requested_dtype_and_contiguous_layout() -> None:
    tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64).T
    result = _kernel_array(tensor, torch.int32)
    assert result.dtype == np.dtype(np.int32)
    assert result.flags.c_contiguous
    assert np.array_equal(result, np.array([[1, 3], [2, 4]], dtype=np.int32))


def test_new_state_converts_all_kernel_arrays_to_fixed_dtypes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = fixture_program()
    mistyped = replace(
        program,
        values=program.values.float(),
        col_indices=program.col_indices.long(),
        crow_indices=program.crow_indices.long(),
        row_lower=program.row_lower.float(),
        row_upper=program.row_upper.float(),
        objective=program.objective.float(),
        lower=program.lower.float(),
        upper=program.upper.float(),
    )
    observed_dtypes: list[tuple[object, ...]] = []
    new_core = core.new_core

    def record_new_core(
        csr: tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]],
        n_cols: int,
        sides: tuple[NDArray[np.float64], NDArray[np.float64]],
        bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
        objective: NDArray[np.float64],
    ) -> core.Core:
        observed_dtypes.append(
            tuple(array.dtype for array in (*csr, *sides, *bounds, objective)),
        )
        return new_core(csr, n_cols, sides, bounds, objective)

    monkeypatch.setattr(core, "new_core", record_new_core)
    state = new_state(mistyped, _CPU)
    assert observed_dtypes == [
        (
            np.dtype(np.float64),
            np.dtype(np.int32),
            np.dtype(np.int32),
            np.dtype(np.float64),
            np.dtype(np.float64),
            np.dtype(np.float64),
            np.dtype(np.float64),
            np.dtype(np.float64),
        ),
    ]
    fields = dict(zip(core.SNAPSHOT_FIELDS, core.snapshot(state), strict=True))
    for name in ("A_x", "lhs", "rhs", "lb", "ub", "c"):
        assert np.asarray(fields[name]).dtype == np.dtype(np.float64), name
    for name in ("A_i", "A_start", "A_end"):
        assert np.asarray(fields[name]).dtype == np.dtype(np.int32), name


def test_position_overflow_check_includes_exact_largest_valid_size() -> None:
    assert not _positions_overflow(1, 536_870_911, 1)
    assert _positions_overflow(1, 536_870_911, 2)


def test_trivial_phase_finishes_after_singleton_rows_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _stub_trivial_core(monkeypatch, events)
    list(_trivial(new_state(fixture_program(), _CPU)))
    assert events == [
        "close_bounds",
        "empty_cols",
        "simple_dual",
        "ston_rows",
        "empty_rows",
        "empty_cols",
    ]


def test_fast_phase_finishes_cleanup_after_doubleton_pass_stalls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _stub_trivial_core(monkeypatch, events)

    def record_ston_cols(state: core.Core) -> None:
        del state
        events.append("ston_cols")

    def has_doubletons(state: core.Core) -> bool:
        del state
        return True

    monkeypatch.setattr(core, "remove_ston_cols", record_ston_cols)
    monkeypatch.setattr(core, "dton_pending", has_doubletons)

    def doubleton_pass(state: core.Core, max_shift: int) -> int:
        del state
        events.append(f"doubleton:{max_shift}")
        return core.UNCHANGED

    monkeypatch.setattr(core, "remove_dton_eq_rows_pass", doubleton_pass)
    list(_fast(new_state(fixture_program(), _CPU)))
    assert events == [
        "ston_cols",
        "close_bounds",
        "empty_cols",
        "simple_dual",
        "ston_rows",
        "empty_rows",
        "empty_cols",
        "doubleton:10",
        "close_bounds",
        "empty_cols",
        "simple_dual",
        "ston_rows",
        "empty_rows",
        "empty_cols",
    ]


def test_presolve_forwards_the_requested_device_to_every_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transpose_devices: list[object] = []
    sort_devices: list[object] = []
    column_flags: list[object] = []
    transpose_slots = bulk.transpose_slots
    sort_rows = bulk.sort_rows
    parallel_hashes = core.parallel_hashes

    def record_transpose_device(
        cols: NDArray[np.int32],
        starts: NDArray[np.int32],
        ends: NDArray[np.int32],
        n_cols: int,
        device: torch.device,
    ) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], int]:
        transpose_devices.append(device)
        return transpose_slots(cols, starts, ends, n_cols, device)

    def record_sort_device(
        active: NDArray[np.int32],
        sparsity: NDArray[np.int32],
        coeff: NDArray[np.int32],
        device: torch.device,
    ) -> NDArray[np.int32]:
        sort_devices.append(device)
        return sort_rows(active, sparsity, coeff, device)

    def record_parallel_hashes(
        state: core.Core,
        columns: bool,
    ) -> tuple[IBuf, IBuf, IBuf]:
        column_flags.append(columns)
        return parallel_hashes(state, columns)

    monkeypatch.setattr(bulk, "transpose_slots", record_transpose_device)
    monkeypatch.setattr(bulk, "sort_rows", record_sort_device)
    monkeypatch.setattr(core, "parallel_hashes", record_parallel_hashes)
    presolve(fixture_program(), _CPU)
    assert transpose_devices == [_CPU]
    assert len(sort_devices) >= 2
    assert all(device == _CPU for device in sort_devices)
    assert set(column_flags) == {False, True}
    assert all(type(columns) is bool for columns in column_flags)


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
                from_plain(pretokens.get("pretokens"), list[str], default=[]),
                from_plain(pretokens.get("frequencies"), list[int], default=[]),
                strict=True,
            ),
        ),
        from_plain(_read_json("candidates.json").get("tokens"), list[str], default=[]),
        budget=from_plain(_read_json("corpus.json").get("budget"), int, default=0),
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


def _wide_indices(program: LinearProgram) -> LinearProgram:
    """Return ``program`` with 64-bit row and column indices."""
    return replace(
        program,
        crow_indices=program.crow_indices.long(),
        col_indices=program.col_indices.long(),
    )


def _npz(path: Path) -> np.lib.npyio.NpzFile:
    loaded = cast(object, np.load(path))
    assert isinstance(loaded, np.lib.npyio.NpzFile)
    return loaded


def _read_json(name: str) -> dict[str, object]:
    raw = cast(object, json.loads((_TESTDATA / name).read_text()))
    return dict(from_plain(raw, dict[str, object]))


def _stub_trivial_core(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> None:
    def record_close_bounds(state: core.Core) -> None:
        del state
        events.append("close_bounds")

    def record_empty_cols(state: core.Core) -> None:
        del state
        events.append("empty_cols")

    def record_simple_dual(state: core.Core) -> None:
        del state
        events.append("simple_dual")

    def stop_ston_rows(state: core.Core) -> int:
        del state
        events.append("ston_rows")
        return core.UNCHANGED

    def record_empty_rows(state: core.Core) -> None:
        del state
        events.append("empty_rows")

    monkeypatch.setattr(core, "remove_variables_with_close_bounds", record_close_bounds)
    monkeypatch.setattr(core, "remove_empty_cols", record_empty_cols)
    monkeypatch.setattr(core, "simple_dual_fix", record_simple_dual)
    monkeypatch.setattr(core, "remove_ston_rows", stop_ston_rows)
    monkeypatch.setattr(core, "remove_empty_rows", record_empty_rows)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
