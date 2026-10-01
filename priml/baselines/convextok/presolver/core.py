"""PSLP 0.0.8's presolve, as Numba kernels over one mutable state.

A line-for-line port of PSLP's reductions that keeps every floating-point
operation in PSLP's order, so the reduced program and the postsolved primal
match PSLP's bit for bit. The matrix keeps PSLP's storage layout -- rows with
spare capacity, shifted when full -- because whether a doubleton substitution
is attempted depends on that capacity. Where PSLP returns INFEASIBLE or
UNBNDORINFEAS, the kernels raise ValueError, naming which.

The public kernels are the steps of PSLP's main loop, which ``presolve``
sequences; each names the PSLP function it ports. Every kernel lives in this module on purpose: Numba stamps a cached kernel with
its own file only, so a caller cached before a callee in another file changed
would keep running the old callee.
"""

from typing import TYPE_CHECKING, Final, Protocol, overload

from numpy.typing import NDArray

import numpy as np

from priml.baselines.convextok.presolver.custom_typings import (
    FBuf,
    IBuf,
)
from priml.baselines.convextok.presolver.numba_api import (
    StructRef,
    StructRefProxy,
    define_proxy,
    float64,
    int32,
    int64,
    new,
    njit,
    prange,
    register,
    uint8,
)


R_LHS_INF: Final = 1
"""Row tag: no lower side (Tags.h)."""
R_RHS_INF: Final = 2
"""Row tag: no upper side."""
R_EQ: Final = 4
"""Row tag: an equality."""
R_INACTIVE: Final = 8
"""Row tag: removed."""
C_LB_INF: Final = 1
"""Column tag: no lower bound."""
C_UB_INF: Final = 4
"""Column tag: no upper bound."""
C_FIXED: Final = 16
"""Column tag: fixed and removed."""
C_SUBSTITUTED: Final = 32
"""Column tag: substituted out."""
C_INACTIVE: Final = C_FIXED | C_SUBSTITUTED
"""Column tag: removed either way."""
NOT_ADDED: Final = 0
"""Activity status: not queued (Activity.h)."""
ADDED: Final = 1
"""Activity status: queued for checks and propagation."""
PROPAGATED_THIS_ROUND: Final = 2
"""Activity status: propagated in the current round."""
PROPAGATE_NEXT_ROUND: Final = 3
"""Activity status: to propagate again next round."""
FIXED_COL: Final = 0
"""Postsolve record type (Postsolver.h); only the primal types are recorded."""
FIXED_COL_INF: Final = 1
"""Postsolve record: a column fixed to an infinite bound."""
SUB_COL: Final = 2
"""Postsolve record: a column substituted out through an equality."""
PARALLEL_COL: Final = 4
"""Postsolve record: a column merged into a parallel one."""
FEAS_TOL: Final = 1e-6
"""PSLP's feasibility tolerance (Numerics.h)."""
ZERO_TOL: Final = 1e-10
"""Coefficients at or below this magnitude are dropped."""
HUGE_VAL: Final = 1e7
"""Implied bounds at or beyond this magnitude are not applied."""
MAX_RATIO_PIVOT: Final = 1e3
"""Doubleton pivots outside [1 / this, this] are rejected."""
SIZE_INACTIVE: Final = -1
"""Row or column size marking a removed one."""
EXTRA_ROW_SPACE: Final = 4
"""Spare slots per row or column beyond ``EXTRA_MEMORY_RATIO`` (glbopts.h)."""
EXTRA_MEMORY_RATIO: Final = 2.0
"""Slots reserved per entry of a row or column."""
UNCHANGED: Final = 0
"""PresolveStatus: nothing reduced."""
REDUCED: Final = 1
"""PresolveStatus: something reduced."""

type BoundChange = tuple[float, float, bool, bool]
"""One column bound's (old value, new value, old value finite, is the lower)."""


@register
class MatrixType(StructRef):
    """Row-major sparse matrix with per-row ranges into spare-capacity storage."""


@register
class IVecType(StructRef):
    """Growable int32 vector."""


@register
class DVecType(StructRef):
    """Growable float64 vector."""


@register
class CoreType(StructRef):
    """Every array PSLP's presolver mutates."""


class Matrix(Protocol):
    """A ``MatrixType`` as kernels see it; Python sees only a handle."""

    x: FBuf
    i: IBuf
    start: IBuf
    end: IBuf
    m: int
    n: int
    nnz: int
    n_alloc: int


class IVec(Protocol):
    """An ``IVecType`` as kernels see it; Python sees only a handle."""

    data: IBuf
    len: int


class DVec(Protocol):
    """A ``DVecType`` as kernels see it; Python sees only a handle."""

    data: FBuf
    len: int


class Core(Protocol):
    """A ``CoreType`` as kernels see it; Python reads it through ``snapshot``."""

    A: Matrix
    AT: Matrix
    m: int
    n: int
    lhs: FBuf
    rhs: FBuf
    row_tags: IBuf
    col_tags: IBuf
    lb: FBuf
    ub: FBuf
    c: FBuf
    offset: float
    row_sizes: IBuf
    col_sizes: IBuf
    act_min: FBuf
    act_max: FBuf
    act_n_inf_min: IBuf
    act_n_inf_max: IBuf
    act_status: IBuf
    lock_up: IBuf
    lock_down: IBuf
    ston_rows: IVec
    ston_cols: IVec
    dton_rows: IVec
    empty_cols: IVec
    empty_rows: IVec
    updated_activities: IVec
    fixed_cols_to_delete: IVec
    sub_cols_to_delete: IVec
    rows_to_delete: IVec
    int_vec: IVec
    ps_type: IVec
    ps_starts: IVec
    ps_indices: IVec
    ps_vals: DVec
    iwork_n_rows: IBuf
    iwork1: IBuf
    iwork2: IBuf
    map_rows: IBuf
    map_cols: IBuf


_MATRIX_FIELDS: Final = (
    ("x", float64[::1]),
    ("i", int32[::1]),
    ("start", int32[::1]),
    ("end", int32[::1]),
    ("m", int64),
    ("n", int64),
    ("nnz", int64),
    ("n_alloc", int64),
)
MATRIX_T: Final = MatrixType(fields=_MATRIX_FIELDS)
IVEC_T: Final = IVecType(fields=[("data", int32[::1]), ("len", int64)])
DVEC_T: Final = DVecType(fields=[("data", float64[::1]), ("len", int64)])
_VECTORS: Final = (
    "ston_rows",
    "ston_cols",
    "dton_rows",
    "empty_cols",
    "empty_rows",
    "updated_activities",
    "fixed_cols_to_delete",
    "sub_cols_to_delete",
    "rows_to_delete",
    "int_vec",
    "ps_type",
    "ps_starts",
    "ps_indices",
)
_CORE_FIELDS: Final = (
    ("A", MATRIX_T),
    ("AT", MATRIX_T),
    ("m", int64),
    ("n", int64),
    ("lhs", float64[::1]),
    ("rhs", float64[::1]),
    ("row_tags", uint8[::1]),
    ("col_tags", uint8[::1]),
    ("lb", float64[::1]),
    ("ub", float64[::1]),
    ("c", float64[::1]),
    ("offset", float64),
    ("row_sizes", int32[::1]),
    ("col_sizes", int32[::1]),
    ("act_min", float64[::1]),
    ("act_max", float64[::1]),
    ("act_n_inf_min", int32[::1]),
    ("act_n_inf_max", int32[::1]),
    ("act_status", uint8[::1]),
    ("lock_up", int32[::1]),
    ("lock_down", int32[::1]),
    *((name, IVEC_T) for name in _VECTORS),
    ("ps_vals", DVEC_T),
    ("iwork_n_rows", int32[::1]),
    ("iwork1", int32[::1]),
    ("iwork2", int32[::1]),
    ("map_rows", int32[::1]),
    ("map_cols", int32[::1]),
)
CORE_T: Final = CoreType(fields=_CORE_FIELDS)


class _MatrixHandle(StructRefProxy):
    """Python handle on a ``MatrixType``."""


class _IVecHandle(StructRefProxy):
    """Python handle on an ``IVecType``."""


class _DVecHandle(StructRefProxy):
    """Python handle on a ``DVecType``."""


class _CoreHandle(StructRefProxy):
    """Python handle on a ``CoreType``."""


def _define_proxies() -> None:
    """Bind each struct type to its Python handle, as Numba's boxing requires."""
    define_proxy(_MatrixHandle, MatrixType, [name for name, _ in _MATRIX_FIELDS])
    define_proxy(_IVecHandle, IVecType, ["data", "len"])
    define_proxy(_DVecHandle, DVecType, ["data", "len"])
    define_proxy(_CoreHandle, CoreType, [name for name, _ in _CORE_FIELDS])


_define_proxies()


class _NewStruct(Protocol):
    """``structref.new``, typed per struct type."""

    @overload
    def __call__(self, struct_type: MatrixType, /) -> Matrix: ...
    @overload
    def __call__(self, struct_type: IVecType, /) -> IVec: ...
    @overload
    def __call__(self, struct_type: DVecType, /) -> DVec: ...
    @overload
    def __call__(self, struct_type: CoreType, /) -> Core: ...


if TYPE_CHECKING:
    _new: _NewStruct
else:
    # Numba compiles only its own ``new``; ``_NewStruct`` types its result.
    _new = new


@njit(cache=True, error_model="numpy")
def a_arrays(s: Core) -> tuple[FBuf, IBuf, IBuf, IBuf, int]:
    """A's values, columns, row starts and ends (shared, not copied) and n."""
    return s.A.x, s.A.i, s.A.start, s.A.end, s.A.n


# ---------------------------------------------------------------------------
# Initialization (Presolver.c new_presolver, Tags.c, Locks.c, Activity.c,
# State.c, Postsolver.c).
# ---------------------------------------------------------------------------


@njit(cache=True, error_model="numpy")
def new_core(
    csr: tuple[FBuf, IBuf, IBuf],
    n_cols: int,
    sides: tuple[FBuf, FBuf],
    bounds: tuple[FBuf, FBuf],
    c: FBuf,
) -> Core:
    """Build PSLP's presolver state for a CSR program with row bounds.

    Args:
      csr: A as (values, column indices, row pointers).
      n_cols: Columns of A.
      sides: Row (lower, upper) sides.
      bounds: Column (lower, upper) bounds.
      c: Objective coefficients.

    Returns:
      s: The state, lacking A's transpose until ``attach_transpose``.

    """
    ax, ai, ap = csr
    lhs, rhs = sides
    lbs, ubs = bounds
    n_rows = ap.size - 1
    s = _new(CORE_T)
    s.lhs, s.rhs, s.c = lhs.copy(), rhs.copy(), c.copy()
    s.offset = 0.0
    # Every work array is written before it is read, so none needs zeroing.
    widest = max(n_rows, n_cols)
    s.iwork_n_rows = np.empty(n_rows, np.int32)
    s.iwork1 = np.empty(widest, np.int32)
    s.iwork2 = np.empty(widest, np.int32)
    s.map_rows = np.empty(n_rows, np.int32)
    s.map_cols = np.empty(n_cols, np.int32)
    s.int_vec = _ivec_new(25)
    s.A = _matrix_new_no_extra_space(ax, ai, ap, n_rows, n_cols)
    s.m, s.n = n_rows, n_cols
    s.row_tags = np.empty(n_rows, np.uint8)
    _new_row_tags(s.lhs, s.rhs, s.row_tags)
    s.lb, s.ub = lbs.copy(), ubs.copy()
    s.col_tags = np.empty(n_cols, np.uint8)
    _new_col_tags(lbs, ubs, s.col_tags)
    a = s.A
    s.act_min = np.zeros(n_rows, np.float64)
    s.act_max = np.zeros(n_rows, np.float64)
    s.act_n_inf_min = np.zeros(n_rows, np.int32)
    s.act_n_inf_max = np.zeros(n_rows, np.int32)
    s.act_status = np.zeros(n_rows, np.uint8)
    _new_activities(s)
    s.row_sizes = np.subtract(a.end[:n_rows], a.start[:n_rows]).astype(np.int32)
    return s


@njit(cache=True, error_model="numpy")
def attach_transpose(
    s: Core,
    dest: IBuf,
    at_start: IBuf,
    at_end: IBuf,
    n_alloc: int,
) -> None:
    """Build AT from each live entry's slot, then count locks and seed the lists.

    Slots are distinct, so the scatter is parallel. Spare slots stay
    uninitialized, as PSLP's malloc leaves them, and nothing reads them.

    Args:
      s: State from ``new_core``.
      dest: Slot in PSLP's transposed layout of every live entry of A, in row
        order (``bulk.transpose_slots``).
      at_start: First slot of each column, then the allocation's end.
      at_end: One past each column's last entry, then the allocation's end.
      n_alloc: Slots in AT.

    """
    a = s.A
    at = _new(MATRIX_T)
    at.x = np.empty(n_alloc, np.float64)
    at.i = np.empty(n_alloc, np.int32)
    at.start, at.end = at_start, at_end
    at.m, at.n, at.nnz, at.n_alloc = s.n, s.m, a.nnz, n_alloc
    ordinal = np.empty(s.m + 1, np.int64)
    ordinal[0] = 0
    for row in range(s.m):
        ordinal[row + 1] = ordinal[row] + a.end[row] - a.start[row]
    _scatter_transpose(a, ordinal, dest, at)
    s.AT = at
    s.col_sizes = np.subtract(at_end[: s.n], at_start[: s.n]).astype(np.int32)
    s.lock_up = np.empty(s.n, np.int32)
    s.lock_down = np.empty(s.n, np.int32)
    _new_locks(s)
    _new_state_lists(s)


# ---------------------------------------------------------------------------
# Python-side view: structref proxies expose no fields, so one kernel returns
# the whole state in SNAPSHOT_FIELDS order.
# ---------------------------------------------------------------------------

SNAPSHOT_FIELDS: Final = (
    "m",
    "n",
    "offset",
    "A_x",
    "A_i",
    "A_start",
    "A_end",
    "A_meta",
    "AT_x",
    "AT_i",
    "AT_start",
    "AT_end",
    "AT_meta",
    "lhs",
    "rhs",
    "row_tags",
    "col_tags",
    "lb",
    "ub",
    "c",
    "row_sizes",
    "col_sizes",
    "act_min",
    "act_max",
    "act_n_inf_min",
    "act_n_inf_max",
    "act_status",
    "lock_up",
    "lock_down",
    "ston_rows",
    "ston_cols",
    "dton_rows",
    "empty_cols",
    "empty_rows",
    "updated_activities",
    "fixed_cols_to_delete",
    "sub_cols_to_delete",
    "rows_to_delete",
    "ps_type",
    "ps_starts",
    "ps_indices",
    "ps_vals",
    "map_rows",
    "map_cols",
)


@njit(cache=True, error_model="numpy")
def snapshot(s: Core) -> tuple[object, ...]:
    """Copy every field of the state, to compare with PSLP's own state dumps.

    Args:
      s: Presolver state.

    Returns:
      fields: Each field's copy, in ``SNAPSHOT_FIELDS`` order; a vector's live
        entries only, and a matrix's (m, n, nnz, n_alloc) as its ``_meta``.

    """
    a, at = s.A, s.AT
    return (
        s.m,
        s.n,
        s.offset,
        a.x.copy(),
        a.i.copy(),
        a.start.copy(),
        a.end.copy(),
        np.array([a.m, a.n, a.nnz, a.n_alloc]),
        at.x.copy(),
        at.i.copy(),
        at.start.copy(),
        at.end.copy(),
        np.array([at.m, at.n, at.nnz, at.n_alloc]),
        s.lhs.copy(),
        s.rhs.copy(),
        s.row_tags.copy(),
        s.col_tags.copy(),
        s.lb.copy(),
        s.ub.copy(),
        s.c.copy(),
        s.row_sizes.copy(),
        s.col_sizes.copy(),
        s.act_min.copy(),
        s.act_max.copy(),
        s.act_n_inf_min.copy(),
        s.act_n_inf_max.copy(),
        s.act_status.copy(),
        s.lock_up.copy(),
        s.lock_down.copy(),
        s.ston_rows.data[: s.ston_rows.len].copy(),
        s.ston_cols.data[: s.ston_cols.len].copy(),
        s.dton_rows.data[: s.dton_rows.len].copy(),
        s.empty_cols.data[: s.empty_cols.len].copy(),
        s.empty_rows.data[: s.empty_rows.len].copy(),
        s.updated_activities.data[: s.updated_activities.len].copy(),
        s.fixed_cols_to_delete.data[: s.fixed_cols_to_delete.len].copy(),
        s.sub_cols_to_delete.data[: s.sub_cols_to_delete.len].copy(),
        s.rows_to_delete.data[: s.rows_to_delete.len].copy(),
        s.ps_type.data[: s.ps_type.len].copy(),
        s.ps_starts.data[: s.ps_starts.len].copy(),
        s.ps_indices.data[: s.ps_indices.len].copy(),
        s.ps_vals.data[: s.ps_vals.len].copy(),
        s.map_rows.copy(),
        s.map_cols.copy(),
    )


# ---------------------------------------------------------------------------
# Postsolve (Postsolver.c). Only the primal records; primal recovery never
# reads a dual record, so skipping them leaves every recovered x unchanged.
# ---------------------------------------------------------------------------


@njit(cache=True, error_model="numpy")
def postsolve_primal(s: Core, x: FBuf, n_cols_original: int) -> NDArray[np.float64]:
    """Map a reduced primal point back to the original columns (postsolver_run).

    Args:
      s: State after ``problem_clean``, holding the postsolve records.
      x: Value per reduced column.
      n_cols_original: Columns of the original program.

    Returns:
      sol: Value per original column.

    """
    sol = np.empty(n_cols_original, np.float64)
    for col in range(n_cols_original):
        mapped = s.map_cols[col]
        sol[col] = np.inf if mapped == -1 else x[mapped]
    kinds, starts = s.ps_type.data, s.ps_starts.data
    idx, vals = s.ps_indices.data, s.ps_vals.data
    for r in range(s.ps_type.len - 1, -1, -1):
        start, stop = starts[r], starts[r + 1]
        kind = kinds[r]
        if kind == FIXED_COL:
            sol[idx[start]] = vals[start]
        elif kind == SUB_COL:
            _retrieve_sub_col(sol, idx, vals, start, stop - start - 2)
        elif kind == FIXED_COL_INF:
            _retrieve_fix_col_inf(sol, idx, vals, start)
        else:
            _retrieve_parallel_col(sol, idx, vals, start)
    return sol


# Activity updates report which side moved (Activity.c).
NO_RECOMPUTE: Final = 1
MAX_ALTERED: Final = 2
MIN_ALTERED: Final = 4
MAX_ALTERED_RECOMPUTE: Final = 8 | MAX_ALTERED
MIN_ALTERED_RECOMPUTE: Final = 16 | MIN_ALTERED


# ---------------------------------------------------------------------------
# Trivial explorers (SimpleReductions.c).
# ---------------------------------------------------------------------------


@njit(cache=True, error_model="numpy")
def remove_variables_with_close_bounds(s: Core) -> None:
    """Fix every live column whose bounds lie within FEAS_TOL of each other.

    Args:
      s: Presolver state.

    Raises:
      ValueError: If a column's lower bound exceeds its upper by FEAS_TOL.

    """
    for col in range(s.n):
        if s.col_sizes[col] < 0 or s.col_tags[col] & (C_LB_INF | C_UB_INF):
            continue
        if abs(s.lb[col] - s.ub[col]) <= FEAS_TOL:
            _fix_col(s, col, s.lb[col], s.c[col])
        elif s.lb[col] > s.ub[col] + FEAS_TOL:
            raise ValueError("Presolve proved the program infeasible.")
    _delete_fixed_cols_from_problem(s)
    _delete_inactive_cols_from_a_and_at(s)


@njit(cache=True, error_model="numpy")
def remove_ston_rows(s: Core) -> int:
    """Eliminate every queued singleton row: equalities first, then inequalities.

    Args:
      s: Presolver state.

    Returns:
      status: ``REDUCED`` if any row was queued, else ``UNCHANGED``.

    """
    rows = s.ston_rows
    if rows.len == 0:
        return UNCHANGED
    for ii in range(rows.len):
        row = rows.data[ii]
        if s.row_tags[row] & R_EQ and not s.row_tags[row] & R_INACTIVE:
            _remove_ston_row(s, row)
    for ii in range(rows.len):
        row = rows.data[ii]
        if not s.row_tags[row] & (R_INACTIVE | R_EQ):
            _remove_ston_row(s, row)
    s.AT.nnz = s.A.nnz
    rows.len = 0
    _delete_inactive_rows(s)
    _delete_fixed_cols_from_problem(s)
    _delete_inactive_cols_from_a_and_at(s)
    return REDUCED


@njit(cache=True, error_model="numpy")
def remove_empty_rows(s: Core) -> None:
    """Deactivate queued empty rows.

    Args:
      s: Presolver state.

    Raises:
      ValueError: If an empty row's sides exclude zero.

    """
    for ii in range(s.empty_rows.len):
        row = s.empty_rows.data[ii]
        tag = s.row_tags[row]
        if (not tag & R_LHS_INF and s.lhs[row] >= FEAS_TOL) or (
            not tag & R_RHS_INF and s.rhs[row] <= -FEAS_TOL
        ):
            raise ValueError("Presolve proved the program infeasible.")
        s.row_tags[row] |= R_INACTIVE
        s.row_sizes[row] = SIZE_INACTIVE
    s.empty_rows.len = 0


@njit(cache=True, error_model="numpy")
def remove_empty_cols(s: Core) -> None:
    """Fix queued empty columns at their objective-optimal bound.

    Args:
      s: Presolver state.

    Raises:
      ValueError: If the objective drives an empty column to an infinite bound.

    """
    for ii in range(s.empty_cols.len):
        k = s.empty_cols.data[ii]
        tag = s.col_tags[k]
        if s.c[k] == 0:
            if not tag & C_LB_INF and s.lb[k] > 0:
                val = s.lb[k]
            elif not tag & C_UB_INF and s.ub[k] < 0:
                val = s.ub[k]
            else:
                val = 0.0
        elif s.c[k] < 0:
            if tag & C_UB_INF:
                raise ValueError("Presolve proved the program unbounded.")
            val = s.ub[k]
            s.lb[k] = s.ub[k]
        else:
            if tag & C_LB_INF:
                raise ValueError("Presolve proved the program unbounded.")
            val = s.lb[k]
            s.ub[k] = s.lb[k]
        s.offset += s.c[k] * val
        s.col_tags[k] |= C_FIXED
        s.col_sizes[k] = SIZE_INACTIVE
        _save_fixed_col(s, k, val, s.c[k], False)
    s.empty_cols.len = 0


@njit(cache=True, error_model="numpy")
def check_activities(s: Core) -> None:
    """Drop redundant row sides (and fully redundant rows) found by activities.

    Args:
      s: Presolver state.

    Raises:
      ValueError: If a row's activity range misses its sides.

    """
    a = s.A
    for ii in range(s.updated_activities.len):
        row = s.updated_activities.data[ii]
        tag = s.row_tags[row]
        if tag & (R_INACTIVE | R_EQ) or s.row_sizes[row] <= 1:
            continue
        status = _check_activity(s, row)
        combined = status | tag
        if combined & R_LHS_INF and combined & R_RHS_INF:
            _set_row_to_inactive(s, row)
        elif status & R_LHS_INF:
            s.row_tags[row] = R_LHS_INF
            for k in range(a.start[row], a.end[row]):
                if a.x[k] > 0:
                    s.lock_down[a.i[k]] -= 1
                else:
                    s.lock_up[a.i[k]] -= 1
            s.lhs[row] = -np.inf
        elif status & R_RHS_INF:
            s.row_tags[row] = R_RHS_INF
            for k in range(a.start[row], a.end[row]):
                if a.x[k] > 0:
                    s.lock_up[a.i[k]] -= 1
                else:
                    s.lock_down[a.i[k]] -= 1
            s.rhs[row] = np.inf
        elif status & 16:
            raise ValueError("Presolve proved the program infeasible.")
    _delete_inactive_rows(s)


# ---------------------------------------------------------------------------
# Simple dual fix (Simple_dual_fix.c, CoreTransformation.c).
# ---------------------------------------------------------------------------


@njit(cache=True, error_model="numpy")
def simple_dual_fix(s: Core) -> None:
    """Fix each column whose objective and locks push it to one bound.

    Args:
      s: Presolver state.

    Raises:
      ValueError: If the objective pushes a column to an infinite bound.

    """
    cols_to_inf = s.int_vec
    cols_to_inf.len = 0
    for k in range(s.n):
        tag = s.col_tags[k]
        if tag & C_INACTIVE or (s.lock_up[k] > 0 and s.lock_down[k] > 0):
            continue
        ck = s.c[k]
        if ck > 0 and s.lock_down[k] == 0:
            if tag & C_LB_INF:
                raise ValueError("Presolve proved the program unbounded.")
            _fix_col(s, k, s.lb[k], ck)
        elif ck < 0 and s.lock_up[k] == 0:
            if tag & C_UB_INF:
                raise ValueError("Presolve proved the program unbounded.")
            _fix_col(s, k, s.ub[k], ck)
        elif ck == 0:
            if s.lock_down[k] == 0:
                if tag & C_LB_INF:
                    _ivec_append(cols_to_inf, -k)
                else:
                    _fix_col(s, k, s.lb[k], ck)
            elif s.lock_up[k] == 0:
                if tag & C_UB_INF:
                    _ivec_append(cols_to_inf, k)
                else:
                    _fix_col(s, k, s.ub[k], ck)
    for ii in range(cols_to_inf.len):
        col = cols_to_inf.data[ii]
        if col < 0:
            _fix_col_to_inf(s, -col, -1)
        elif col > 0:
            _fix_col_to_inf(s, col, 1)
        elif s.lock_down[0] == 0:
            _fix_col_to_inf(s, 0, -1)
        elif s.lock_up[0] == 0:
            _fix_col_to_inf(s, 0, 1)
    _delete_inactive_rows(s)
    _delete_fixed_cols_from_problem(s)
    _delete_inactive_cols_from_a_and_at(s)


# ---------------------------------------------------------------------------
# Column singletons (StonCols.c, CoreTransformation.c).
# ---------------------------------------------------------------------------


@njit(cache=True, error_model="numpy")
def remove_ston_cols(s: Core) -> None:
    """Repeat the singleton-column pass until it reduces nothing."""
    while _remove_ston_cols_pass(s) == REDUCED:
        pass


# ---------------------------------------------------------------------------
# Doubleton equality rows (DtonsEq.c).
# ---------------------------------------------------------------------------


@njit(cache=True, error_model="numpy")
def remove_dton_eq_rows_pass(s: Core, max_shift: int) -> int:
    """Substitute out one column of each queued doubleton equality row.

    Rows whose pivot or fill-in PSLP rejects go back on the queue for the
    next pass.

    Args:
      s: Presolver state.
      max_shift: Most entries of neighbouring columns of AT that
        ``_shift_row`` may move to make room for fill-in.

    Returns:
      status: ``REDUCED`` if any row was substituted, else ``UNCHANGED``.

    """
    a = s.A
    status = UNCHANGED
    s.int_vec.len = 0
    for ii in range(s.dton_rows.len):
        row = s.dton_rows.data[ii]
        j, k = _find_substitution(s, row, max_shift)
        if j < 0:
            continue
        status = REDUCED
        first = a.start[row]
        if k == a.i[first]:
            aik, aij = a.x[first], a.x[first + 1]
        else:
            aik, aij = a.x[first + 1], a.x[first]
        _modify_bounds(s, row, (j, k), (aij, aik))
        s.c[j] -= (aij / aik) * s.c[k]
        s.offset += (s.rhs[row] / aik) * s.c[k]
        s.row_tags[row] = R_INACTIVE
        s.row_sizes[row] = SIZE_INACTIVE
        a.end[row] = a.start[row]
        a.nnz -= 2
        _execute_substitution(s, row, (j, k), (aij, aik), s.c[k])
    s.AT.nnz = a.nnz
    s.dton_rows.len = 0
    for ii in range(s.int_vec.len):
        _ivec_append(s.dton_rows, s.int_vec.data[ii])
    return status


# ---------------------------------------------------------------------------
# Main-loop accessors; the loop itself is presolve.py's, so torch passes can
# run between kernels.
# ---------------------------------------------------------------------------


@njit(cache=True, error_model="numpy")
def nnz(s: Core) -> int:
    """Live nonzeros of A, which PSLP's phase and cycle rules compare."""
    return s.A.nnz


@njit(cache=True, error_model="numpy")
def dton_pending(s: Core) -> bool:
    """Whether doubleton equality rows are queued."""
    return s.dton_rows.len > 0


@njit(cache=True, error_model="numpy")
def problem_clean(s: Core) -> None:
    """Drop inactive rows and columns and renumber (problem_clean, remove_all).

    Args:
      s: Presolver state; ``map_rows`` and ``map_cols`` then give each
        original row's and column's new index, or -1.

    """
    n_cols_old, n_rows_old = s.n, s.m
    new_n_cols = _update_map(s.col_sizes, s.map_cols, n_cols_old)
    _remove_extra_space(s.A, s.row_sizes, True, s.map_cols, new_n_cols)
    _update_map(s.row_sizes, s.map_rows, n_rows_old)
    _shrink_f8(s.rhs, s.map_rows)
    _shrink_f8(s.lhs, s.map_rows)
    _shrink_f8(s.lb, s.map_cols)
    _shrink_f8(s.ub, s.map_cols)
    s.m, s.n = s.A.m, s.A.n
    for arr in (s.row_sizes, s.act_n_inf_min, s.act_n_inf_max):
        _shrink_i4(arr, s.map_rows)
    _shrink_f8(s.act_min, s.map_rows)
    _shrink_f8(s.act_max, s.map_rows)
    _shrink_u1(s.act_status, s.map_rows)
    for arr in (s.lock_up, s.lock_down, s.col_sizes):
        _shrink_i4(arr, s.map_cols)
    for vec in (s.ston_rows, s.dton_rows, s.empty_rows, s.updated_activities):
        _shrink_idx(vec, s.map_rows)
    for vec in (s.ston_cols, s.empty_cols):
        _shrink_idx(vec, s.map_cols)
    _shrink_f8(s.c, s.map_cols)


# ---------------------------------------------------------------------------
# Primal propagation (Primal_propagation.c), with PSLP's default finite-bound
# tightening on.
# ---------------------------------------------------------------------------


@njit(cache=True, error_model="numpy")
def propagate_primal(s: Core) -> None:
    """Tighten bounds row by row over the queue of rows with usable activity.

    The arrays are bound once: structref field reads cost reference counting
    per access, which dominated this loop when each entry read them afresh.
    None is ever reallocated here, so the bindings stay live across fixes.

    Args:
      s: Presolver state.

    Raises:
      ValueError: If an implied bound crosses the opposite bound.

    """
    queue = s.updated_activities
    ax, ai, a_start = s.A.x, s.A.i, s.A.start
    sizes, row_tags, col_tags = s.row_sizes, s.row_tags, s.col_tags
    lb, ub, lhs, rhs = s.lb, s.ub, s.lhs, s.rhs
    act_min, act_max = s.act_min, s.act_max
    n_inf_min, n_inf_max, status = s.act_n_inf_min, s.act_n_inf_max, s.act_status
    ii = 0
    while ii < queue.len:
        current_len = queue.len
        while ii < current_len:
            row = queue.data[ii]
            ii += 1
            n_min, n_max = n_inf_min[row], n_inf_max[row]
            tag = row_tags[row]
            if (
                sizes[row] <= 1
                or status[row] != ADDED
                or not (
                    n_min == 0
                    or n_max == 0
                    or (n_min == 1 and not tag & R_RHS_INF)
                    or (n_max == 1 and not tag & R_LHS_INF)
                )
            ):
                continue
            status[row] = PROPAGATED_THIS_ROUND
            first, length = a_start[row], sizes[row]
            for side in range(2):
                # bound_tightening_single_row_rhs, then its lhs mirror.
                rhs_side = side == 0
                side_inf = (row_tags[row] & (R_RHS_INF if rhs_side else R_LHS_INF)) != 0
                n_inf = n_inf_min[row] if rhs_side else n_inf_max[row]
                n_inf_other = n_inf_max[row] if rhs_side else n_inf_min[row]
                if not side_inf and n_inf == 0:
                    for jj in range(length):
                        k = ai[first + jj]
                        ctag = col_tags[k]
                        if ctag & C_INACTIVE:
                            continue
                        aik = ax[first + jj]
                        slack = (
                            rhs[row] - act_min[row]
                            if rhs_side
                            else lhs[row] - act_max[row]
                        )
                        # Positive coefficients bound from above on the rhs side.
                        upper = (aik > 0) == rhs_side
                        implied = (lb[k] if upper else ub[k]) + slack / aik
                        # _tighten_bound's test, inline: almost no entry passes it.
                        if abs(implied) < HUGE_VAL and (
                            (ctag & C_UB_INF or implied < ub[k])
                            if upper
                            else (ctag & C_LB_INF or implied > lb[k])
                        ):
                            _tighten_bound(s, k, implied, upper)
                elif (not side_inf or n_inf_other == 0) and n_inf == 1:
                    k = -1
                    aik = np.inf
                    for jj in range(length):
                        aik = ax[first + jj]
                        k = ai[first + jj]
                        ctag = col_tags[k]
                        lb_inf, ub_inf = (ctag & C_LB_INF) != 0, (ctag & C_UB_INF) != 0
                        if rhs_side:
                            culprit = lb_inf if aik > 0 else ub_inf
                        else:
                            culprit = ub_inf if aik > 0 else lb_inf
                        if culprit and not ctag & C_INACTIVE:
                            break
                    # compute_{min,max}_act_one_tag.
                    partial = 0.0
                    for kk in range(first, first + length):
                        col = ai[kk]
                        if col == k:
                            continue
                        take_ub = (ax[kk] > 0) != rhs_side
                        partial += ax[kk] * (ub[col] if take_ub else lb[col])
                    if rhs_side:
                        bound_side = act_max[row] if side_inf else rhs[row]
                    else:
                        bound_side = act_min[row] if side_inf else lhs[row]
                    implied = (bound_side - partial) / aik
                    _tighten_bound(s, k, implied, (aik > 0) == rhs_side)
    next_round = s.iwork_n_rows
    new_len = 0
    for jj in range(queue.len):
        row = queue.data[jj]
        if s.act_status[row] == PROPAGATE_NEXT_ROUND:
            next_round[new_len] = row
            new_len += 1
            s.act_status[row] = ADDED
    s.act_status[: s.A.m] = NOT_ADDED
    for jj in range(new_len):
        s.act_status[next_round[jj]] = ADDED
    queue.len = 0
    for jj in range(new_len):
        _ivec_append(queue, next_round[jj])
    _delete_fixed_cols_from_problem(s)
    _delete_inactive_cols_from_a_and_at(s)
    remove_empty_rows(s)


# ---------------------------------------------------------------------------
# Parallel rows and columns (Parallel_rows.c, Parallel_cols.c, radix_sort.c).
# ---------------------------------------------------------------------------


@njit(cache=True, error_model="numpy")
def parallel_hashes(
    s: Core,
    columns: bool,
) -> tuple[IBuf, IBuf, IBuf]:
    """Hash every active row of A (or column, via AT) for parallel detection.

    ``bulk.sort_rows`` then orders the rows as PSLP's sort does.

    Args:
      s: Presolver state.
      columns: Hash AT's rows (A's columns) instead of A's rows.

    Returns:
      active: Active rows (or columns), ascending.
      sparsity: Support hash per row, as int32.
      coeff: Coefficient hash per row, as int32.

    """
    mat = s.AT if columns else s.A
    tags = s.col_tags if columns else s.row_tags
    inactive = C_INACTIVE if columns else R_INACTIVE
    sparsity, coeff = s.iwork1, s.iwork2
    _row_hashes(mat, tags, inactive, sparsity, coeff)
    n_active = 0
    for row in range(mat.m):
        n_active += sparsity[row] != 2_147_483_647
    active = np.empty(n_active, np.int32)
    n_active = 0
    for row in range(mat.m):
        if sparsity[row] != 2_147_483_647:
            active[n_active] = row
            n_active += 1
    return active, sparsity, coeff


@njit(cache=True, error_model="numpy")
def remove_parallel_rows(s: Core, rows: IBuf) -> None:
    """Merge groups of parallel rows (``rows`` hash-sorted) into their kept row."""
    members, starts = _parallel_groups(s.A, rows, s.iwork1, s.iwork2)
    for g in range(starts.size - 1):
        _process_row_bin(s, members[starts[g] : starts[g + 1]])
    _delete_inactive_rows(s)


@njit(cache=True, error_model="numpy")
def remove_parallel_cols(s: Core, cols: IBuf) -> None:
    """Merge or fix groups of parallel columns.

    Args:
      s: Presolver state.
      cols: Active columns, sorted by hash (``bulk.sort_rows``).

    Raises:
      ValueError: If the objective drives a column to fix to an infinite bound.

    """
    members, starts = _parallel_groups(s.AT, cols, s.iwork1, s.iwork2)
    if starts.size > 1:
        recompute = s.iwork1
        recompute[: s.A.m] = 0
        for g in range(starts.size - 1):
            _process_col_bin(s, members[starts[g] : starts[g + 1]], recompute)
        for row in range(s.A.m):
            if recompute[row]:
                _recompute_n_infs(s, row)
    _delete_fixed_cols_from_problem(s)
    _delete_inactive_cols_from_a_and_at(s)


@njit(cache=True, error_model="numpy")
def reduced_arrays(
    s: Core,
) -> tuple[
    IBuf, IBuf, FBuf, FBuf, FBuf,
    FBuf, FBuf, FBuf, float,
]:  # fmt: skip
    """Return the cleaned program (populate_presolved_problem).

    Views into the state, not copies: nothing mutates it after problem_clean,
    and copying the paper-scale program cost 4 s and as many gigabytes.

    Args:
      s: State after ``problem_clean``.

    Returns:
      ap: Row pointers, as int64.
      ai: Column indices.
      ax: Values.
      lhs: Row lower sides.
      rhs: Row upper sides.
      c: Objective coefficients.
      lb: Column lower bounds.
      ub: Column upper bounds.
      offset: Objective offset the reductions accumulated.

    """
    a = s.A
    m, n = s.m, s.n
    stop = a.start[m]
    return (
        a.start[: m + 1].astype(np.int64),
        a.i[:stop],
        a.x[:stop],
        s.lhs[:m],
        s.rhs[:m],
        s.c[:n],
        s.lb[:n],
        s.ub[:n],
        s.offset,
    )


@njit(cache=True, error_model="numpy")
def _tighten_bound(s: Core, k: int, implied: float, upper: bool) -> None:
    """Apply one implied bound if it improves and is not huge."""
    if abs(implied) >= HUGE_VAL:
        return
    if upper:
        if s.col_tags[k] & C_UB_INF or implied < s.ub[k]:
            _propagate_ub(s, implied, k)
    elif s.col_tags[k] & C_LB_INF or implied > s.lb[k]:
        _propagate_lb(s, implied, k)


@njit(cache=True, error_model="numpy", parallel=True)
def _count_zeros(x: FBuf) -> int:
    """Explicit zeros in ``x``; almost always none, which skips PSLP's compaction."""
    zeros = 0
    for k in prange(x.size):
        zeros += x[k] == 0.0
    return zeros


# Parallel kernels take arrays bare or through a struct, never unpacked from a tuple
# argument: Numba 0.67's parfor silently drops writes to arrays unpacked from one.
@njit(cache=True, error_model="numpy", parallel=True)
def _scatter_transpose(a: Matrix, ordinal: IBuf, dest: IBuf, at: Matrix) -> None:
    """Scatter the rows of ``a`` to their slots ``dest`` in ``at``."""
    ax, starts, ends = a.x, a.start, a.end
    at_x, at_i = at.x, at.i
    for row in prange(ordinal.size - 1):
        base = ordinal[row] - starts[row]
        for k in range(starts[row], ends[row]):
            slot = dest[base + k]
            at_x[slot] = ax[k]
            at_i[slot] = row


@njit(cache=True, error_model="numpy", parallel=True)
def _new_row_tags(lhs: FBuf, rhs: FBuf, tags: IBuf) -> None:
    """Tag infinite sides and equalities, normalizing infinite sides to +-inf."""
    for row in prange(lhs.size):
        tag = 0
        if np.isinf(rhs[row]) and rhs[row] > 0:
            rhs[row] = np.inf
            tag |= R_RHS_INF
        if np.isinf(lhs[row]) and lhs[row] < 0:
            lhs[row] = -np.inf
            tag |= R_LHS_INF
        elif lhs[row] == rhs[row]:
            tag |= R_EQ
        tags[row] = tag


@njit(cache=True, error_model="numpy", parallel=True)
def _new_col_tags(lbs: FBuf, ubs: FBuf, tags: IBuf) -> None:
    for col in prange(lbs.size):
        tag = 0
        if np.isinf(lbs[col]) and lbs[col] < 0:
            tag |= C_LB_INF
        if np.isinf(ubs[col]) and ubs[col] > 0:
            tag |= C_UB_INF
        tags[col] = tag


@njit(cache=True, error_model="numpy", parallel=True)
def _new_locks(s: Core) -> None:
    """new_locks, column by column: integer counts, so any order is exact."""
    at = s.AT
    at_x, at_i, at_start, at_end = at.x, at.i, at.start, at.end
    row_tags, lock_up, lock_down = s.row_tags, s.lock_up, s.lock_down
    for col in prange(lock_up.size):
        up = down = 0
        for k in range(at_start[col], at_end[col]):
            tag = row_tags[at_i[k]]
            rhs_finite = (tag & R_RHS_INF) == 0
            lhs_finite = (tag & R_LHS_INF) == 0
            if at_x[k] > 0:
                up += rhs_finite
                down += lhs_finite
            else:
                down += rhs_finite
                up += lhs_finite
        lock_up[col] = up
        lock_down[col] = down


@njit(cache=True, error_model="numpy", parallel=True)
def _new_activities(s: Core) -> None:
    """new_activities, rows in parallel; each row still sums in PSLP's order."""
    a = s.A
    ax, ai, starts, ends = a.x, a.i, a.start, a.end
    col_tags, lb, ub = s.col_tags, s.lb, s.ub
    act_min, act_max = s.act_min, s.act_max
    n_inf_min_out, n_inf_max_out = s.act_n_inf_min, s.act_n_inf_max
    for row in prange(act_min.size):
        n_inf_min = n_inf_max = 0
        for k in range(starts[row], ends[row]):
            tag = col_tags[ai[k]]
            if ax[k] > 0:
                n_inf_max += (tag & C_UB_INF) != 0
                n_inf_min += (tag & C_LB_INF) != 0
            else:
                n_inf_max += (tag & C_LB_INF) != 0
                n_inf_min += (tag & C_UB_INF) != 0
        n_inf_min_out[row] = n_inf_min
        n_inf_max_out[row] = n_inf_max
        total_max = 0.0
        if n_inf_max == 0:
            for k in range(starts[row], ends[row]):
                col = ai[k]
                total_max += ax[k] * (ub[col] if ax[k] > 0 else lb[col])
        total_min = 0.0
        if n_inf_min == 0:
            for k in range(starts[row], ends[row]):
                col = ai[k]
                total_min += ax[k] * (lb[col] if ax[k] > 0 else ub[col])
        act_max[row] = total_max
        act_min[row] = total_min


# Each list holds its indices ascending, as PSLP's appending loops leave them.
@njit(cache=True, error_model="numpy")
def _new_state_lists(s: Core) -> None:
    """new_state: seed the work lists from row and column sizes and activities."""
    m, n = s.m, s.n
    sizes, cols = s.row_sizes, s.col_sizes
    s.empty_rows = _ivec_of(np.flatnonzero(np.equal(sizes, 0)))
    s.ston_rows = _ivec_of(np.flatnonzero(np.equal(sizes, 1)))
    is_eq = np.not_equal(np.bitwise_and(s.row_tags, R_EQ), 0)
    s.dton_rows = _ivec_of(np.flatnonzero(np.logical_and(np.equal(sizes, 2), is_eq)))
    s.empty_cols = _ivec_of(np.flatnonzero(np.equal(cols, 0)))
    s.ston_cols = _ivec_of(np.flatnonzero(np.equal(cols, 1)))
    usable = np.logical_or(np.equal(s.act_n_inf_min, 0), np.equal(s.act_n_inf_max, 0))
    updated = np.flatnonzero(usable)
    for row in updated:
        s.act_status[row] = ADDED
    s.updated_activities = _ivec_of(updated)
    s.fixed_cols_to_delete = _ivec_new(n // 100)
    s.sub_cols_to_delete = _ivec_new(n // 100)
    s.rows_to_delete = _ivec_new(m // 100)
    s.ps_starts = _ivec_new(m // 4)
    s.ps_indices = _ivec_new(m // 4)
    s.ps_type = _ivec_new(m // 4)
    s.ps_vals = _dvec_new(m // 4)
    _ivec_append(s.ps_starts, 0)


@njit(cache=True, error_model="numpy")
def _ivec_of(values: IBuf) -> IVec:
    """Return a vector holding ``values``, with room to grow."""
    vec = _ivec_new(values.size + 16)
    vec.data[: values.size] = values
    vec.len = values.size
    return vec


@njit(cache=True, error_model="numpy")
def _close_record(s: Core, kind: int) -> None:
    _ivec_append(s.ps_type, kind)
    _ivec_append(s.ps_starts, s.ps_indices.len)


@njit(cache=True, error_model="numpy")
def _retrieve_sub_col(
    sol: FBuf,
    idx: IBuf,
    vals: FBuf,
    start: int,
    length: int,
) -> None:
    k = idx[start]
    value = vals[start]
    aik = 0.0
    for ii in range(start + 1, start + 1 + length):
        if idx[ii] == k:
            aik = vals[ii]
            continue
        value -= vals[ii] * sol[idx[ii]]
    sol[k] = value / aik


@njit(cache=True, error_model="numpy")
def _retrieve_fix_col_inf(
    sol: FBuf,
    idx: IBuf,
    vals: FBuf,
    start: int,
) -> None:
    n_rows = int(vals[start])
    extreme = vals[start + 1]
    to_pos_inf = idx[start] > 0
    col = idx[start + 1]
    coeff = 0.0
    counter = start + 2
    for _ in range(n_rows):
        side = vals[counter]
        row_len = idx[counter]
        for jj in range(counter + 1, counter + 1 + row_len):
            if idx[jj] == col:
                coeff = vals[jj]
                continue
            if sol[idx[jj]] == np.inf:
                continue
            side -= vals[jj] * sol[idx[jj]]
        counter += row_len + 1
        val = side / coeff
        extreme = max(extreme, val) if to_pos_inf else min(extreme, val)
    sol[col] = extreme


@njit(cache=True, error_model="numpy")
def _retrieve_parallel_col(
    sol: FBuf,
    idx: IBuf,
    vals: FBuf,
    start: int,
) -> None:
    j, k, tag_j, tag_k = idx[start], idx[start + 1], idx[start + 2], idx[start + 3]
    lb_j, ub_j, lb_k, ub_k, ratio = vals[start : start + 5]
    x_new = sol[j]
    if not tag_j & C_LB_INF:
        xj = lb_j
    elif not tag_j & C_UB_INF:
        xj = ub_j
    else:
        xj = 0.0
    xk = (x_new - xj) / ratio
    if not tag_k & C_LB_INF and lb_k >= xk + FEAS_TOL:
        xk = lb_k
        xj = x_new - ratio * xk
    elif not tag_k & C_UB_INF and xk >= ub_k + FEAS_TOL:
        xk = ub_k
        xj = x_new - ratio * xk
    sol[j] = xj
    sol[k] = xk


@njit(cache=True, error_model="numpy")
def _mark_updated(s: Core, row: int, altered: int) -> None:
    """Queue ``row`` for activity checks when a finite side moved."""
    if (s.act_n_inf_max[row] == 0 and altered & MAX_ALTERED) or (
        s.act_n_inf_min[row] == 0 and altered & MIN_ALTERED
    ):
        if s.act_status[row] == NOT_ADDED:
            s.act_status[row] = ADDED
            _ivec_append(s.updated_activities, row)
        elif s.act_status[row] == PROPAGATED_THIS_ROUND:
            s.act_status[row] = PROPAGATE_NEXT_ROUND
            _ivec_append(s.updated_activities, row)


@njit(cache=True, error_model="numpy")
def _update_lock_coeff_deletion(
    s: Core,
    col: int,
    val: float,
    rhs_inf: bool,
    lhs_inf: bool,
) -> None:
    if val > 0:
        if not rhs_inf:
            s.lock_up[col] -= 1
        if not lhs_inf:
            s.lock_down[col] -= 1
    else:
        if not rhs_inf:
            s.lock_down[col] -= 1
        if not lhs_inf:
            s.lock_up[col] -= 1


@njit(cache=True, error_model="numpy")
def _remove_ston_row(s: Core, row: int) -> None:
    """Turn a singleton row into a fixing (equality) or bound change (inequality)."""
    a = s.A
    k = a.i[a.start[row]]
    aik = a.x[a.start[row]]
    if s.col_tags[k] & C_FIXED:
        return
    lhs, rhs = s.lhs[row], s.rhs[row]
    if s.row_tags[row] & R_EQ:
        _fix_col(s, k, rhs / aik, s.c[k])
        s.row_sizes[row] = SIZE_INACTIVE
        s.row_tags[row] = R_INACTIVE
        a.end[row] = a.start[row]
        a.nnz -= 1
        return
    rhs_inf = (s.row_tags[row] & R_RHS_INF) != 0
    lhs_inf = (s.row_tags[row] & R_LHS_INF) != 0
    if aik > 0:
        if not rhs_inf:
            _update_ub(s, k, rhs / aik)
        if not lhs_inf:
            _update_lb(s, k, lhs / aik)
    else:
        if not rhs_inf:
            _update_lb(s, k, rhs / aik)
        if not lhs_inf:
            _update_ub(s, k, lhs / aik)
    _set_row_to_inactive(s, row)


@njit(cache=True, error_model="numpy")
def _check_activity(s: Core, row: int) -> int:
    """Which sides of ``row`` its activity bounds make redundant (16: infeasible)."""
    tag = s.row_tags[row]
    result = 0
    if not tag & R_RHS_INF:
        if s.act_n_inf_min[row] == 0 and s.act_min[row] >= s.rhs[row] + FEAS_TOL:
            return 16
        if s.act_n_inf_max[row] == 0 and s.act_max[row] <= s.rhs[row] + FEAS_TOL:
            result |= R_RHS_INF
    if not tag & R_LHS_INF:
        if s.act_n_inf_max[row] == 0 and s.act_max[row] <= s.lhs[row] - FEAS_TOL:
            return 16
        if s.act_n_inf_min[row] == 0 and s.act_min[row] >= s.lhs[row] - FEAS_TOL:
            result |= R_LHS_INF
    return result


@njit(cache=True, error_model="numpy")
def _update_locks_side_change(s: Core, row: int, sign: int, flip: bool) -> None:
    """Add ``sign`` to down locks of positive entries (up if ``flip``), and so on."""
    a = s.A
    for kk in range(a.start[row], a.start[row] + s.row_sizes[row]):
        down = (a.x[kk] > 0) != flip
        if down:
            s.lock_down[a.i[kk]] += sign
        else:
            s.lock_up[a.i[kk]] += sign


@njit(cache=True, error_model="numpy")
def _fix_col_to_inf(s: Core, col: int, sign: int) -> None:
    """fix_col_to_{negative,positive}_inf: drop ``col`` and every row it meets."""
    at = s.AT
    _save_fixed_col_inf(s, col, sign, s.ub[col] if sign < 0 else s.lb[col])
    for kk in range(at.start[col], at.end[col]):
        row = at.i[kk]
        if not s.row_tags[row] & R_INACTIVE:
            _set_row_to_inactive(s, row)
    s.col_tags[col] |= C_FIXED
    s.col_sizes[col] = SIZE_INACTIVE
    at.end[col] = at.start[col]
    s.lb[col] = sign * np.inf
    s.ub[col] = sign * np.inf


@njit(cache=True, error_model="numpy")
def _remove_ston_cols_pass(s: Core) -> int:
    at = s.AT
    cols = s.ston_cols
    kept = 0
    for ii in range(cols.len):
        if s.col_sizes[cols.data[ii]] == 1:
            cols.data[kept] = cols.data[ii]
            kept += 1
    cols.len = kept
    status = UNCHANGED
    for kk in range(cols.len):
        k = cols.data[kk]
        i = at.i[at.start[k]]
        aik = at.x[at.start[k]]
        if s.row_tags[i] & R_INACTIVE or s.row_sizes[i] <= 1:
            continue
        free = (
            _implied_free_from_above(s, aik, i, k),
            _implied_free_from_below(s, aik, i, k),
        )
        if s.row_tags[i] & R_EQ:
            status |= _colston_eq(s, i, k, aik, free)
        else:
            status |= _colston_ineq(s, i, k, aik, free)
    at.nnz = s.A.nnz
    _delete_inactive_rows(s)
    return status


@njit(cache=True, error_model="numpy")
def _substitute_col(s: Core, k: int) -> None:
    s.col_tags[k] |= C_SUBSTITUTED
    s.col_sizes[k] = SIZE_INACTIVE
    s.AT.end[k] = s.AT.start[k]


@njit(cache=True, error_model="numpy")
def _colston_eq(s: Core, i: int, k: int, aik: float, free: tuple[bool, bool]) -> int:
    """process_colston_eq: eliminate a free column singleton of an equality row."""
    free_above, free_below = free
    if not free_above and not free_below:
        return UNCHANGED
    ck = s.c[k]
    _sub_var_in_obj(s, i, k, aik, s.rhs[i])
    _substitute_col(s, k)
    if free_above and free_below:
        _save_sub_col(s, k, i, s.row_sizes[i], (s.rhs[i], ck))
        _set_row_to_inactive(s, i)
        return REDUCED
    lb, ub = s.lb[k], s.ub[k]
    tag = s.col_tags[k]
    _save_sub_col(s, k, i, s.row_sizes[i], (s.rhs[i], 0.0))
    _remove_coeff(s.A, i, k, s.row_sizes)
    s.A.nnz -= 1
    # Implied free from above keeps the side the finite lower bound meets;
    # from below, the side the finite upper bound meets.
    bound = lb if free_above else ub
    keeps_rhs = (aik > 0) == free_above
    if keeps_rhs:
        s.rhs[i] -= bound * aik
        s.lhs[i] = -np.inf
        s.row_tags[i] = R_LHS_INF
        _update_locks_side_change(s, i, -1, False)
    else:
        s.lhs[i] -= bound * aik
        s.rhs[i] = np.inf
        s.row_tags[i] = R_RHS_INF
        _update_locks_side_change(s, i, -1, True)
    dropped = (
        bound,
        ub if free_above else lb,
        (tag & (C_UB_INF if free_above else C_LB_INF)) != 0,
    )
    _drop_bound_activity(s, i, aik, dropped, keeps_rhs)
    # PSLP also queues the row as a doubleton equality here, but it has just
    # become an inequality.
    if s.row_sizes[i] == 1:
        _ivec_append(s.ston_rows, i)
    return REDUCED


# handle_impl_free_*_eq: the kept bound leaves the activity side matching the kept row
# side (min for a kept rhs), the other bound or its infinity the other.
@njit(cache=True, error_model="numpy")
def _drop_bound_activity(
    s: Core,
    row: int,
    aik: float,
    dropped: tuple[float, float, bool],
    keeps_rhs: bool,
) -> None:
    """Remove a substituted singleton's contributions from its row's activity."""
    bound, other, other_inf = dropped
    if keeps_rhs:
        if s.act_n_inf_min[row] == 0:
            s.act_min[row] -= bound * aik
        if other_inf:
            s.act_n_inf_max[row] -= 1
            if s.act_n_inf_max[row] == 0:
                s.act_max[row] = _max_act_no_tags(s, row)
        elif s.act_n_inf_max[row] == 0:
            s.act_max[row] -= other * aik
    else:
        if s.act_n_inf_max[row] == 0:
            s.act_max[row] -= bound * aik
        if other_inf:
            s.act_n_inf_min[row] -= 1
            if s.act_n_inf_min[row] == 0:
                s.act_min[row] = _min_act_no_tags(s, row)
        elif s.act_n_inf_min[row] == 0:
            s.act_min[row] -= other * aik


@njit(cache=True, error_model="numpy")
def _colston_ineq(s: Core, i: int, k: int, aik: float, free: tuple[bool, bool]) -> int:
    """process_colston_ineq: substitute, or tighten an inequality to equality."""
    free_above, free_below = free
    tag_row = s.row_tags[i]
    lhs_inf = (tag_row & R_LHS_INF) != 0
    rhs_inf = (tag_row & R_RHS_INF) != 0
    ub_inf = (s.col_tags[k] & C_UB_INF) != 0
    lb_inf = (s.col_tags[k] & C_LB_INF) != 0
    ck = s.c[k]
    if (
        (ck > 0 and aik > 0 and lhs_inf and lb_inf)
        or (ck > 0 and aik < 0 and rhs_inf and lb_inf)
        or (ck < 0 and aik < 0 and lhs_inf and ub_inf)
        or (ck < 0 and aik > 0 and rhs_inf and ub_inf)
    ):
        raise ValueError("Presolve proved the program unbounded.")
    # PSLP raises again below when the side the objective pushes to is infinite;
    # a column free on that side has the matching bound infinite, so the check
    # above has already raised.
    if free_above and free_below:
        if (ck > 0 and aik > 0) or (ck < 0 and aik < 0):
            new_side = s.lhs[i]
        elif (ck > 0 and aik < 0) or (ck < 0 and aik > 0):
            new_side = s.rhs[i]
        else:
            new_side = s.rhs[i] if lhs_inf else s.lhs[i]
        _sub_var_in_obj(s, i, k, aik, new_side)
        _substitute_col(s, k)
        _save_sub_col(s, k, i, s.row_sizes[i], (new_side, ck))
        _set_row_to_inactive(s, i)
        return REDUCED
    if (ck < 0 and aik > 0 and free_above) or (ck > 0 and aik < 0 and free_below):
        if lhs_inf:
            _update_locks_side_change(s, i, 1, False)
        s.lhs[i] = s.rhs[i]
    elif (ck > 0 and aik > 0 and free_below) or (ck < 0 and aik < 0 and free_above):
        if rhs_inf:
            _update_locks_side_change(s, i, 1, True)
        s.rhs[i] = s.lhs[i]
    else:
        return UNCHANGED
    s.row_tags[i] = R_EQ
    if s.row_sizes[i] == 2:
        _ivec_append(s.dton_rows, i)
    return REDUCED


@njit(cache=True, error_model="numpy")
def _sufficient_space_at(s: Core, j: int, k: int, max_shift: int) -> bool:
    """Whether column j of AT can absorb column k's fill-in, shifting if needed."""
    at = s.AT
    rj, len_j = at.start[j], at.end[j] - at.start[j]
    rk, len_k = at.start[k], at.end[k] - at.start[k]
    fill_in = -1
    jj = kk = 0
    while jj < len_j and kk < len_k:
        if at.i[rj + jj] == at.i[rk + kk]:
            jj += 1
            kk += 1
        elif at.i[rk + kk] < at.i[rj + jj]:
            kk += 1
            fill_in += 1
        else:
            jj += 1
    fill_in += len_k - kk
    free_space = at.start[j + 1] - at.end[j]
    return free_space >= fill_in or _shift_row(at, j, fill_in, max_shift)


@njit(cache=True, error_model="numpy")
def _find_substitution(s: Core, row: int, max_shift: int) -> tuple[int, int]:
    """(stay, subst) columns for doubleton ``row``, or (-1, -1) when rejected."""
    a = s.A
    if s.row_sizes[row] < 2:
        return -1, -1
    first = a.start[row]
    c0, c1 = a.i[first], a.i[first + 1]
    if (s.col_tags[c0] | s.col_tags[c1]) & C_INACTIVE:
        return -1, -1
    v0, v1 = a.x[first], a.x[first + 1]
    a_abs, b_abs = abs(v0), abs(v1)
    ratio0, ratio1 = a_abs / b_abs, b_abs / a_abs
    integral0 = ratio0 == int(ratio0)
    integral1 = ratio1 == int(ratio1)
    size0, size1 = s.col_sizes[c0], s.col_sizes[c1]
    if size0 == 1 and size1 != 1:
        subst = 0
    elif (size0 != 1 and size1 == 1) or (integral0 and not integral1):
        subst = 1
    elif integral1 and not integral0:
        subst = 0
    else:
        subst = 0 if size0 < size1 else 1
    stay = 1 - subst
    vals = (v0, v1)
    cols = (c0, c1)
    pivot = abs(vals[stay] / vals[subst])
    if (
        pivot > MAX_RATIO_PIVOT
        or pivot < 1 / MAX_RATIO_PIVOT
        or not _sufficient_space_at(s, cols[stay], cols[subst], max_shift)
    ):
        # Rejected rows are retried on the next pass.
        _ivec_append(s.int_vec, row)
        return -1, -1
    return cols[stay], cols[subst]


# PSLP returns early, and its caller skips the row, when the column that stays comes out
# fixed; ``update_lb`` and ``update_ub`` never fix a column, so neither happens.
@njit(cache=True, error_model="numpy")
def _modify_bounds(
    s: Core,
    row: int,
    cols: tuple[int, int],
    coeffs: tuple[float, float],
) -> None:
    """Move the substituted column's bounds onto the column that stays."""
    j, k = cols
    aij, aik = coeffs
    rhs = s.rhs[row]
    lb_subst, ub_subst, tag_subst = s.lb[k], s.ub[k], s.col_tags[k]
    if aik * aij > 0.0:
        if not tag_subst & C_LB_INF:
            _update_ub(s, j, (rhs - aik * lb_subst) / aij)
        if not tag_subst & C_UB_INF:
            _update_lb(s, j, (rhs - aik * ub_subst) / aij)
    else:
        if not tag_subst & C_LB_INF:
            _update_lb(s, j, (rhs - aik * lb_subst) / aij)
        if not tag_subst & C_UB_INF:
            _update_ub(s, j, (rhs - aik * ub_subst) / aij)


@njit(cache=True, error_model="numpy")
def _update_row_a_dton(
    s: Core,
    q: int,
    cols: tuple[int, int],
    coeffs: tuple[float, float],
) -> tuple[float, float, float]:
    """Substitute k out of row q of A; return (old stay, old subst, new stay)."""
    j, k = cols
    aij, aik = coeffs
    a = s.A
    start, end = a.start[q], a.end[q]
    row_len = end - start
    subst_idx = start + _sorted_find(a.i, start, row_len, k)
    insertion = start + _sorted_lower_bound(a.i, start, row_len, j)
    old_val = 0.0
    if insertion != end and a.i[insertion] == j:
        old_val = a.x[insertion]
    old_subst = a.x[subst_idx]
    new_val = old_val - (aij / aik) * old_subst
    diff = 0
    _move_entries(a, subst_idx, subst_idx + 1, end - subst_idx - 1)
    end -= 1
    diff += 1
    if subst_idx < insertion:
        insertion -= 1
    if abs(new_val) <= ZERO_TOL:
        _move_entries(a, insertion, insertion + 1, end - insertion - 1)
        end -= 1
        diff += 1
    elif insertion == end:
        a.x[insertion] = new_val
        a.i[insertion] = j
        end += 1
        diff -= 1
    elif a.i[insertion] == j:
        a.x[insertion] = new_val
    else:
        _move_entries(a, insertion + 1, insertion, end - insertion)
        a.x[insertion] = new_val
        a.i[insertion] = j
        end += 1
        diff -= 1
    a.end[q] = end
    s.row_sizes[q] -= diff
    a.nnz -= diff
    return old_val, old_subst, new_val


@njit(cache=True, error_model="numpy")
def _execute_substitution(
    s: Core,
    row: int,
    cols: tuple[int, int],
    coeffs: tuple[float, float],
    ck: float,
) -> None:
    """Eliminate column k from every row it appears in, through doubleton ``row``."""
    j, k = cols
    aik = coeffs[1]
    at = s.AT
    rows_subst = at.i[at.start[k] : at.end[k]].copy()
    _remove_coeff(at, j, row, s.col_sizes)
    at.end[k] = at.start[k]
    rhs_dton = s.rhs[row]
    for q in rows_subst:
        if s.row_tags[q] & R_INACTIVE:
            continue
        old_len = s.row_sizes[q]
        old_stay, old_subst, new_stay = _update_row_a_dton(s, q, cols, coeffs)
        size = s.row_sizes[q]
        if size == 0:
            _ivec_append(s.empty_rows, q)
        elif size == 1:
            if old_len != 1:
                _ivec_append(s.ston_rows, q)
        elif size == 2 and s.row_tags[q] & R_EQ and old_len != 2:
            _ivec_append(s.int_vec, q)
        _insert_or_update_coeff(at, j, q, new_stay, s.col_sizes)
        if rhs_dton != 0.0:
            change = old_subst / aik * rhs_dton
            if not s.row_tags[q] & R_LHS_INF:
                s.lhs[q] -= change
            if not s.row_tags[q] & R_RHS_INF:
                s.rhs[q] -= change
        altered = _update_activity_coeff_change(s, q, j, old_stay, new_stay)
        altered |= _update_activity_coeff_change(s, q, k, old_subst, 0.0)
        if altered & MIN_ALTERED_RECOMPUTE:
            s.act_min[q] = _min_act_no_tags(s, q)
            _mark_added(s, q)
        if altered & MAX_ALTERED_RECOMPUTE:
            s.act_max[q] = _max_act_no_tags(s, q)
            _mark_added(s, q)
    if s.col_sizes[j] == 0:
        _ivec_append(s.empty_cols, j)
    elif s.col_sizes[j] == 1:
        _ivec_append(s.ston_cols, j)
    s.col_tags[k] = C_INACTIVE
    s.col_sizes[k] = SIZE_INACTIVE
    _count_locks_one_column(s, j)
    _save_sub_col(s, k, row, 2, (rhs_dton, ck))


@njit(cache=True, error_model="numpy")
def _mark_added(s: Core, row: int) -> None:
    if s.act_status[row] == NOT_ADDED:
        s.act_status[row] = ADDED
        _ivec_append(s.updated_activities, row)


@njit(cache=True, error_model="numpy")
def _update_map(sizes: IBuf, mapping: IBuf, n: int) -> int:
    count = 0
    for idx in range(n):
        if sizes[idx] == SIZE_INACTIVE:
            mapping[idx] = -1
        else:
            mapping[idx] = count
            count += 1
    return count


@njit(cache=True, error_model="numpy")
def _shrink_f8(arr: FBuf, mapping: IBuf) -> None:
    for idx in range(mapping.size):
        if mapping[idx] != -1:
            arr[mapping[idx]] = arr[idx]


@njit(cache=True, error_model="numpy")
def _shrink_i4(arr: IBuf, mapping: IBuf) -> None:
    for idx in range(mapping.size):
        if mapping[idx] != -1:
            arr[mapping[idx]] = arr[idx]


@njit(cache=True, error_model="numpy")
def _shrink_u1(arr: IBuf, mapping: IBuf) -> None:
    for idx in range(mapping.size):
        if mapping[idx] != -1:
            arr[mapping[idx]] = arr[idx]


@njit(cache=True, error_model="numpy")
def _shrink_idx(vec: IVec, mapping: IBuf) -> None:
    kept = 0
    for ii in range(vec.len):
        mapped = mapping[vec.data[ii]]
        if mapped != -1:
            vec.data[kept] = mapped
            kept += 1
    vec.len = kept


@njit(cache=True, error_model="numpy")
def _col_max_abs(s: Core, col: int) -> float:
    at = s.AT
    best = 0.0
    for kk in range(at.start[col], at.end[col]):
        best = max(best, abs(at.x[kk]))
    return best


@njit(cache=True, error_model="numpy")
def _propagate_lb(s: Core, new_lb: float, col: int) -> int:
    """update_lb_within_propagation."""
    tag = s.col_tags[col]
    is_ub_inf = (tag & C_UB_INF) != 0
    is_lb_inf = (tag & C_LB_INF) != 0
    lb, ub = s.lb[col], s.ub[col]
    if not is_ub_inf:
        if new_lb >= ub + FEAS_TOL:
            raise ValueError("Presolve proved the program infeasible.")
        if new_lb >= ub or (ub - new_lb) * _col_max_abs(s, col) <= FEAS_TOL:
            _fix_col(s, col, ub, s.c[col])
            return REDUCED
    if is_lb_inf or (new_lb - lb > FEAS_TOL * 1e4 and new_lb - lb > 1e-2 * abs(lb)):
        if new_lb != int(new_lb):
            new_lb -= 0.5 * FEAS_TOL * abs(new_lb)
        s.col_tags[col] &= 0xFF ^ C_LB_INF
        s.lb[col] = new_lb
        _update_activities_bound_change(s, col, (lb, new_lb, not is_lb_inf, True))
        return REDUCED
    return UNCHANGED


@njit(cache=True, error_model="numpy")
def _propagate_ub(s: Core, new_ub: float, col: int) -> int:
    """update_ub_within_propagation."""
    tag = s.col_tags[col]
    is_lb_inf = (tag & C_LB_INF) != 0
    is_ub_inf = (tag & C_UB_INF) != 0
    lb, ub = s.lb[col], s.ub[col]
    if not is_lb_inf:
        if new_ub <= lb - FEAS_TOL:
            raise ValueError("Presolve proved the program infeasible.")
        if new_ub <= lb or (new_ub - lb) * _col_max_abs(s, col) <= FEAS_TOL:
            _fix_col(s, col, lb, s.c[col])
            return REDUCED
    if is_ub_inf or (ub - new_ub > FEAS_TOL * 1e4 and ub - new_ub > 1e-2 * abs(ub)):
        if new_ub != int(new_ub):
            new_ub += 0.5 * FEAS_TOL * abs(new_ub)
        s.col_tags[col] &= 0xFF ^ C_UB_INF
        s.ub[col] = new_ub
        _update_activities_bound_change(s, col, (ub, new_ub, not is_ub_inf, False))
        return REDUCED
    return UNCHANGED


@njit(cache=True, error_model="numpy")
def _c_round(value: float) -> int:
    """C's round of a value within int64: halves away from zero, not to even."""
    truncated = int(value)
    if abs(value - truncated) >= 0.5:
        truncated += 1 if value > 0 else -1
    return truncated


# PSLP hashes in uint32 arithmetic and stores the hash as an int; the loops below do the
# same in int64 masked to 32 bits (33 * h + x stays below 2**38) and store the int32
# that the uint32 reinterprets as.
@njit(cache=True, error_model="numpy", parallel=True)
def _row_hashes(
    mat: Matrix,
    tags: IBuf,
    inactive: int,
    sparsity: IBuf,
    coeff: IBuf,
) -> None:
    """compute_supp_and_coeff_hash: djb2 over columns and scaled coefficients."""
    x, idx, starts, ends = mat.x, mat.i, mat.start, mat.end
    for row in prange(mat.m):
        if tags[row] & inactive:
            sparsity[row] = 2_147_483_647
            continue
        start, end = starts[row], ends[row]
        h = 5381
        for kk in range(start, end):
            h = (h * 33 + int(idx[kk])) & 0xFFFF_FFFF
        sparsity[row] = h - (1 << 32) if h >= 1 << 31 else h
        biggest = abs(x[start])
        for kk in range(start + 1, end):
            biggest = max(biggest, abs(x[kk]))
        scale = 1 / biggest if x[start] > 0 else -1 / biggest
        h = 5381
        for kk in range(start, end):
            # x86-64 converts a negative double to uint32 through int64.
            scaled = _c_round((x[kk] * scale) * 1e6) & 0xFFFF_FFFF
            h = (h * 33 + scaled) & 0xFFFF_FFFF
        coeff[row] = h - (1 << 32) if h >= 1 << 31 else h


@njit(cache=True, error_model="numpy")
def _parallel_groups(
    mat: Matrix,
    rows: IBuf,
    sparsity: IBuf,
    coeff: IBuf,
) -> tuple[IBuf, IBuf]:
    """Group parallel rows among hash-sorted rows: (flat members, group starts)."""
    members = _ivec_new(16)
    starts = _ivec_new(16)
    _ivec_append(starts, 0)
    i = 0
    while i < rows.size:
        size = 1
        while (
            i + size < rows.size
            and sparsity[rows[i + size]] == sparsity[rows[i]]
            and coeff[rows[i + size]] == coeff[rows[i]]
        ):
            size += 1
        if size > 1:
            _parallel_in_bin(mat, rows[i : i + size], members, starts)
        i += size
    return members.data[: members.len].copy(), starts.data[: starts.len].copy()


@njit(cache=True, error_model="numpy")
def _parallel_in_bin(
    mat: Matrix,
    bin_rows: IBuf,
    members: IVec,
    starts: IVec,
) -> None:
    """find_parallel_rows_in_bin: rows parallel to the bin's first, then the first."""
    first = bin_rows[0]
    s1, len1 = mat.start[first], mat.end[first] - mat.start[first]
    found = 0
    for b in range(1, bin_rows.size):
        other = bin_rows[b]
        s2, len2 = mat.start[other], mat.end[other] - mat.start[other]
        if len1 != len2:
            continue
        ratio = mat.x[s1] / mat.x[s2]
        same = True
        for jj in range(len2):
            diff = mat.x[s1 + jj] - ratio * mat.x[s2 + jj]
            if abs(diff) > FEAS_TOL or mat.i[s1 + jj] != mat.i[s2 + jj]:
                same = False
                break
        if same:
            _ivec_append(members, other)
            found += 1
    if found > 0:
        _ivec_append(members, first)
        _ivec_append(starts, starts.data[starts.len - 1] + found + 1)


@njit(cache=True, error_model="numpy")
def _change_side_of_ineq(s: Core, row: int, new_side: float, rhs_side: bool) -> None:
    """change_{rhs,lhs}_of_ineq: tighten one side of an inequality."""
    a = s.A
    tag = s.row_tags[row]
    other_inf = (tag & (R_LHS_INF if rhs_side else R_RHS_INF)) != 0
    if rhs_side:
        if not other_inf and s.lhs[row] >= new_side + FEAS_TOL:
            raise ValueError("Presolve proved the program infeasible.")
    elif not other_inf and s.rhs[row] <= new_side - FEAS_TOL:
        raise ValueError("Presolve proved the program infeasible.")
    was_inf = (tag & (R_RHS_INF if rhs_side else R_LHS_INF)) != 0
    length = a.end[row] - a.start[row]
    if was_inf:
        _update_locks_side_change(s, row, 1, rhs_side)
    if rhs_side:
        s.rhs[row] = new_side
    else:
        s.lhs[row] = new_side
    s.row_tags[row] &= 0xFF ^ (R_RHS_INF if rhs_side else R_LHS_INF)
    if not s.row_tags[row] & (R_LHS_INF if rhs_side else R_RHS_INF) and (
        abs(s.lhs[row] - s.rhs[row]) <= FEAS_TOL
    ):
        # The side just set takes the other's exact value.
        if rhs_side:
            s.rhs[row] = s.lhs[row]
        else:
            s.lhs[row] = s.rhs[row]
        s.row_tags[row] |= R_EQ
        if length == 2:
            _ivec_append(s.dton_rows, row)


@njit(cache=True, error_model="numpy")
def _process_row_bin(s: Core, bin_rows: IBuf) -> None:
    """Parallel_rows.c process_single_bin."""
    a = s.A
    keep = bin_rows[0]
    keep_eq = (s.row_tags[keep] & R_EQ) != 0
    keep_rhs_inf = (s.row_tags[keep] & R_RHS_INF) != 0
    keep_lhs_inf = (s.row_tags[keep] & R_LHS_INF) != 0
    new_rhs, new_lhs = s.rhs[keep], s.lhs[keep]
    keep_coeff = a.x[a.start[keep]]
    for b in range(1, bin_rows.size):
        other = bin_rows[b]
        other_eq = (s.row_tags[other] & R_EQ) != 0
        if not keep_eq and other_eq:
            keep, other = other, keep
            keep_eq, other_eq = True, False
            new_rhs, new_lhs = s.rhs[keep], s.lhs[keep]
            keep_coeff = a.x[a.start[keep]]
        ratio = keep_coeff / a.x[a.start[other]]
        other_rhs, other_lhs = s.rhs[other] * ratio, s.lhs[other] * ratio
        other_rhs_inf = (s.row_tags[other] & R_RHS_INF) != 0
        other_lhs_inf = (s.row_tags[other] & R_LHS_INF) != 0
        if keep_eq and other_eq:
            if abs(new_rhs - other_rhs) > FEAS_TOL:
                raise ValueError("Presolve proved the program infeasible.")
        elif keep_eq:
            infeasible_rhs = not other_rhs_inf and (
                (ratio > 0 and new_rhs > other_rhs)
                or (ratio < 0 and new_rhs < other_rhs)
            )
            infeasible_lhs = not other_lhs_inf and (
                (ratio > 0 and new_rhs < other_lhs)
                or (ratio < 0 and new_rhs > other_lhs)
            )
            if infeasible_rhs or infeasible_lhs:
                raise ValueError("Presolve proved the program infeasible.")
        elif ratio > 0:
            if not other_rhs_inf and (keep_rhs_inf or other_rhs < new_rhs):
                new_rhs = other_rhs
                keep_rhs_inf = False
            if not other_lhs_inf and (keep_lhs_inf or other_lhs > new_lhs):
                new_lhs = other_lhs
                keep_lhs_inf = False
        else:
            if not other_rhs_inf and (keep_lhs_inf or other_rhs > new_lhs):
                new_lhs = other_rhs
                keep_lhs_inf = False
            if not other_lhs_inf and (keep_rhs_inf or other_lhs < new_rhs):
                new_rhs = other_lhs
                keep_rhs_inf = False
    if not keep_eq:
        if not keep_rhs_inf and new_rhs != s.rhs[keep]:
            _change_side_of_ineq(s, keep, new_rhs, True)
        if not keep_lhs_inf and new_lhs != s.lhs[keep]:
            _change_side_of_ineq(s, keep, new_lhs, False)
    for b in range(bin_rows.size):
        if bin_rows[b] != keep:
            _set_row_to_inactive(s, bin_rows[b])


@njit(cache=True, error_model="numpy")
def _recompute_n_infs(s: Core, row: int) -> None:
    a = s.A
    n_inf_min = n_inf_max = 0
    for kk in range(a.start[row], a.end[row]):
        tag = s.col_tags[a.i[kk]]
        if tag & C_INACTIVE:
            continue
        if a.x[kk] > 0:
            n_inf_max += (tag & C_UB_INF) != 0
            n_inf_min += (tag & C_LB_INF) != 0
        else:
            n_inf_max += (tag & C_LB_INF) != 0
            n_inf_min += (tag & C_UB_INF) != 0
    s.act_n_inf_max[row] = n_inf_max
    s.act_n_inf_min[row] = n_inf_min


@njit(cache=True, error_model="numpy")
def _process_col_bin(s: Core, bin_cols: IBuf, recompute: IBuf) -> None:
    """Parallel_cols.c process_single_bin."""
    at = s.AT
    for ii in range(bin_cols.size - 1):
        j = bin_cols[ii]
        if s.col_tags[j] & C_INACTIVE:
            continue
        cj = s.c[j]
        first_j = at.start[j]
        len_j = at.end[j] - first_j
        aj0 = at.x[first_j]
        recount = False
        for jj in range(ii + 1, bin_cols.size):
            k = bin_cols[jj]
            if s.col_tags[k] & C_INACTIVE:
                continue
            ck = s.c[k]
            ak0 = at.x[at.start[k]]
            ratio = ak0 / aj0
            if abs(ck * aj0 - cj * ak0) <= FEAS_TOL:
                recount = True
                _merge_parallel_cols(s, j, k, ratio)
                continue
            outcome = _fix_parallel_cols(s, (j, k), (cj, ck), ratio)
            if outcome == 2:
                recount = False
                break
        if recount:
            for kk in range(first_j, first_j + len_j):
                recompute[at.i[kk]] = 1


@njit(cache=True, error_model="numpy")
def _merge_parallel_cols(s: Core, j: int, k: int, ratio: float) -> None:
    """Replace x_j and x_k by x_j + ratio * x_k, widening j's bounds."""
    old = np.array([s.lb[j], s.ub[j], 0.0, 0.0])
    tags = s.col_tags[j] | s.col_tags[k]
    if ratio > 0:
        lb_inf, ub_inf = (tags & C_LB_INF) != 0, (tags & C_UB_INF) != 0
        lb_k, ub_k = s.lb[k], s.ub[k]
    else:
        lb_inf = (s.col_tags[j] & C_LB_INF) != 0 or (s.col_tags[k] & C_UB_INF) != 0
        ub_inf = (s.col_tags[j] & C_UB_INF) != 0 or (s.col_tags[k] & C_LB_INF) != 0
        lb_k, ub_k = s.ub[k], s.lb[k]
    if lb_inf:
        s.lb[j] = -np.inf
        s.col_tags[j] |= C_LB_INF
    else:
        s.lb[j] += lb_k * ratio
    if ub_inf:
        s.ub[j] = np.inf
        s.col_tags[j] |= C_UB_INF
    else:
        s.ub[j] += ub_k * ratio
    s.col_tags[k] |= C_SUBSTITUTED
    _ivec_append(s.sub_cols_to_delete, k)
    old[2], old[3] = s.lb[k], s.ub[k]
    _save_parallel_col(s, j, k, old, ratio)


# Returns 2 if j was fixed, 1 if k was, else 0.
@njit(cache=True, error_model="numpy")
def _fix_parallel_cols(
    s: Core,
    cols: tuple[int, int],
    costs: tuple[float, float],
    ratio: float,
) -> int:
    """Fix one of two parallel columns the objective prefers."""
    j, k = cols
    cj, ck = costs
    tag_j, tag_k = s.col_tags[j], s.col_tags[k]
    k_lower = k_upper = j_upper = j_lower = False
    if ck > ratio * cj:
        if ratio > 0:
            k_lower = (tag_j & C_UB_INF) != 0
            j_upper = (tag_k & C_LB_INF) != 0
        else:
            k_lower = (tag_j & C_LB_INF) != 0
            j_lower = (tag_k & C_LB_INF) != 0
    elif ratio > 0:
        k_upper = (tag_j & C_LB_INF) != 0
        j_lower = (tag_k & C_UB_INF) != 0
    else:
        k_upper = (tag_j & C_UB_INF) != 0
        j_upper = (tag_k & C_UB_INF) != 0
    if k_lower or k_upper:
        if tag_k & (C_LB_INF if k_lower else C_UB_INF):
            raise ValueError("Presolve proved the program unbounded.")
        _fix_col(s, k, s.lb[k] if k_lower else s.ub[k], ck)
        return 1
    # PSLP raises here too if j's bound is infinite; in each branch above that
    # bound being infinite sets k's flag, so k has been fixed instead.
    if j_lower or j_upper:
        _fix_col(s, j, s.lb[j] if j_lower else s.ub[j], cj)
        return 2
    return 0


@njit(cache=True, error_model="numpy")
def _remove_explicit_zeros(a: Matrix) -> None:
    """Close each row over its explicit zeros, as PSLP stores only nonzeros."""
    for row in range(a.m):
        shift = 0
        for k in range(a.start[row], a.end[row]):
            if a.x[k] == 0.0:
                shift += 1
            elif shift > 0:
                a.x[k - shift] = a.x[k]
                a.i[k - shift] = a.i[k]
        a.end[row] -= shift
        a.nnz -= shift


@njit(cache=True, error_model="numpy")
def _ivec_new(capacity: int) -> IVec:
    """Allocate an empty int32 vector."""
    vec = _new(IVEC_T)
    vec.data = np.empty(max(capacity, 1), np.int32)
    vec.len = 0
    return vec


@njit(cache=True, error_model="numpy")
def _dvec_new(capacity: int) -> DVec:
    """Allocate an empty float64 vector."""
    vec = _new(DVEC_T)
    vec.data = np.empty(max(capacity, 1), np.float64)
    vec.len = 0
    return vec


# Capacity never changes contents, so growth need not follow PSLP's policy.
@njit(cache=True, error_model="numpy")
def _ivec_append(vec: IVec, value: int) -> None:
    """Append one value, doubling capacity when full (Vec_macros.h)."""
    if vec.len == vec.data.size:
        grown = np.empty(2 * vec.data.size, np.int32)
        grown[: vec.len] = vec.data[: vec.len]
        vec.data = grown
    vec.data[vec.len] = value
    vec.len += 1


@njit(cache=True, error_model="numpy")
def _dvec_append(vec: DVec, value: float) -> None:
    """Append one value."""
    if vec.len == vec.data.size:
        grown = np.empty(2 * vec.data.size, np.float64)
        grown[: vec.len] = vec.data[: vec.len]
        vec.data = grown
    vec.data[vec.len] = value
    vec.len += 1


@njit(cache=True, error_model="numpy")
def _calc_memory_row(size: int, extra_row_space: int, memory_ratio: float) -> int:
    """Capacity PSLP reserves for a row of ``size`` entries."""
    return int(size * memory_ratio) + extra_row_space


@njit(cache=True, error_model="numpy")
def _matrix_new_no_extra_space(
    ax: FBuf,
    ai: IBuf,
    ap: IBuf,
    n_rows: int,
    n_cols: int,
) -> Matrix:
    """Copy CSR arrays without spare capacity, dropping explicit zeros."""
    a = _new(MATRIX_T)
    nnz = ap[n_rows]
    a.m, a.n, a.nnz, a.n_alloc = n_rows, n_cols, nnz, nnz
    a.x = ax[:nnz].copy()
    a.i = ai[:nnz].astype(np.int32)
    a.start = ap[: n_rows + 1].astype(np.int32)
    a.end = np.empty(n_rows + 1, np.int32)
    a.end[:n_rows] = ap[1 : n_rows + 1]
    # PSLP reads Ap[n_rows + 1] here, one past the array; the sentinel row is
    # never read, so its end is set to its start.
    a.end[n_rows] = ap[n_rows]
    if _count_zeros(a.x) > 0:
        _remove_explicit_zeros(a)
    return a


@njit(cache=True, error_model="numpy")
def _sorted_find(arr: IBuf, first: int, length: int, target: int) -> int:
    """Relative index of ``target`` in ``arr[first:first+length]``, or -1."""
    if length <= 8:
        for k in range(length):
            if arr[first + k] == target:
                return k
            if arr[first + k] > target:
                return -1
        return -1
    lo, hi = 0, length - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if arr[first + mid] == target:
            return mid
        if arr[first + mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


@njit(cache=True, error_model="numpy")
def _sorted_lower_bound(arr: IBuf, first: int, length: int, target: int) -> int:
    """First relative index in ``arr[first:first+length]`` holding ``>= target``."""
    if length <= 8:
        for k in range(length):
            if arr[first + k] >= target:
                return k
        return length
    lo, hi = 0, length
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if arr[first + mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True, error_model="numpy")
def _move_entries(a: Matrix, dst: int, src: int, length: int) -> None:
    """Memmove ``length`` entries of ``a`` from ``src`` to ``dst`` (ranges may overlap)."""
    x, idx = a.x, a.i
    if dst < src:
        for k in range(length):
            x[dst + k] = x[src + k]
            idx[dst + k] = idx[src + k]
    else:
        for k in range(length - 1, -1, -1):
            x[dst + k] = x[src + k]
            idx[dst + k] = idx[src + k]


@njit(cache=True, error_model="numpy")
def _insert_or_update_coeff(
    a: Matrix,
    row: int,
    col: int,
    val: float,
    sizes: IBuf,
) -> float:
    """Set ``a[row, col]`` to ``val`` (removing it at zero); return the old value."""
    old_val = 0.0
    start, end = a.start[row], a.end[row]
    insertion = start + _sorted_lower_bound(a.i, start, end - start, col)
    if abs(val) > ZERO_TOL:
        if insertion == end:
            a.x[insertion] = val
            a.i[insertion] = col
            a.end[row] += 1
            a.nnz += 1
            sizes[row] += 1
        elif a.i[insertion] == col:
            old_val = a.x[insertion]
            a.x[insertion] = val
        else:
            _move_entries(a, insertion + 1, insertion, end - insertion)
            a.x[insertion] = val
            a.i[insertion] = col
            a.end[row] += 1
            a.nnz += 1
            sizes[row] += 1
    else:
        if insertion != end - 1:
            _move_entries(a, insertion, insertion + 1, end - insertion - 1)
        a.end[row] -= 1
        a.nnz -= 1
        sizes[row] -= 1
    return old_val


@njit(cache=True, error_model="numpy")
def _remove_coeff(a: Matrix, row: int, col: int, sizes: IBuf) -> None:
    """Delete ``a[row, col]``, which must exist, shifting the row's tail left."""
    first = a.start[row]
    length = sizes[row]
    shift = 0
    for k in range(length):
        if a.i[first + k] == col:
            shift = 1
        # PSLP reads one entry past the row's end here; that slot is outside
        # the row once the end moves left, so any value it copies is inert.
        if shift and k + 1 < length:
            a.x[first + k] = a.x[first + k + 1]
            a.i[first + k] = a.i[first + k + 1]
    a.end[row] -= 1
    sizes[row] -= 1


@njit(cache=True, error_model="numpy")
def _shift_row(a: Matrix, row: int, extra_space: int, max_shift: int) -> bool:
    """Open ``extra_space`` slots after ``row`` by moving neighbours; False if not."""
    left, right = row, row + 1
    remaining_shifts = max_shift
    left_shifts = right_shifts = 0
    # The one caller shifts only when the row lacks space, so this is positive.
    missing_space = extra_space - (a.start[right] - a.end[row])
    while missing_space > 0:
        if left == 0 and right == a.m:
            return False
        space_left = 0 if left == 0 else a.start[left] - a.end[left - 1]
        space_right = 0 if right == a.m else a.start[right + 1] - a.end[right]
        n_move_right = a.end[right] - a.start[right]
        n_move_left = a.end[left] - a.start[left]
        if left == 0:
            if right != a.m and n_move_right <= remaining_shifts:
                shift_left = False
            else:
                return False
        elif right == a.m:
            if n_move_left <= remaining_shifts:
                shift_left = True
            else:
                return False
        elif n_move_left == 0:
            shift_left = True
        elif n_move_right == 0:
            shift_left = False
        elif n_move_left <= remaining_shifts and (
            space_left / float(n_move_left) >= space_right / float(n_move_right)
        ):
            shift_left = True
        elif n_move_right <= remaining_shifts:
            shift_left = False
        else:
            return False
        if shift_left:
            left_shifts = min(missing_space, space_left)
            missing_space -= left_shifts
            remaining_shifts -= n_move_left
            left -= 1
        else:
            right_shifts = min(missing_space, space_right)
            missing_space -= right_shifts
            remaining_shifts -= n_move_right
            right += 1
    next_start = a.start[left + 1] - left_shifts
    while left < row:
        length = a.end[left + 1] - a.start[left + 1]
        if length > 0:
            _move_entries(a, next_start, a.start[left + 1], length)
        a.start[left + 1] = next_start
        a.end[left + 1] = next_start + length
        next_start += length
        left += 1
    next_end = a.end[right - 1] + right_shifts
    while right > row + 1:
        length = a.end[right - 1] - a.start[right - 1]
        if length > 0:
            _move_entries(a, next_end - length, a.start[right - 1], length)
        a.start[right - 1] = next_end - length
        a.end[right - 1] = next_end
        next_end = a.start[right - 1]
        right -= 1
    return True


@njit(cache=True, error_model="numpy")
def _remove_extra_space(
    a: Matrix,
    sizes: IBuf,
    remove_all: bool,
    col_map: IBuf,
    new_n_cols: int,
) -> None:
    """Compact live rows to the front, drop inactive rows, and renumber columns."""
    extra_row_space = 0 if remove_all else EXTRA_ROW_SPACE
    extra_mem_ratio = 1.0 if remove_all else EXTRA_MEMORY_RATIO
    x, idx, starts, ends = a.x, a.i, a.start, a.end
    curr = 0
    n_deleted = 0
    for row in range(a.m):
        if sizes[row] == SIZE_INACTIVE:
            n_deleted += 1
            continue
        start = starts[row]
        length = ends[row] - start
        for k in range(length):
            x[curr + k] = x[start + k]
            idx[curr + k] = idx[start + k]
        starts[row - n_deleted] = curr
        ends[row - n_deleted] = curr + length
        curr += _calc_memory_row(length, extra_row_space, extra_mem_ratio)
    a.m -= n_deleted
    starts[a.m] = curr
    ends[a.m] = curr
    a.n = new_n_cols
    for row in range(a.m):
        for k in range(starts[row], ends[row]):
            idx[k] = col_map[idx[k]]


@njit(cache=True, error_model="numpy")
def _save_fixed_col(s: Core, col: int, val: float, ck: float, column: bool) -> None:
    """Record ``col`` fixed to ``val``, with its column of AT if ``column``."""
    _ivec_append(s.ps_indices, col)
    _ivec_append(s.ps_indices, -382_749)
    _dvec_append(s.ps_vals, val)
    _dvec_append(s.ps_vals, ck)
    if column:
        for k in range(s.AT.start[col], s.AT.end[col]):
            _ivec_append(s.ps_indices, s.AT.i[k])
            _dvec_append(s.ps_vals, s.AT.x[k])
    _close_record(s, FIXED_COL)


@njit(cache=True, error_model="numpy")
def _save_sub_col(
    s: Core,
    col: int,
    row: int,
    length: int,
    values: tuple[float, float],
) -> None:
    """Record ``col`` substituted out through equality ``row``."""
    rhs, ck = values
    a = s.A
    _ivec_append(s.ps_indices, col)
    _dvec_append(s.ps_vals, rhs)
    for k in range(a.start[row], a.start[row] + length):
        _ivec_append(s.ps_indices, a.i[k])
        _dvec_append(s.ps_vals, a.x[k])
    _ivec_append(s.ps_indices, row)
    _dvec_append(s.ps_vals, ck)
    _close_record(s, SUB_COL)


@njit(cache=True, error_model="numpy")
def _save_parallel_col(s: Core, j: int, k: int, bounds: FBuf, ratio: float) -> None:
    """Record ``k`` merged into ``j``; ``bounds`` is (lb_j, ub_j, lb_k, ub_k)."""
    _ivec_append(s.ps_indices, j)
    _ivec_append(s.ps_indices, k)
    _ivec_append(s.ps_indices, s.col_tags[j])
    _ivec_append(s.ps_indices, s.col_tags[k])
    _ivec_append(s.ps_indices, -382_749)
    for b in range(4):
        _dvec_append(s.ps_vals, bounds[b])
    _dvec_append(s.ps_vals, ratio)
    _close_record(s, PARALLEL_COL)


@njit(cache=True, error_model="numpy")
def _save_fixed_col_inf(s: Core, col: int, pos_inf: int, bound: float) -> None:
    """Record ``col`` fixed to +-inf, with every row it appears in."""
    at, a = s.AT, s.A
    n_rows = at.end[col] - at.start[col]
    _ivec_append(s.ps_indices, pos_inf)
    _ivec_append(s.ps_indices, col)
    _dvec_append(s.ps_vals, float(n_rows))
    _dvec_append(s.ps_vals, bound)
    for kk in range(at.start[col], at.end[col]):
        row = at.i[kk]
        _dvec_append(
            s.ps_vals,
            s.rhs[row] if s.row_tags[row] & R_LHS_INF else s.lhs[row],
        )
        size = s.row_sizes[row]
        for k in range(a.start[row], a.start[row] + size):
            _dvec_append(s.ps_vals, a.x[k])
        _ivec_append(s.ps_indices, size)
        for k in range(a.start[row], a.start[row] + size):
            _ivec_append(s.ps_indices, a.i[k])
    _close_record(s, FIXED_COL_INF)


@njit(cache=True, error_model="numpy")
def _min_act_no_tags(s: Core, row: int) -> float:
    """Row minimum activity from bounds, ignoring infinity tags."""
    a = s.A
    total = 0.0
    for k in range(a.start[row], a.end[row]):
        col = a.i[k]
        total += a.x[k] * (s.lb[col] if a.x[k] > 0 else s.ub[col])
    return total


@njit(cache=True, error_model="numpy")
def _max_act_no_tags(s: Core, row: int) -> float:
    """Row maximum activity from bounds, ignoring infinity tags."""
    a = s.A
    total = 0.0
    for k in range(a.start[row], a.end[row]):
        col = a.i[k]
        total += a.x[k] * (s.ub[col] if a.x[k] > 0 else s.lb[col])
    return total


@njit(cache=True, error_model="numpy")
def _min_act_tags(s: Core, row: int) -> float:
    """Row minimum activity over finite contributions only."""
    a = s.A
    total = 0.0
    for k in range(a.start[row], a.end[row]):
        col = a.i[k]
        if a.x[k] > 0:
            if not s.col_tags[col] & C_LB_INF:
                total += a.x[k] * s.lb[col]
        elif not s.col_tags[col] & C_UB_INF:
            total += a.x[k] * s.ub[col]
    return total


@njit(cache=True, error_model="numpy")
def _max_act_tags(s: Core, row: int) -> float:
    """Row maximum activity over finite contributions only."""
    a = s.A
    total = 0.0
    for k in range(a.start[row], a.end[row]):
        col = a.i[k]
        if a.x[k] > 0:
            if not s.col_tags[col] & C_UB_INF:
                total += a.x[k] * s.ub[col]
        elif not s.col_tags[col] & C_LB_INF:
            total += a.x[k] * s.lb[col]
    return total


@njit(cache=True, error_model="numpy")
def _update_act_bound_change(
    s: Core,
    row: int,
    coeff: float,
    change: BoundChange,
) -> int:
    """Shift one row's activity for one column's bound change; return what moved."""
    old_bound, new_bound, finite_bound, lower = change
    affects_max = (coeff < 0) == lower
    if affects_max:
        if s.act_n_inf_max[row] == 0:
            s.act_max[row] += (new_bound - old_bound) * coeff
        elif not finite_bound:
            s.act_n_inf_max[row] -= 1
            if s.act_n_inf_max[row] == 0:
                s.act_max[row] = _max_act_no_tags(s, row)
        return MAX_ALTERED
    if s.act_n_inf_min[row] == 0:
        s.act_min[row] += (new_bound - old_bound) * coeff
    elif not finite_bound:
        s.act_n_inf_min[row] -= 1
        if s.act_n_inf_min[row] == 0:
            s.act_min[row] = _min_act_no_tags(s, row)
    return MIN_ALTERED


@njit(cache=True, error_model="numpy")
def _update_activities_bound_change(s: Core, col: int, change: BoundChange) -> None:
    """Propagate one column's bound change into the activity of each of its rows."""
    at = s.AT
    for kk in range(at.start[col], at.end[col]):
        row = at.i[kk]
        altered = _update_act_bound_change(s, row, at.x[kk], change)
        _mark_updated(s, row, altered)


# Fixing leaves the infinity tags, so they still say which old bounds were infinite.
@njit(cache=True, error_model="numpy")
def _update_activities_fixed_col(
    s: Core,
    col: int,
    old_lb: float,
    old_ub: float,
    val: float,
) -> None:
    """Propagate fixing ``col`` to ``val`` into the activity of each of its rows."""
    is_ub_inf = (s.col_tags[col] & C_UB_INF) != 0
    is_lb_inf = (s.col_tags[col] & C_LB_INF) != 0
    ub_update = is_ub_inf or val != old_ub
    lb_update = is_lb_inf or val != old_lb
    at = s.AT
    for kk in range(at.start[col], at.end[col]):
        row = at.i[kk]
        altered = NO_RECOMPUTE
        if ub_update:
            altered |= _update_act_bound_change(
                s,
                row,
                at.x[kk],
                (old_ub, val, not is_ub_inf, False),
            )
        if lb_update:
            altered |= _update_act_bound_change(
                s,
                row,
                at.x[kk],
                (old_lb, val, not is_lb_inf, True),
            )
        _mark_updated(s, row, altered)


@njit(cache=True, error_model="numpy")
def _update_activity_coeff_change(
    s: Core,
    row: int,
    col: int,
    old: float,
    new: float,
) -> int:
    """Swap ``col``'s coefficient's contribution to a row's activity; flag recomputes."""
    lb, ub, tag = s.lb[col], s.ub[col], s.col_tags[col]
    lb_finite = not tag & C_LB_INF
    ub_finite = not tag & C_UB_INF
    inf_min_before, inf_max_before = s.act_n_inf_min[row], s.act_n_inf_max[row]
    for coeff, sign in ((old, -1.0), (new, 1.0)):
        step = int(sign)
        if coeff > 0:
            if lb_finite:
                if s.act_n_inf_min[row] == 0:
                    s.act_min[row] += sign * (coeff * lb)
            else:
                s.act_n_inf_min[row] += step
            if ub_finite:
                if s.act_n_inf_max[row] == 0:
                    s.act_max[row] += sign * (coeff * ub)
            else:
                s.act_n_inf_max[row] += step
        elif coeff < 0:
            if lb_finite:
                if s.act_n_inf_max[row] == 0:
                    s.act_max[row] += sign * (coeff * lb)
            else:
                s.act_n_inf_max[row] += step
            if ub_finite:
                if s.act_n_inf_min[row] == 0:
                    s.act_min[row] += sign * (coeff * ub)
            else:
                s.act_n_inf_min[row] += step
    recompute_min = s.act_n_inf_min[row] == 0 and inf_min_before == 1
    recompute_max = s.act_n_inf_max[row] == 0 and inf_max_before == 1
    if recompute_max and recompute_min:
        return MIN_ALTERED_RECOMPUTE | MAX_ALTERED_RECOMPUTE
    if recompute_max:
        return MAX_ALTERED_RECOMPUTE
    if recompute_min:
        return MIN_ALTERED_RECOMPUTE
    return NO_RECOMPUTE


@njit(cache=True, error_model="numpy")
def _fix_col(s: Core, col: int, val: float, ck: float) -> None:
    """Fix ``col`` to ``val`` and queue it for deletion."""
    old_lb, old_ub = s.lb[col], s.ub[col]
    tag = s.col_tags[col]
    is_ub_inf = (tag & C_UB_INF) != 0
    is_lb_inf = (tag & C_LB_INF) != 0
    if (not is_ub_inf and val >= old_ub + FEAS_TOL) or (
        not is_lb_inf and val <= old_lb - FEAS_TOL
    ):
        raise ValueError("Presolve proved the program infeasible.")
    s.col_tags[col] |= C_FIXED
    _ivec_append(s.fixed_cols_to_delete, col)
    _save_fixed_col(s, col, val, ck, True)
    s.ub[col] = val
    s.lb[col] = val
    _update_activities_fixed_col(s, col, old_lb, old_ub, val)


@njit(cache=True, error_model="numpy")
def _update_lb(s: Core, col: int, new_lb: float) -> None:
    """Tighten ``col``'s lower bound to ``new_lb`` if it improves by FEAS_TOL."""
    lb, ub = s.lb[col], s.ub[col]
    if not s.col_tags[col] & C_UB_INF and new_lb >= ub + FEAS_TOL:
        raise ValueError("Presolve proved the program infeasible.")
    is_lb_inf = (s.col_tags[col] & C_LB_INF) != 0
    if is_lb_inf or new_lb >= lb + FEAS_TOL:
        s.lb[col] = new_lb
        _update_activities_bound_change(s, col, (lb, new_lb, not is_lb_inf, True))
        s.col_tags[col] &= 0xFF ^ C_LB_INF


@njit(cache=True, error_model="numpy")
def _update_ub(s: Core, col: int, new_ub: float) -> None:
    """Tighten ``col``'s upper bound to ``new_ub`` if it improves by FEAS_TOL."""
    lb, ub = s.lb[col], s.ub[col]
    if not s.col_tags[col] & C_LB_INF and new_ub <= lb - FEAS_TOL:
        raise ValueError("Presolve proved the program infeasible.")
    is_ub_inf = (s.col_tags[col] & C_UB_INF) != 0
    if is_ub_inf or new_ub <= ub - FEAS_TOL:
        s.ub[col] = new_ub
        _update_activities_bound_change(s, col, (ub, new_ub, not is_ub_inf, False))
        s.col_tags[col] &= 0xFF ^ C_UB_INF


@njit(cache=True, error_model="numpy")
def _set_row_to_inactive(s: Core, row: int) -> None:
    """Mark ``row`` inactive, keeping its infinity tags for the lock update."""
    _ivec_append(s.rows_to_delete, row)
    s.row_tags[row] |= R_INACTIVE


@njit(cache=True, error_model="numpy")
def _count_locks_one_column(s: Core, col: int) -> None:
    """Recount ``col``'s locks from its column in AT."""
    at = s.AT
    up = down = 0
    for kk in range(at.start[col], at.start[col] + s.col_sizes[col]):
        tag = s.row_tags[at.i[kk]]
        rhs_finite = not tag & R_RHS_INF
        lhs_finite = not tag & R_LHS_INF
        if at.x[kk] > 0:
            up += rhs_finite
            down += lhs_finite
        else:
            down += rhs_finite
            up += lhs_finite
    s.lock_up[col] = up
    s.lock_down[col] = down


@njit(cache=True, error_model="numpy")
def _delete_inactive_rows(s: Core) -> None:
    """Drop queued rows from A's sizes, AT and the locks."""
    if s.rows_to_delete.len == 0:
        return
    a, at = s.A, s.AT
    for ii in range(s.rows_to_delete.len):
        row = s.rows_to_delete.data[ii]
        if s.row_sizes[row] == 0:
            s.row_sizes[row] = SIZE_INACTIVE
            continue
        a.nnz -= s.row_sizes[row]
        s.row_sizes[row] = SIZE_INACTIVE
        rhs_inf = (s.row_tags[row] & R_RHS_INF) != 0
        lhs_inf = (s.row_tags[row] & R_LHS_INF) != 0
        for k in range(a.start[row], a.end[row]):
            col = a.i[k]
            if s.col_sizes[col] == SIZE_INACTIVE:
                continue
            s.col_sizes[col] -= 1
            _update_lock_coeff_deletion(s, col, a.x[k], rhs_inf, lhs_inf)
        a.end[row] = a.start[row]
    for col in range(at.m):
        if (
            s.col_tags[col] & C_INACTIVE
            or s.col_sizes[col] == at.end[col] - at.start[col]
        ):
            continue
        size = s.col_sizes[col]
        if size == 0:
            _ivec_append(s.empty_cols, col)
            at.end[col] = at.start[col]
        elif size == 1:
            _ivec_append(s.ston_cols, col)
        shift = 0
        for k in range(at.start[col], at.end[col]):
            if s.row_sizes[at.i[k]] == SIZE_INACTIVE:
                shift += 1
            else:
                at.x[k - shift] = at.x[k]
                at.i[k - shift] = at.i[k]
        at.end[col] -= shift
    at.nnz = a.nnz
    s.rows_to_delete.len = 0


@njit(cache=True, error_model="numpy")
def _delete_inactive_cols_from_a_and_at(s: Core) -> None:
    """Drop queued fixed and substituted columns from A's rows and AT."""
    a, at = s.A, s.AT
    for vec in (s.fixed_cols_to_delete, s.sub_cols_to_delete):
        for ii in range(vec.len):
            col = vec.data[ii]
            s.col_sizes[col] = SIZE_INACTIVE
            for k in range(at.start[col], at.end[col]):
                if s.row_sizes[at.i[k]] != SIZE_INACTIVE:
                    s.row_sizes[at.i[k]] -= 1
            at.end[col] = at.start[col]
    for row in range(a.m):
        if (
            s.row_tags[row] & R_INACTIVE
            or s.row_sizes[row] == a.end[row] - a.start[row]
        ):
            continue
        size = s.row_sizes[row]
        if size == 0:
            _ivec_append(s.empty_rows, row)
            a.nnz -= a.end[row] - a.start[row]
            a.end[row] = a.start[row]
        elif size == 1:
            _ivec_append(s.ston_rows, row)
        elif size == 2 and s.row_tags[row] & R_EQ:
            _ivec_append(s.dton_rows, row)
        shift = 0
        for k in range(a.start[row], a.end[row]):
            if s.col_sizes[a.i[k]] == SIZE_INACTIVE:
                shift += 1
            else:
                a.x[k - shift] = a.x[k]
                a.i[k - shift] = a.i[k]
        a.end[row] -= shift
        a.nnz -= shift
    at.nnz = a.nnz
    s.fixed_cols_to_delete.len = 0
    s.sub_cols_to_delete.len = 0


@njit(cache=True, error_model="numpy")
def _delete_fixed_cols_from_problem(s: Core) -> None:
    """Move each queued fixed column's value into row sides, activities, offset."""
    at = s.AT
    for ii in range(s.fixed_cols_to_delete.len):
        col = s.fixed_cols_to_delete.data[ii]
        ub = s.ub[col]
        if ub == 0:
            continue
        for kk in range(at.start[col], at.end[col]):
            row = at.i[kk]
            if s.row_tags[row] & R_INACTIVE:
                continue
            contribution = at.x[kk] * ub
            if not s.row_tags[row] & R_LHS_INF:
                s.lhs[row] -= contribution
            if not s.row_tags[row] & R_RHS_INF:
                s.rhs[row] -= contribution
            if s.act_n_inf_max[row] == 0:
                s.act_max[row] -= contribution
            if s.act_n_inf_min[row] == 0:
                s.act_min[row] -= contribution
        s.offset += s.c[col] * ub


@njit(cache=True, error_model="numpy")
def _sub_var_in_obj(s: Core, row: int, k: int, aik: float, rhs: float) -> None:
    """Substitute column ``k`` out of the objective through equality ``row``."""
    if s.c[k] == 0.0:
        return
    ratio = s.c[k] / aik
    a = s.A
    for kk in range(a.start[row], a.start[row] + s.row_sizes[row]):
        s.c[a.i[kk]] -= ratio * a.x[kk]
    s.offset += rhs * ratio


@njit(cache=True, error_model="numpy")
def _implied_free_from_above(s: Core, aik: float, row: int, col: int) -> bool:
    """Whether ``row`` implies ``col``'s upper bound."""
    tag = s.col_tags[col]
    if tag & C_UB_INF:
        return True
    lb, ub = s.lb[col], s.ub[col]
    implied_ub = np.inf
    if aik > 0 and not s.row_tags[row] & R_RHS_INF:
        if s.act_n_inf_min[row] == 0:
            implied_ub = lb + (s.rhs[row] - s.act_min[row]) / aik
        elif s.act_n_inf_min[row] == 1 and tag & C_LB_INF:
            implied_ub = (s.rhs[row] - _min_act_tags(s, row)) / aik
    elif aik < 0 and not s.row_tags[row] & R_LHS_INF:
        if s.act_n_inf_max[row] == 0:
            implied_ub = lb + (s.lhs[row] - s.act_max[row]) / aik
        elif s.act_n_inf_max[row] == 1 and tag & C_LB_INF:
            implied_ub = (s.lhs[row] - _max_act_tags(s, row)) / aik
    return implied_ub - ub <= 0


@njit(cache=True, error_model="numpy")
def _implied_free_from_below(s: Core, aik: float, row: int, col: int) -> bool:
    """Whether ``row`` implies ``col``'s lower bound."""
    tag = s.col_tags[col]
    if tag & C_LB_INF:
        return True
    lb, ub = s.lb[col], s.ub[col]
    implied_lb = -np.inf
    if aik > 0 and not s.row_tags[row] & R_LHS_INF:
        if s.act_n_inf_max[row] == 0:
            implied_lb = ub + (s.lhs[row] - s.act_max[row]) / aik
        elif s.act_n_inf_max[row] == 1 and tag & C_UB_INF:
            implied_lb = (s.lhs[row] - _max_act_tags(s, row)) / aik
    elif aik < 0 and not s.row_tags[row] & R_RHS_INF:
        if s.act_n_inf_min[row] == 0:
            implied_lb = ub + (s.rhs[row] - s.act_min[row]) / aik
        elif s.act_n_inf_min[row] == 1 and tag & C_UB_INF:
            implied_lb = (s.rhs[row] - _min_act_tags(s, row)) / aik
    return lb - implied_lb <= 0
