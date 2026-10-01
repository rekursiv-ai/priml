"""Reduce a ConvexTok program exactly as cuOpt's default presolve (PSLP 0.0.8) does.

cuOpt runs PDLP on the problem PSLP returns, so the reduced program, its
objective offset and the recovery of an original primal point must match PSLP
bit for bit. The reductions are Numba kernels (``core``); the whole-matrix sorts
run in torch (``bulk``); this module sequences them as PSLP's main loop does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor

import torch

from priml.baselines.convextok.presolver import bulk, core
from priml.baselines.convextok.program import LinearProgram


if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class Presolved:
    """A program reduced by presolve, and the state that maps its solutions back.

    Attributes:
      program: The reduced program; its arrays are views into ``state``.
      state: The presolver's final state, holding the postsolve records.
      num_columns: Columns of the original program.

    """

    program: LinearProgram
    state: core.Core
    num_columns: int

    def postsolve(self, primal: Tensor) -> Tensor:
        """Map a primal point of the reduced program to the original columns.

        Args:
          primal: Value per reduced column.

        Returns:
          primal: Value per original column.

        Raises:
          ValueError: If ``primal`` does not have one value per reduced column;
            the Numba kernel would read past it rather than fail.

        """
        if primal.shape != (self.program.num_columns,):
            raise ValueError(
                f"A primal point of shape {tuple(primal.shape)} does not match the "
                f"{self.program.num_columns} reduced columns.",
            )
        return torch.from_numpy(
            core.postsolve_primal(
                self.state,
                _kernel_array(primal.cpu(), torch.float64),
                self.num_columns,
            ),
        )


def presolve(program: LinearProgram, device: torch.device) -> Presolved:
    """Reduce ``program`` with PSLP's default reductions, as cuOpt does.

    Args:
      program: The program to reduce, on the CPU.
      device: Device for the whole-matrix sorts.

    Returns:
      presolved: The reduced program and the state that postsolves it.

    """
    state = new_state(program, device=device)
    for _ in presolve_steps(state, device=device):
        pass
    ap, ai, ax, lhs, rhs, c, lb, ub, offset = core.reduced_arrays(state)
    return Presolved(
        program=LinearProgram(
            crow_indices=torch.from_numpy(ap),
            col_indices=torch.from_numpy(ai),
            values=torch.from_numpy(ax),
            num_columns=len(c),
            row_lower=torch.from_numpy(lhs),
            row_upper=torch.from_numpy(rhs),
            objective=torch.from_numpy(c),
            lower=torch.from_numpy(lb),
            upper=torch.from_numpy(ub),
            objective_offset=program.objective_offset + offset,
        ),
        state=state,
        num_columns=program.num_columns,
    )


def new_state(program: LinearProgram, device: torch.device) -> core.Core:
    """Build PSLP's presolver state for ``program``: A, its transpose, tags and lists.

    Args:
      program: The program to reduce, on the CPU.
      device: Device for the transpose's sort.

    Returns:
      state: The state ``presolve_steps`` reduces.

    Raises:
      ValueError: If the transpose's slots overflow PSLP's 32-bit positions,
        two per nonzero and four per row or column.

    """
    nonzeros = len(program.values)
    if 2 * nonzeros + 4 * max(program.num_rows, program.num_columns) > 2**31 - 1:
        raise ValueError(
            f"A program of {program.num_rows} rows, {program.num_columns} columns and "
            f"{nonzeros} nonzeros overflows PSLP's 32-bit entry positions.",
        )
    # One dtype and layout per argument, so each kernel compiles once: Numba 0.67
    # segfaults loading a cached specialization that was compiled in a process
    # where another specialization of the same kernel came from the cache.
    # 32-bit indices hold every program the guard above admits.
    ints, floats = torch.int32, torch.float64
    state = core.new_core(
        (
            _kernel_array(program.values, floats),
            _kernel_array(program.col_indices, ints),
            _kernel_array(program.crow_indices, ints),
        ),
        program.num_columns,
        (
            _kernel_array(program.row_lower, floats),
            _kernel_array(program.row_upper, floats),
        ),
        (_kernel_array(program.lower, floats), _kernel_array(program.upper, floats)),
        _kernel_array(program.objective, floats),
    )
    _, cols, starts, ends, n_cols = core.a_arrays(state)
    core.attach_transpose(
        state,
        *bulk.transpose_slots(cols, starts, ends, n_cols, device),
    )
    return state


def presolve_steps(state: core.Core, device: torch.device) -> Iterator[None]:
    """Run PSLP's main loop, yielding after each step PSLP's state dumps mark.

    Phases alternate as PSLP's do: fast explorers until a phase removes under
    5% of the nonzeros, then one medium phase; presolve ends when a medium
    cycle removes under 5%.

    Args:
      state: State from ``new_state``, reduced in place.
      device: Device for the parallel-row and parallel-column sorts.

    Yields:
      step: None, after each step; the last follows ``problem_clean``.

    """
    before_cycle = after_cycle = core.nnz(state)
    medium = False
    while True:
        yield from _trivial(state)
        before_phase = core.nnz(state)
        if medium:
            yield from _medium(state, device)
            after_cycle = core.nnz(state)
        else:
            yield from _fast(state)
        after_phase = core.nnz(state)
        if medium and after_cycle >= 0.95 * before_cycle:
            break
        if medium:
            before_cycle = after_cycle
            medium = False
        else:
            medium = after_phase >= 0.95 * before_phase
    core.problem_clean(state)
    yield


def _trivial(state: core.Core) -> Iterator[None]:
    """run_trivial_explorers."""
    core.remove_variables_with_close_bounds(state)
    yield
    core.remove_empty_cols(state)
    core.simple_dual_fix(state)
    yield
    while True:
        status = core.remove_ston_rows(state)
        yield
        if status != core.REDUCED:
            break
    core.remove_empty_rows(state)
    yield
    core.remove_empty_cols(state)
    yield


def _fast(state: core.Core) -> Iterator[None]:
    """run_fast_explorers: singleton columns, then doubleton equalities."""
    core.remove_ston_cols(state)
    yield
    yield from _trivial(state)
    while core.dton_pending(state):
        status = core.remove_dton_eq_rows_pass(state, 10)
        yield
        if status != core.REDUCED:
            break
    # PSLP drops the columns its doubleton pass fixed here; the pass fixes none.
    yield
    yield from _trivial(state)


def _medium(state: core.Core, device: torch.device) -> Iterator[None]:
    """run_medium_explorers: propagation, parallel rows, parallel columns."""
    core.check_activities(state)
    yield
    core.propagate_primal(state)
    yield
    yield from _trivial(state)
    core.remove_parallel_rows(
        state,
        bulk.sort_rows(*core.parallel_hashes(state, False), device),
    )
    yield
    core.remove_parallel_cols(
        state,
        bulk.sort_rows(*core.parallel_hashes(state, True), device),
    )
    yield
    yield from _trivial(state)


def _kernel_array(tensor: Tensor, dtype: torch.dtype) -> np.ndarray:
    """Return ``tensor`` as a C-contiguous CPU array of ``dtype``, copying only if needed."""
    return tensor.to(dtype).contiguous().numpy()
