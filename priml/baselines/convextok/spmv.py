"""Sparse matrix-vector products in CSR form, deterministic on every device.

Each row's products are summed in a fixed order, never by atomics, so repeated products
return the same bits. On the CPU a segmented reduction does this. On CUDA, rows are
grouped by length once; each group of up to 32 entries per row is one fused
gather-multiply-sum kernel, and longer rows are cut into fixed chunks whose partial
sums are reduced by the same plan, level by level.
"""

# No ``from __future__ import annotations``: Triton reads the kernels' source, and
# their annotations must stay the quoted strings it knows how to skip.

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from types import FunctionType
from typing import TYPE_CHECKING, Final

from torch import Tensor

import torch


if TYPE_CHECKING:
    from triton import language

    import triton
else:
    from wrapt import lazy_import

    triton = lazy_import("triton")
    language = lazy_import("triton.language")


_WIDTHS: Final = (1, 2, 4, 8, 16, 32)
_CHUNK: Final = 256


class CsrMatrix:
    """A CSR matrix whose product with a vector is deterministic and, on CUDA, fast.

    Attributes:
      crow_indices: Row start offsets, length rows + 1.
      col_indices: Column of each stored entry.
      values: Value of each stored entry.

    """

    def __init__(
        self,
        crow_indices: Tensor,
        col_indices: Tensor,
        values: Tensor,
    ) -> None:
        self.crow_indices = crow_indices.to(torch.int64)
        self.col_indices = col_indices
        self.values = values
        self.num_rows = len(crow_indices) - 1
        self._plan = _plan(self.crow_indices) if values.is_cuda else None

    def __matmul__(self, vector: Tensor) -> Tensor:
        """Multiply by ``vector``, one entry per column; return one entry per row."""
        if self._plan is None:
            return torch.segment_reduce(
                self.values * vector[self.col_indices],
                "sum",
                offsets=self.crow_indices,
            )
        out = torch.empty(self.num_rows, dtype=vector.dtype, device=vector.device)
        _run(self._plan, self.col_indices, self.values, vector, out)
        return out


@dataclass(frozen=True, slots=True, kw_only=True)
class _Bucket:
    """Rows of up to ``width`` entries, and where each row's sum is written."""

    width: int
    rows: Tensor
    targets: Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class _Plan:
    """How to sum every row of one CSR structure in a fixed order."""

    crow: Tensor
    buckets: list[_Bucket]
    chunk_start: Tensor
    chunk_end: Tensor
    partial_columns: Tensor
    partial_plan: "_Plan | None"


def _plan(crow: Tensor, targets: Tensor | None = None) -> _Plan:
    """Group rows by length; cut rows longer than the widest group into chunks."""
    lengths = torch.diff(crow)
    rows = torch.arange(len(lengths), device=crow.device)
    if targets is None:
        targets = rows
    buckets: list[_Bucket] = []
    for index, width in enumerate(_WIDTHS):
        lower_bound = lengths >= 0 if index == 0 else lengths > _WIDTHS[index - 1]
        members = rows[lower_bound & (lengths <= width)]
        if len(members):
            buckets.append(_Bucket(width=width, rows=members, targets=targets[members]))
    long_rows = rows[lengths > _WIDTHS[-1]]
    if not len(long_rows):
        empty = torch.zeros(0, dtype=torch.int64, device=crow.device)
        return _Plan(
            crow=crow,
            buckets=buckets,
            chunk_start=empty,
            chunk_end=empty,
            partial_columns=empty,
            partial_plan=None,
        )
    chunks_per_row = (lengths[long_rows] + _CHUNK - 1) // _CHUNK
    chunk_row = torch.repeat_interleave(long_rows, chunks_per_row)
    first_chunk = torch.cumsum(chunks_per_row, 0) - chunks_per_row
    within = torch.arange(len(chunk_row), device=crow.device) - torch.repeat_interleave(
        first_chunk,
        chunks_per_row,
    )
    chunk_start = crow[chunk_row] + within * _CHUNK
    chunk_end = torch.minimum(chunk_start + _CHUNK, crow[chunk_row + 1])
    partial_crow = torch.zeros(
        len(long_rows) + 1,
        dtype=torch.int64,
        device=crow.device,
    )
    partial_crow[1:] = torch.cumsum(chunks_per_row, 0)
    return _Plan(
        crow=crow,
        buckets=buckets,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        partial_columns=torch.zeros(
            len(chunk_row),
            dtype=torch.int64,
            device=crow.device,
        ),
        # The partial sums are a CSR of their own, one row per long row, written
        # where the long rows belong.
        partial_plan=_plan(partial_crow, targets[long_rows]),
    )


def _run(
    plan: _Plan,
    columns: Tensor,
    values: Tensor,
    vector: Tensor,
    out: Tensor,
) -> None:
    """Write every row sum of ``plan`` over ``values * vector[columns]`` into ``out``."""
    for bucket in plan.buckets:
        # About 1,024 entries per program, whatever the width.
        block = 1024 // bucket.width
        _bucket_kernel()[(triton.cdiv(len(bucket.rows), block),)](
            (bucket.rows, bucket.targets, plan.crow, columns, values, vector, out),
            len(bucket.rows),
            bucket.width,
            block,
        )
    if plan.partial_plan is None:
        return
    partials = torch.empty(len(plan.chunk_start), dtype=out.dtype, device=out.device)
    _chunk_kernel()[(len(plan.chunk_start),)](
        (plan.chunk_start, plan.chunk_end, columns, values, vector, partials),
        _CHUNK,
    )
    ones = torch.ones(1, dtype=out.dtype, device=out.device)
    _run(plan.partial_plan, plan.partial_columns, partials, ones, out)


def _bucket_sum(
    buffers: "tuple[language.tensor, ...]",
    count: int,
    width: "language.constexpr",
    block: "language.constexpr",
) -> None:
    """Sum ``block`` rows of at most ``width`` entries each, one row per lane group."""
    rows_ptr, targets_ptr, crow_ptr, col_ptr, value_ptr, x_ptr, out_ptr = buffers
    position = language.program_id(0) * block + language.arange(0, block)
    live = position < count
    row = language.load(rows_ptr + position, mask=live, other=0)
    start = language.load(crow_ptr + row, mask=live, other=0)
    end = language.load(crow_ptr + row + 1, mask=live, other=0)
    index = start[:, None] + language.arange(0, width)[None, :]
    mask = live[:, None] & (index < end[:, None])
    column = language.load(col_ptr + index, mask=mask, other=0)
    value = language.load(value_ptr + index, mask=mask, other=0.0)
    x = language.load(x_ptr + column, mask=mask, other=0.0)
    target = language.load(targets_ptr + position, mask=live, other=0)
    language.store(out_ptr + target, language.sum(value * x, axis=1), mask=live)


def _chunk_sum(
    buffers: "tuple[language.tensor, ...]",
    chunk: "language.constexpr",
) -> None:
    """Sum one fixed-size chunk of a long row."""
    start_ptr, end_ptr, col_ptr, value_ptr, x_ptr, partial_ptr = buffers
    program = language.program_id(0)
    start = language.load(start_ptr + program)
    end = language.load(end_ptr + program)
    index = start + language.arange(0, chunk)
    mask = index < end
    column = language.load(col_ptr + index, mask=mask, other=0)
    value = language.load(value_ptr + index, mask=mask, other=0.0)
    x = language.load(x_ptr + column, mask=mask, other=0.0)
    language.store(partial_ptr + program, language.sum(value * x, axis=0))


@lru_cache(maxsize=1)
def _bucket_kernel() -> "triton.JITFunction[..., None]":
    return _jit(_bucket_sum)


@lru_cache(maxsize=1)
def _chunk_kernel() -> "triton.JITFunction[..., None]":
    return _jit(_chunk_sum)


def _jit(function: Callable[..., None]) -> "triton.JITFunction[..., None]":
    """Bind the real ``triton.language`` before Triton hashes and compiles ``function``."""
    assert isinstance(function, FunctionType)
    bound = FunctionType(
        function.__code__,
        function.__globals__ | {"language": import_module("triton.language")},
        argdefs=function.__defaults__,
    )
    bound.__annotations__ = function.__annotations__
    return triton.jit(bound)
