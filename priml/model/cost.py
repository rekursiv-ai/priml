"""Analytical per-token cost of a model, read from its config.

Work is attributed to five kernel silos, one per roofline regime:

- ``matmul``: mm, bmm, conv, sdpa. Tensor-core work; the MFU numerator and
  what ``torch.utils.flop_counter`` counts.
- ``elementwise``: one output per input element.
- ``reduction``: many inputs to one output; a scan is a reduction that keeps
  its prefixes.
- ``selection``: gather in the primal, scatter-add in the adjoint.
- ``sort``: argsort, top-k; operand traffic depends on input/output geometry.

The silo follows the kernel dispatched, not the algebra: ``sum(x)`` is a
reduction, ``ones @ x`` is a matmul. Softmax is elementwise, reduction,
elementwise.

Counting policy:

- A multiply-accumulate is two FLOPs; any other floating op is one. A
  nonempty reduction over n is n-1. A gather is zero FLOPs and reads/writes
  its selected values; scatter-add adds once per element and reads/writes
  the destination.
- Analytical training algorithm with saved primal values, not a fused
  kernel. The adjoint includes local derivatives and parameter-gradient
  reductions; not the optimizer.
- ``rows`` is the rows sharing one parameter. Its gradient is summed
  over them, (n-1)/n per row, in ``adjoint.flops.reduction``. Only the two
  primitives write that term.
- Attention is counted over ``min(window, seq_len)`` keys with no causal
  discount (PaLM convention). Recompute excluded: MFU, not HFU.
- ``bytes`` and ``bytes_state`` hold bytes, not element counts. Each leaf
  prices its tensors at its own storage dtype (``None`` is torch's default)
  and tags every cell with it, so traffic can be read per dtype.
- Traffic is the analytical unfused algorithm's minimum tensor operand I/O:
  each primitive reads its inputs and writes its outputs once. Intermediates
  between primitives count even when a fused implementation keeps them on chip.
  This does not predict HBM traffic, cache reuse, or physical memory transactions.
  Traffic geometry is explicit and never inferred from FLOP counts. Nonlinear
  tensor operators move operands once, not once per internal scalar operation.

``cost(*, seq_len, batch_size, dtype, **kwargs)`` takes the batch shape and
dtype as required keywords -- no defaults, so a caller that forgets one fails
here rather than pricing a guessed batch -- and an open message bus. A leaf
reads its own config for everything else and forwards the bus unchanged; a
container that shares a child's weights over rows other than the batch's
tokens sends ``rows=`` on the bus, which :func:`shared_rows` reads.

:func:`matmul_cost` (``weight=False`` for ``QK^T``) and
:func:`elementwise_cost` are the two primitives that own parameters. The other
silos are ``Cost`` literals. A module-level function assigned as a class
attribute (``cost = attention_kernel_cost``) binds as the method.

A :class:`Cost` is a sparse table over ``measure x phase x kernel x dtype``,
the measures being ``flops`` and ``bytes``, plus three owned integers. The phases are ``primal`` and
``adjoint`` (VJP) today; a ``tangent`` (JVP) or ``hessian_vector`` is one more
phase and one line in each operator when a consumer needs it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Final,
    Literal,
    Protocol,
    overload,
    override,
    runtime_checkable,
)

import math

import torch


__all__ = [
    "KERNELS",
    "MEASURES",
    "PHASES",
    "Cost",
    "HasCost",
    "Index",
    "Kernel",
    "Key",
    "Measure",
    "Phase",
    "cost",
    "elementwise_cost",
    "matmul_cost",
    "mbu",
    "mfu",
    "reduction_cost",
    "resolve_dtype",
    "shared_rows",
    "traffic",
    "utilization",
    "with_rows",
]


type Measure = Literal["flops", "bytes"]
type Phase = Literal["primal", "adjoint"]
type Kernel = Literal["matmul", "elementwise", "reduction", "selection", "sort"]
type Key = tuple[Measure, Phase, Kernel, torch.dtype]

type Axis = str | torch.dtype
type _Part = Axis | slice
# A sub-table index: fewer than four parts, or four with at least one ``:``. A
# full ``Key`` is none of these, so the two ``__getitem__`` overloads are disjoint.
type Index = (
    Axis
    | tuple[_Part]
    | tuple[_Part, _Part]
    | tuple[_Part, _Part, _Part]
    | tuple[slice, _Part, _Part, _Part]
    | tuple[_Part, slice, _Part, _Part]
    | tuple[_Part, _Part, slice, _Part]
    | tuple[_Part, _Part, _Part, slice]
)

MEASURES: Final = ("flops", "bytes")
PHASES: Final = ("primal", "adjoint")
KERNELS: Final = ("matmul", "elementwise", "reduction", "selection", "sort")


@dataclass(frozen=True, slots=True, kw_only=True)
class Cost:
    """Per-token cost of one module: a sparse table plus what it owns.

    ``cells`` is a ``measure x phase x kernel x dtype`` grid of floats; a
    cell never written is zero. Index with a full key for the float, or a
    prefix -- leading axes, ``:`` to skip one -- for the sub-table, then
    ``sum()`` it: ``cost["flops", "primal", "matmul"].sum()`` is the forward
    matmul work, ``cost["bytes", :, :, torch.int64].sum()`` the index traffic.
    Fixed leading axes are dropped from the sub-table's keys, so
    ``cost["flops"] / cost["bytes"]`` lines up cell for cell; ``/`` follows
    the IEEE conventions of :func:`_div`.

    The three integers are not per cell: a slice is cells alone and owns
    nothing; they add under ``+`` and scale only by ``copies`` in :meth:`tile`.
    """

    cells: Mapping[tuple[object, ...], float] = field(
        default_factory=dict[tuple[object, ...], float],
    )
    """Nonzero cells; zeros are dropped on construction."""

    params: int = 0
    """Parameters the module owns."""

    params_active: int = 0
    """Parameters one token reads; fewer than ``params`` only when routed."""

    bytes_state: int = 0
    """Bytes of per-token state carried across a decode step."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cells",
            MappingProxyType(
                {
                    key: value
                    for key, value in self.cells.items()
                    if value != 0 or math.isnan(value)
                },
            ),
        )

    @overload
    def __getitem__(self, index: Key) -> float: ...
    @overload
    def __getitem__(self, index: Index) -> Cost: ...
    def __getitem__(self, index: Key | Index) -> float | Cost:
        parts: tuple[_Part, ...] = index if isinstance(index, tuple) else (index,)
        width = len(next(iter(self.cells))) if self.cells else 4
        if len(parts) == width and not any(isinstance(p, slice) for p in parts):
            return self.cells.get(parts, 0)
        keep = [i for i, p in enumerate(parts) if isinstance(p, slice)]
        keep += range(len(parts), width)
        return Cost(
            cells={
                tuple(key[i] for i in keep): value
                for key, value in self.cells.items()
                if all(
                    isinstance(want, slice) or want == have
                    for want, have in zip(parts, key, strict=False)
                )
            },
        )

    def sum(self) -> float:
        """Total over every cell."""
        return math.fsum(self.cells.values())

    def __add__(self, other: Cost) -> Cost:
        merged = dict(self.cells)
        for key, value in other.cells.items():
            merged[key] = merged.get(key, 0) + value
        return Cost(
            cells=merged,
            params=self.params + other.params,
            params_active=self.params_active + other.params_active,
            bytes_state=self.bytes_state + other.bytes_state,
        )

    def __truediv__(self, other: float | Cost) -> Cost:
        """Divide cell-wise; the quotient is a ratio table and owns nothing."""
        if isinstance(other, Cost):
            keys = self.cells.keys() | other.cells.keys()
            return Cost(
                cells={
                    key: _div(self.cells.get(key, 0), other.cells.get(key, 0))
                    for key in keys
                },
            )
        return Cost(
            cells={key: _div(value, other) for key, value in self.cells.items()},
        )

    def tile(self, rows: float, *, copies: int = 1) -> Cost:
        """Run ``rows`` times per token, owned ``copies`` times.

        A norm over every head row of one shared weight runs ``rows`` times
        but is owned once; a stack of ``n`` blocks is ``tile(n, copies=n)``; a
        child priced per its own row spread over a container's rows is
        ``tile(child_rows / container_rows)``.

        Args:
          rows: Multiplier on every cell.
          copies: Times the parameters exist.

        Returns:
          tiled: Cells scaled by ``rows``; ownership by ``copies``;
            ``bytes_state`` by ``rows``, since state is per row.

        """
        return Cost(
            cells={key: value * rows for key, value in self.cells.items()},
            params=copies * self.params,
            params_active=copies * self.params_active,
            bytes_state=int(self.bytes_state * rows),
        )

    @override
    def __hash__(self) -> int:
        return hash(
            (
                frozenset(self.cells.items()),
                self.params,
                self.params_active,
                self.bytes_state,
            ),
        )

    @override
    def __repr__(self) -> str:
        return (
            f"Cost(params={self.params}, params_active={self.params_active}, "
            f"bytes_state={self.bytes_state})\n{_grid(self.cells)}"
        )


def traffic(
    phase: Phase,
    kernel: Kernel,
    *,
    elements: float,
    dtype: torch.dtype | None = None,
    flops: float = 0,
) -> Cost:
    """One cell of tensor I/O, and optionally its FLOPs, owning nothing.

    Args:
      phase: Pass the traffic belongs to.
      kernel: Silo the kernel dispatches to.
      elements: Elements moved; bytes are ``elements * dtype.itemsize``.
      dtype: Element type; ``None`` is torch's default.
      flops: Operations in the same cell, when the literal carries both.

    Returns:
      cost: The one-cell ledger.

    """
    dt = resolve_dtype(dtype)
    return Cost(
        cells={
            ("flops", phase, kernel, dt): flops,
            ("bytes", phase, kernel, dt): elements * dt.itemsize,
        },
    )


def resolve_dtype(dtype: torch.dtype | None) -> torch.dtype:
    """Return ``dtype``, or torch's default when a config left it ``None``."""
    return torch.get_default_dtype() if dtype is None else dtype


def shared_rows(seq_len: int, batch_size: int, **kwargs: object) -> float:
    """Rows sharing one parameter: ``rows`` on the bus, else the batch's tokens.

    Args:
      seq_len: Tokens per sequence.
      batch_size: Sequences per step.
      **kwargs: The bus; ``rows`` overrides when a container shares a child's
        weights over something other than the batch's tokens.

    Returns:
      rows: Finite and at least one.

    Raises:
      TypeError: ``rows`` is on the bus but is not a number.

    """
    if seq_len < 1 or batch_size < 1:
        raise ValueError("seq_len and batch_size must be at least one.")
    rows = kwargs.get("rows")
    if rows is None:
        return seq_len * batch_size
    if isinstance(rows, bool) or not isinstance(rows, (int, float)):
        raise TypeError(f"rows must be a number; got {rows!r}.")
    _validate_rows(rows)
    return rows


def with_rows(rows: float, /, **kwargs: object) -> dict[str, object]:
    """Return the bus with ``rows`` replaced, for a child's own sharing rows.

    ``rows`` is positional-only so a bus already carrying one passes through
    ``**kwargs`` and is overwritten rather than colliding.

    Args:
      rows: The child's sharing rows.
      **kwargs: The bus, with or without a ``rows`` of its own.

    Returns:
      bus: ``kwargs`` with ``rows`` set.

    """
    return {**kwargs, "rows": rows}


@runtime_checkable
class HasCost(Protocol):
    """A config that prices the module it builds."""

    def cost(
        self,
        *,
        seq_len: int,
        batch_size: int,
        dtype: torch.dtype | None,
        **kwargs: object,
    ) -> Cost:
        """Price one token through the module ``self`` builds.

        Args:
          seq_len: Tokens per sequence: attention's reach before any window.
          batch_size: Sequences per step; amortizes weights only.
          dtype: Activation dtype; ``None`` is torch's default.
          **kwargs: The open bus, forwarded to every child.

        Returns:
          cost: Per-token cost.

        """
        ...


def cost(
    config: object,
    *,
    seq_len: int,
    batch_size: int,
    dtype: torch.dtype | None,
    **kwargs: object,
) -> Cost:
    """Price a config per token for ``batch_size`` sequences of ``seq_len`` tokens.

    Args:
      config: A config with ``cost``.
      seq_len: Tokens per sequence.
      batch_size: Sequences per step.
      dtype: Activation dtype; ``None`` is torch's default.
      **kwargs: The open bus, forwarded unchanged.

    Returns:
      cost: The config's per-token cost.

    Raises:
      TypeError: ``config`` has no ``cost``.

    """
    if not isinstance(config, HasCost):
        raise TypeError(
            f"{type(config).__qualname__} has no cost(); every config under a "
            "priced container must implement HasCost.",
        )
    return config.cost(seq_len=seq_len, batch_size=batch_size, dtype=dtype, **kwargs)


def matmul_cost(
    *,
    channels_in: int,
    channels_out: int,
    bias: bool = False,
    weight: bool = True,
    rows: float = 1,
    dtype: torch.dtype | None = None,
) -> Cost:
    """Price one row of ``[M, K] @ [K, N]`` and its two adjoint products.

    Args:
      channels_in: Inner dimension K.
      channels_out: Output width N.
      bias: Add a separate bias map and its gradient reduction.
      weight: Own the right matrix as parameters; False keeps its activation
        traffic but owns no matrix parameters.
      rows: Rows M sharing the right matrix and bias, or an analytical
        average of at least one. For attention, use rows sharing one sequence's
        matrix, not rows across the batch.
      dtype: Element type of every operand, gradients included; ``None`` is
        torch's default. Tags every cell and sets the bytes per element.

    Returns:
      cost: Per-row FLOPs and unfused tensor I/O. The primal moves
        ``itemsize * (K + N + K*N/M)`` bytes; each adjoint product moves the
        same amount. Bias traffic belongs to elementwise and reduction silos.

    Raises:
      ValueError: ``rows`` is nonfinite or below one.

    """
    _validate_rows(rows)
    dt = resolve_dtype(dtype)
    s = dt.itemsize
    products = channels_in * channels_out
    biases = channels_out if bias else 0
    params = (products if weight else 0) + biases
    moved = s * (channels_in + channels_out + products / rows)
    return Cost(
        cells={
            ("flops", "primal", "matmul", dt): 2 * products,
            ("flops", "primal", "elementwise", dt): biases,
            ("flops", "adjoint", "matmul", dt): 4 * products,
            ("flops", "adjoint", "reduction", dt): biases * (rows - 1) / rows,
            ("bytes", "primal", "matmul", dt): moved,
            ("bytes", "primal", "elementwise", dt): s * (2 * biases + biases / rows),
            ("bytes", "adjoint", "matmul", dt): 2 * moved,
            ("bytes", "adjoint", "reduction", dt): s * (biases + biases / rows),
        },
        params=params,
        params_active=params,
    )


def elementwise_cost(
    *,
    primal: float,
    adjoint: float,
    channels: float = 0,
    params: int = 0,
    rows: float = 1,
    dtype: torch.dtype | None = None,
    inputs: int = 1,
    outputs: int = 1,
    adjoint_inputs: int = 2,
    adjoint_outputs: int = 1,
) -> Cost:
    """Price explicit elementwise operands and owned parameter gradients.

    The default geometry is a unary map: primal input/output and adjoint
    saved value/incoming gradient/outgoing gradient. Compound maps specify
    summed operand counts explicitly; FLOPs never determine traffic.

    Args:
      primal: Operations per token evaluating the map.
      adjoint: Backward operations excluding parameter-gradient reductions.
      channels: Elements per operand row; may be amortized across tokens.
      params: Owned parameters, read once per pass across ``rows`` rows.
        Adjoint elementwise traffic includes one temporary gradient write per
        parameter per row; reduction then reads these and writes the result.
      rows: Rows sharing parameters and their gradient reduction.
      dtype: Element type of every operand, gradients included; ``None`` is
        torch's default.
      inputs: Primal input operands, excluding owned parameters.
      outputs: Primal output operands.
      adjoint_inputs: Adjoint input operands, excluding owned parameters.
      adjoint_outputs: Adjoint outputs, excluding parameter-gradient temporaries.

    Returns:
      cost: Explicit operand I/O with parameter reductions in the adjoint.

    Raises:
      ValueError: ``rows`` is nonfinite or below one.

    """
    _validate_rows(rows)
    dt = resolve_dtype(dtype)
    s = dt.itemsize
    return Cost(
        cells={
            ("flops", "primal", "elementwise", dt): primal,
            ("flops", "adjoint", "elementwise", dt): adjoint,
            ("flops", "adjoint", "reduction", dt): params * (rows - 1) / rows,
            ("bytes", "primal", "elementwise", dt): s
            * (channels * (inputs + outputs) + params / rows),
            ("bytes", "adjoint", "elementwise", dt): s
            * (channels * (adjoint_inputs + adjoint_outputs) + params / rows + params),
            ("bytes", "adjoint", "reduction", dt): s * (params + params / rows),
        },
        params=params,
        params_active=params,
    )


def reduction_cost(
    *,
    input_elements: float,
    output_groups: float = 1,
    rows: float = 1,
    dtype: torch.dtype | None = None,
    phase: Phase = "primal",
) -> Cost:
    """Price one reduction's explicit tensor geometry, amortized over tokens.

    Args:
      input_elements: Total elements read across all output groups.
      output_groups: Reduced elements written; each group uses n-1 operations.
      rows: Tokens sharing this reduction's work and traffic.
      dtype: Element type of the input and output; ``None`` is torch's default.
      phase: Which pass runs the reduction.

    Returns:
      cost: Reduction FLOPs and minimum unfused operand I/O, owning nothing.
        Singleton groups copy their elements; empty groups write the identity.
        Both use zero FLOPs. This describes one reduction, not its derivative.

    Raises:
      ValueError: ``rows`` is nonfinite or below one.

    """
    _validate_rows(rows)
    dt = resolve_dtype(dtype)
    return Cost(
        cells={
            ("flops", phase, "reduction", dt): max(
                0,
                input_elements - output_groups,
            )
            / rows,
            ("bytes", phase, "reduction", dt): dt.itemsize
            * (input_elements + output_groups)
            / rows,
        },
    )


def utilization(cost: Cost, *, tokens_per_sec: float, peak: Cost) -> Cost:
    """Achieved fraction of each cell's ceiling on a training step.

    Args:
      cost: Per-token cost at the batch's sequence length.
      tokens_per_sec: Measured ``tokens_per_step / step_time``.
      peak: Per-cell FLOP/s ceiling summed over every device the tokens
        spanned, at ``(phase, kernel, dtype)`` keys and owning nothing; a
        matmul cell is the dense tensor-core peak for its dtype.

    Returns:
      achieved: Fraction of peak per cell; the matmul cells are MFU.

    """
    return cost["flops"].tile(tokens_per_sec) / peak


def mfu(cost: Cost, *, tokens_per_sec: float, peak_flops_per_sec: float) -> float:
    """Model FLOPs utilization: the ``matmul`` silo of :func:`utilization`.

    Args:
      cost: Per-token cost at the batch's sequence length.
      tokens_per_sec: Measured ``tokens_per_step / step_time``.
      peak_flops_per_sec: Dense datasheet FLOP/s for the dtype, summed over
        every device the tokens spanned.

    Returns:
      achieved: Fraction of peak; ``0.4`` is a well-tuned dense run.

    """
    return cost["flops", :, "matmul"].sum() * tokens_per_sec / peak_flops_per_sec


def mbu(
    cost: Cost,
    *,
    batch: int,
    context_len: int,
    steps_per_sec: float,
    itemsize: int,
    peak_bytes_per_sec: float,
) -> float:
    """Model bandwidth utilization of a decode step.

    Every active weight is read once per step; each sequence's state is read
    up to its position. Prefill is compute-bound: use :func:`utilization`.

    Args:
      cost: Per-token cost; ``bytes_state`` is per token kept.
      batch: Sequences decoded per step.
      context_len: Positions each sequence's state holds.
      steps_per_sec: Measured decode step rate.
      itemsize: Bytes per weight element; state is already counted in bytes.
      peak_bytes_per_sec: Datasheet HBM bandwidth.

    Returns:
      achieved: Fraction of peak bandwidth.

    """
    moved = cost.params_active * itemsize + batch * context_len * cost.bytes_state
    return moved * steps_per_sec / peak_bytes_per_sec


def _validate_rows(rows: float) -> None:
    if not math.isfinite(rows) or rows < 1:
        raise ValueError("rows must be finite and at least one.")


# Columns are the last key axis (``dtype`` unless it was sliced away), crossed with
# ``measure`` when a table still holds one; rows are every axis between. A table
# holding both measures ends each row with its flops/bytes.
def _grid(cells: Mapping[tuple[object, ...], float]) -> str:
    """Render a table as an aligned grid with a totals row."""
    if not cells:
        return "(empty)"
    width = len(next(iter(cells)))
    if width == 0:
        return _si(next(iter(cells.values())))
    measured = any(k[0] in MEASURES for k in cells) and width > 1
    measures: list[str] = (
        [m for m in MEASURES if any(k[0] == m for k in cells)] if measured else [""]
    )
    columns_axis = sorted({k[-1] for k in cells}, key=_column_order)
    order = (*PHASES, *KERNELS)
    labels = sorted(
        {tuple(str(x) for x in k[measured:-1]) for k in cells},
        key=lambda label: tuple(order.index(x) if x in order else -1 for x in label),
    )
    depth = max(1, *(len(label) for label in labels))
    columns = [
        f"{m}[{_axis_name(d)}]" if m else _axis_name(d)
        for m in measures
        for d in columns_axis
    ]
    per_column = len(columns_axis) if measures == ["flops", "bytes"] else 0
    if per_column:
        columns.append("flops/bytes")
    rows: list[list[str]] = [[*[""] * depth, *columns]]
    totals = [0.0] * (len(measures) * len(columns_axis))
    for label in labels:
        values = [
            cells.get(((m,) if measured else ()) + label + (d,), 0.0)
            for m in measures
            for d in columns_axis
        ]
        totals = [t + v for t, v in zip(totals, values, strict=True)]
        rows.append(_line(label, values, depth=depth, per_column=per_column))
    if len(labels) > 1:
        rows.append(_line(("total",), totals, depth=depth, per_column=per_column))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        " ".join(
            cell.ljust(widths[i]) if i < depth else cell.rjust(widths[i])
            for i, cell in enumerate(row)
        ).rstrip()
        for row in rows
    )


def _column_order(axis: object) -> tuple[int, str]:
    if isinstance(axis, torch.dtype):
        return (axis.itemsize, str(axis))
    order = (*MEASURES, *PHASES, *KERNELS)
    return (order.index(axis) if axis in order else -1, str(axis))


def _line(
    label: tuple[str, ...],
    values: list[float],
    *,
    depth: int,
    per_column: int,
) -> list[str]:
    """Format one grid row; ``per_column > 0`` appends the row's flops/bytes."""
    out = [*label, *[""] * (depth - len(label)), *(_si(v) for v in values)]
    if per_column:
        out.append(
            _si(_div(math.fsum(values[:per_column]), math.fsum(values[per_column:]))),
        )
    return out


def _axis_name(axis: object) -> str:
    return str(axis).removeprefix("torch.")


def _si(value: float) -> str:
    """Format with an SI suffix at four significant digits; zero is ``-``."""
    if value == 0:
        return "-"
    if not math.isfinite(value):
        return str(value)
    magnitude = abs(value)
    for exponent, suffix in ((12, "T"), (9, "G"), (6, "M"), (3, "K")):
        if magnitude >= 10**exponent:
            return f"{value / 10**exponent:.4g}{suffix}"
    return f"{value:.4g}"


def _div(numerator: float, denominator: float) -> float:
    if denominator:
        return numerator / denominator
    if numerator == 0 or math.isnan(numerator):
        return math.nan
    return math.copysign(math.inf, numerator) * math.copysign(1, denominator)
