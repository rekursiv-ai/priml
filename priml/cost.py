"""Analytical whole-invocation cost of a model, read from its config.

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
- Primitive geometry is concrete: execution rows and tensor elements are
  counted for the complete invocation. Parameter gradients reduce over the
  rows sharing those parameters in ``adjoint.flops.reduction``.
- Dense masked attention counts full-sequence products. Local-window kernels
  count ``min(window + 1, seq_len)`` keys with no causal discount (PaLM
  convention); negative windows use the whole sequence. ``window`` is
  previous-key history, so zero includes self only. Recompute excluded:
  MFU, not HFU.
- ``bytes`` and ``bytes_state`` hold bytes, not element counts. Each leaf
  costs its tensors at its own storage dtype (``None`` is torch's default)
  and tags every cell with it, so traffic can be read per dtype.
- Traffic is the analytical unfused algorithm's minimum tensor operand I/O:
  each primitive reads its inputs and writes its outputs once. Intermediates
  between primitives count even when a fused implementation keeps them on chip.
  This does not predict HBM traffic, cache reuse, or physical memory transactions.
  Traffic geometry is explicit and never inferred from FLOP counts. Nonlinear
  tensor operators move operands once, not once per internal scalar operation.
  Convolution reads whole input/weight tensors and writes whole outputs;
  its joint backward counts only operands needed by requested gradients.
  Attention counts each sequence's complete query, key, and value tensors.
  These conventions are logical I/O, not a dispatched-operand or HBM prediction.

``cost(**kwargs)`` takes named arguments only. A leaf declares the ones it
reads -- typically ``seq_len``, ``batch_size``, ``dtype`` -- as required
keywords, so a caller that forgets one fails there rather than costing a
guessed batch, and forwards the rest of the bus unchanged. Every result is a
complete invocation; reporting derives rates and per-token views afterward.

:func:`matmul_cost` (``weight=False`` for ``QK^T``) and
:func:`elementwise_cost` are the two primitives that own parameters. The other
silos are ``Cost`` literals. A module-level function assigned as a class
attribute (``cost = attention_kernel_cost``) binds as the method. A plain
function receives cost metadata through :func:`set_cost`, and
:func:`map_cost` builds the per-element cost function an activation or schedule
transform needs.

A :class:`Cost` is a sparse table over ``measure x phase x kernel x dtype``,
with integer FLOP and byte counts for a concrete invocation, plus three owned
integers. :class:`Report` is the separate floating table for derived ratios,
ceilings, and utilization. The phases are ``primal`` and ``adjoint`` (VJP)
today; a ``tangent`` (JVP) or ``hessian_vector`` is one more phase and one line
in each operator when a consumer needs it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, cast, overload, override

import functools
import math

import torch


__all__ = [
    "KERNELS",
    "MEASURES",
    "PHASES",
    "Cost",
    "Device",
    "Index",
    "Kernel",
    "Key",
    "Measure",
    "Phase",
    "Report",
    "cost",
    "elementwise_cost",
    "intensity",
    "map_cost",
    "matmul_cost",
    "peak",
    "reduction_cost",
    "resolve_dtype",
    "set_cost",
    "traffic",
    "utilization",
]


type Measure = Literal["flops", "bytes"]
type Phase = Literal["primal", "adjoint"]
type Kernel = Literal["matmul", "elementwise", "reduction", "selection", "sort"]
type Axis = str | torch.dtype
type Key = tuple[Axis, Axis, Axis, Axis]
"""Every axis of a four-axis table named, in any order: reads one integer."""
type Index = Axis | tuple[Axis] | tuple[Axis, Axis] | tuple[Axis, Axis, Axis]
"""Fewer axes than the table has, in any order: reads the sub-table."""

MEASURES: Final = ("flops", "bytes")
_REPORT_MEASURES: Final = (*MEASURES, "intensity")
PHASES: Final = ("primal", "adjoint")
KERNELS: Final = ("matmul", "elementwise", "reduction", "selection", "sort")
DTYPE_ALIASES: Final[Mapping[str, torch.dtype]] = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
    "f64": torch.float64,
    "fp8e4m3": torch.float8_e4m3fn,
    "fp8e5m2": torch.float8_e5m2,
    "fp4": torch.float4_e2m1fn_x2,
}
"""Short dtype spellings an index accepts and a grid header prints."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Cost:
    """Whole-invocation cost of one module: a sparse table plus ownership.

    ``cells`` is a ``measure x phase x kernel x dtype`` grid of integer counts;
    a cell never written is zero. Index with axis values in any order -- each
    value names its own axis, since measures, phases, kernels and dtypes never
    collide -- and get the count when every axis is fixed, else the sub-table.
    A combination no cell holds is an empty table, i.e. zero; only a value that
    names no axis at all (``"gemm"``) is a ``KeyError``.

    The three ownership fields are not per cell: a slice owns nothing; they
    add under ``+`` and scale only by ``copies`` in :meth:`tile`.
    """

    cells: Mapping[tuple[object, ...], int] = field(
        default_factory=dict[tuple[object, ...], int],
    )
    """Nonzero integer invocation counts; zeros are dropped on construction."""

    params: int = 0
    """Parameters the module owns."""

    params_active: int = 0
    """Parameters this invocation reads; fewer than ``params`` when routed."""

    bytes_state: int = 0
    """Bytes of state carried to the next invocation."""

    def __post_init__(self) -> None:
        for key, value in self.cells.items():
            if "intensity" in key:
                raise ValueError("Cost cells cannot contain reporting intensity.")
            _validate_nonnegative_integer("cell", value)
        _validate_nonnegative_integer("params", self.params)
        _validate_nonnegative_integer("params_active", self.params_active)
        _validate_nonnegative_integer("bytes_state", self.bytes_state)
        object.__setattr__(
            self,
            "cells",
            MappingProxyType(
                {key: value for key, value in self.cells.items() if value != 0},
            ),
        )

    @overload
    def __getitem__(self, index: Key) -> int: ...
    @overload
    def __getitem__(self, index: Index) -> Cost: ...
    def __getitem__(self, index: Key | Index) -> int | Cost:
        parts = index if isinstance(index, tuple) else (index,)
        return self._select([_canonical(p) for p in parts])

    def _select(self, wanted: list[object]) -> int | Cost:
        if "intensity" in wanted:
            raise KeyError("'intensity' is a reporting-only measure")
        if not wanted:
            return self
        if not self.cells:
            return Cost() if len(wanted) < 4 else 0
        sample = next(iter(self.cells))
        fixed = {_axis_of(sample, want): want for want in wanted}
        if len(fixed) == len(sample):
            return self.cells.get(tuple(fixed[i] for i in range(len(sample))), 0)
        keep = [i for i in range(len(sample)) if i not in fixed]
        return Cost(
            cells={
                tuple(key[i] for i in keep): value
                for key, value in self.cells.items()
                if all(key[i] == want for i, want in fixed.items())
            },
        )

    def only(self, value: Axis) -> Cost:
        """Keep the cells at one axis value, the axis included, owning nothing.

        Where ``cost["primal"]`` drops the phase axis, ``cost.only("primal")``
        keeps it, so the result still adds to a full table: a layer whose
        primal runs three times and adjoint once is
        ``c + c.only("primal").tile(2)``.

        Args:
          value: A measure, phase, kernel, dtype or device the table's keys hold.

        Returns:
          kept: The matching cells at their original keys.

        Raises:
          KeyError: ``value`` names no axis.

        """
        want = _canonical(value)
        if not self.cells:
            _kind(want)
            return Cost()
        axis = _axis_of(next(iter(self.cells)), want)
        return Cost(
            cells={key: v for key, v in self.cells.items() if key[axis] == want},
        )

    def relabel(self, value: Axis) -> Cost:
        """Move every cell to ``value`` on its axis, owning nothing.

        A frozen layer's adjoint is its input gradient alone, the primal's
        size: ``c.only("primal") + c.only("primal").relabel("adjoint")``.

        Args:
          value: A measure, phase, kernel, dtype or device.

        Returns:
          moved: The cells with that axis rewritten; colliding cells add.

        Raises:
          KeyError: ``value`` names no axis.

        """
        want = _canonical(value)
        if not self.cells:
            _kind(want)
            return Cost()
        axis = _axis_of(next(iter(self.cells)), want)
        merged: dict[tuple[object, ...], int] = {}
        for key, cell_value in self.cells.items():
            moved = (*key[:axis], want, *key[axis + 1 :])
            merged[moved] = merged.get(moved, 0) + cell_value
        return Cost(cells=merged)

    def sum(self) -> int:
        """Total over every cell."""
        return sum(self.cells.values(), 0)

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

    def tile(self, repetitions: int, *, copies: int = 1) -> Cost:
        """Repeat an invocation an integer number of times.

        Execution repetitions and parameter copies are independent: a shared
        child may run repeatedly while its parameters remain owned once.

        Args:
          repetitions: Number of concrete executions represented by ``self``.
          copies: Number of parameter copies represented by ``self``.

        Returns:
          tiled: Counts and carried state scaled by ``repetitions``; ownership
            scaled by ``copies``.

        Raises:
          TypeError: A repetition or copy count is not an integer.
          ValueError: A repetition or copy count is negative.

        """
        _validate_integer("repetitions", repetitions)
        _validate_integer("copies", copies)
        if repetitions < 0 or copies < 0:
            raise ValueError("repetitions and copies must be nonnegative.")
        return Cost(
            cells={key: value * repetitions for key, value in self.cells.items()},
            params=copies * self.params,
            params_active=copies * self.params_active,
            bytes_state=self.bytes_state * repetitions,
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


@dataclass(frozen=True, slots=True, kw_only=True)
class Report:
    """Floating reporting table derived from concrete execution counts."""

    cells: Mapping[tuple[object, ...], float] = field(
        default_factory=dict[tuple[object, ...], float],
    )
    """Nonzero reporting values; zeros are dropped on construction."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cells",
            MappingProxyType(
                {
                    key: float(value)
                    for key, value in self.cells.items()
                    if value != 0 or math.isnan(value)
                },
            ),
        )

    @overload
    def __getitem__(self, index: Key) -> float: ...
    @overload
    def __getitem__(self, index: Index) -> Report: ...
    def __getitem__(self, index: Key | Index) -> float | Report:
        parts = index if isinstance(index, tuple) else (index,)
        return self._select([_canonical(part) for part in parts])

    def _select(self, wanted: list[object]) -> float | Report:
        if not wanted:
            return self
        if not self.cells:
            return Report() if len(wanted) < 3 else 0.0
        sample = next(iter(self.cells))
        fixed = {_axis_of(sample, want): want for want in wanted}
        if len(fixed) == len(sample):
            return self.cells.get(
                tuple(fixed[index] for index in range(len(sample))),
                0.0,
            )
        keep = [index for index in range(len(sample)) if index not in fixed]
        return Report(
            cells={
                tuple(key[index] for index in keep): value
                for key, value in self.cells.items()
                if all(key[index] == want for index, want in fixed.items())
            },
        )

    def sum(self) -> float:
        """Total over every reporting cell."""
        return sum(self.cells.values(), 0.0)

    @override
    def __repr__(self) -> str:
        return _grid(self.cells)


def traffic(
    phase: Phase,
    kernel: Kernel,
    *,
    elements: int,
    dtype: torch.dtype | None = None,
    flops: int = 0,
) -> Cost:
    """Record whole-invocation tensor I/O and optional FLOPs.

    Args:
      phase: Pass the traffic belongs to.
      kernel: Silo the kernel dispatches to.
      elements: Concrete integer elements moved.
      dtype: Element type; ``None`` is torch's default.
      flops: Concrete integer operations in the same cell.

    Returns:
      cost: The one-cell integer ledger.

    Raises:
      TypeError: ``elements`` or ``flops`` is not an integer.
      ValueError: A count is negative.

    """
    _validate_nonnegative_integer("elements", elements)
    _validate_nonnegative_integer("flops", flops)
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


_FUNCTION_COSTS: dict[object, Callable[..., Cost]] = {}


def cost(config: object, **kwargs: object) -> Cost:
    """Cost one complete invocation, forwarding the bus unchanged.

    Args:
      config: A config with ``cost``.
      **kwargs: The open bus; the config names what it reads.

    Returns:
      cost: The config's whole-invocation integer cost.

    Raises:
      TypeError: ``config`` has no ``cost``.

    """
    # A ``partial`` binds hyperparameters, not work; the cost is the function's.
    # Duck-typed rather than ``isinstance(config, HasCost)``: the protocol
    # lives in ``custom_types`` and names ``Cost`` in its signature, so
    # importing it here would close an import cycle.
    original: object = config
    method = getattr(original, "cost", None)
    name_target: object = original
    if not callable(method):
        try:
            method = _FUNCTION_COSTS.get(original)
        except TypeError:
            method = None
    if not callable(method) and isinstance(original, functools.partial):
        method = _FUNCTION_COSTS.get(cast(object, original.func))
    if not callable(method):
        name = getattr(name_target, "__qualname__", type(name_target).__qualname__)
        raise TypeError(
            f"{name} has no cost(); every slot under a costed container must "
            "implement HasCost, or be a function decorated with @set_cost.",
        )
    result: object = method(**kwargs)
    assert isinstance(result, Cost)
    return result


def intensity(execution: Cost) -> Report:
    """Derive FLOP-per-byte reporting values from an execution cost.

    Args:
      execution: Concrete whole-invocation FLOP and byte counts.

    Returns:
      report: Floating intensity keyed by phase, kernel, and dtype.

    """
    flops = execution["flops"]
    moved = execution["bytes"]
    assert isinstance(flops, Cost)
    assert isinstance(moved, Cost)
    return Report(
        cells={
            key: _div(float(value), float(moved.cells.get(key, 0)))
            for key, value in flops.cells.items()
        },
    )


def set_cost[**P, R](
    cost_fn: Callable[..., Cost],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Attach a cost to a plain function so a slot can hold it.

    A module whose slot takes a bare function (an activation, a schedule, a
    loss, a target parameterization) costs the slot by calling
    ``cost(fn, ...)`` with whatever ``cost_fn`` reads. ``functools.partial``
    of the result keeps the cost. A function without one is refused by
    :func:`cost`, never costed as zero.

    Args:
      cost_fn: Answers ``cost`` for the function; see :func:`map_cost` for the
        per-element case.

    Returns:
      decorator: Returns the original function unchanged.

    """

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        _FUNCTION_COSTS[function] = cost_fn
        return function

    return decorate


def map_cost(
    *,
    primal: int,
    adjoint: int,
    inputs: int = 1,
    outputs: int = 1,
    adjoint_inputs: int = 2,
    adjoint_outputs: int = 1,
) -> Callable[..., Cost]:
    """Build the cost of an elementwise map as a function of ``channels`` and ``dtype``.

    The counts are per element; the returned function scales them by the
    ``channels`` it is asked about and answers with :func:`elementwise_cost`.

    Args:
      primal: Operations per element evaluating the map.
      adjoint: Operations per element in its derivative.
      inputs: Primal input operands.
      outputs: Primal output operands.
      adjoint_inputs: Adjoint input operands (saved values, incoming gradient).
      adjoint_outputs: Adjoint output operands.

    Returns:
      cost_fn: ``cost_fn(*, channels, dtype)``, for :func:`set_cost`.

    """

    def per_channels(*, channels: int, dtype: torch.dtype | None) -> Cost:
        return elementwise_cost(
            primal=primal * channels,
            adjoint=adjoint * channels,
            channels=channels,
            dtype=dtype,
            inputs=inputs,
            outputs=outputs,
            adjoint_inputs=adjoint_inputs,
            adjoint_outputs=adjoint_outputs,
        )

    return per_channels


def matmul_cost(
    *,
    channels_in: int,
    channels_out: int,
    bias: bool = False,
    weight: bool = True,
    rows: int = 1,
    dtype: torch.dtype | None = None,
) -> Cost:
    """Cost one complete ``[M, K] @ [K, N]`` invocation and its adjoint.

    Args:
      channels_in: Inner dimension K.
      channels_out: Output width N.
      bias: Add a separate bias map and its gradient reduction.
      weight: Own the right matrix as parameters; False keeps its activation
        traffic but owns no matrix parameters.
      rows: Concrete rows M processed by this invocation.
      dtype: Element type of every operand, gradients included; ``None`` is
        torch's default. Tags every cell and sets the bytes per element.

    Returns:
      cost: Integer FLOPs and logical tensor bytes for the complete invocation.
        Bias traffic belongs to elementwise and reduction silos.

    Raises:
      TypeError: ``rows`` is not an integer.
      ValueError: ``rows`` is below one.

    """
    _validate_nonnegative_integer("channels_in", channels_in)
    _validate_nonnegative_integer("channels_out", channels_out)
    _validate_rows(rows)
    dt = resolve_dtype(dtype)
    s = dt.itemsize
    products = channels_in * channels_out
    biases = channels_out if bias else 0
    params = (products if weight else 0) + biases
    moved = s * (rows * channels_in + rows * channels_out + products)
    return Cost(
        cells={
            ("flops", "primal", "matmul", dt): 2 * rows * products,
            ("flops", "primal", "elementwise", dt): rows * biases,
            ("flops", "adjoint", "matmul", dt): 4 * rows * products,
            ("flops", "adjoint", "reduction", dt): biases * (rows - 1),
            ("bytes", "primal", "matmul", dt): moved,
            ("bytes", "primal", "elementwise", dt): s * (2 * rows * biases + biases),
            ("bytes", "adjoint", "matmul", dt): 2 * moved,
            ("bytes", "adjoint", "reduction", dt): s * (rows * biases + biases),
        },
        params=params,
        params_active=params,
    )


def elementwise_cost(
    *,
    primal: int,
    adjoint: int,
    channels: int = 0,
    params: int = 0,
    rows: int = 1,
    dtype: torch.dtype | None = None,
    inputs: int = 1,
    outputs: int = 1,
    adjoint_inputs: int = 2,
    adjoint_outputs: int = 1,
) -> Cost:
    """Cost a complete elementwise invocation and owned parameter gradients.

    The default geometry is a unary map: primal input/output and adjoint
    saved value/incoming gradient/outgoing gradient. Compound maps specify
    operand counts explicitly; FLOPs never determine traffic. ``primal`` and
    ``adjoint`` are already whole-invocation counts.

    Args:
      primal: Concrete integer operations evaluating the complete map.
      adjoint: Concrete integer backward operations excluding reductions.
      channels: Elements in one operand row.
      params: Owned parameters, read once per pass across ``rows`` rows.
      rows: Concrete rows sharing parameters and their gradient reduction.
      dtype: Element type of every operand, gradients included; ``None`` is
        torch's default.
      inputs: Primal input operands, excluding owned parameters.
      outputs: Primal output operands.
      adjoint_inputs: Adjoint input operands, excluding owned parameters.
      adjoint_outputs: Adjoint outputs, excluding parameter-gradient temporaries.

    Returns:
      cost: Whole-invocation operand I/O and parameter reductions.

    Raises:
      TypeError: A count is not an integer.
      ValueError: A count is negative or ``rows`` is below one.

    """
    for name, value in (
        ("primal", primal),
        ("adjoint", adjoint),
        ("channels", channels),
        ("params", params),
        ("rows", rows),
        ("inputs", inputs),
        ("outputs", outputs),
        ("adjoint_inputs", adjoint_inputs),
        ("adjoint_outputs", adjoint_outputs),
    ):
        _validate_nonnegative_integer(name, value)
    if rows < 1:
        raise ValueError("rows must be at least one.")
    dt = resolve_dtype(dtype)
    s = dt.itemsize
    return Cost(
        cells={
            ("flops", "primal", "elementwise", dt): primal,
            ("flops", "adjoint", "elementwise", dt): adjoint,
            ("flops", "adjoint", "reduction", dt): params * (rows - 1),
            ("bytes", "primal", "elementwise", dt): s
            * (rows * channels * (inputs + outputs) + params),
            ("bytes", "adjoint", "elementwise", dt): s
            * (
                rows * channels * (adjoint_inputs + adjoint_outputs)
                + rows * params
                + params
            ),
            ("bytes", "adjoint", "reduction", dt): s * (rows * params + params),
        },
        params=params,
        params_active=params,
    )


def reduction_cost(
    *,
    input_elements: int,
    output_groups: int = 1,
    dtype: torch.dtype | None = None,
    phase: Phase = "primal",
) -> Cost:
    """Cost one reduction's complete tensor geometry.

    Args:
      input_elements: Total concrete elements read across all output groups.
      output_groups: Concrete reduced elements written; each group uses n-1
        operations.
      dtype: Element type of the input and output; ``None`` is torch's default.
      phase: Which pass runs the reduction.

    Returns:
      cost: Reduction FLOPs and minimum unfused operand I/O, owning nothing.
        Singleton groups copy their elements; empty groups write the identity.
        Both use zero FLOPs. This describes one reduction, not its derivative.

    Raises:
      TypeError: A geometry count is not an integer.
      ValueError: A geometry count is negative.

    """
    _validate_nonnegative_integer("input_elements", input_elements)
    _validate_nonnegative_integer("output_groups", output_groups)
    dt = resolve_dtype(dtype)
    return Cost(
        cells={
            ("flops", phase, "reduction", dt): max(
                0,
                input_elements - output_groups,
            ),
            ("bytes", phase, "reduction", dt): dt.itemsize
            * (input_elements + output_groups),
        },
    )


type Device = Literal[
    "a100",
    "h100",
    "h200",
    "b200",
    "rtx5050",
    "rtx5070",
    "rtx5090",
    "rtxpro6000",
]

# Dense figures, one device, at FP32 accumulate (what training runs): TB/s, then
# tensor-core TFLOP/s by dtype, then the CUDA-core TFLOP/s every non-matmul silo
# runs at. Sparse (2:4) is twice these and is not what a dense model achieves. The
# ``float32`` tensor entry is TF32, what a matmul runs at under
# ``torch.backends.cuda.matmul.allow_tf32``.
#
# A100 80GB SXM: https://www.nvidia.com/en-us/data-center/a100/ "Specifications";
#   the starred SXM column is "with sparsity", so each is halved (TF32 312 -> 156,
#   BF16 624 -> 312, INT8 1248 -> 624). No FP8. FP64 TC 19.5; FP32 CUDA 19.5.
# H100 SXM5: NVIDIA H100 Tensor Core GPU datasheet, "no sparsity" column
#   (FP64 TC 67 is not used; the vector rate is FP32 CUDA 67). Mirrored at
#   https://www.spheron.network/blog/nvidia-h100-specs/ "Throughput by Precision".
# H200 SXM: same GH100 die and rates; 4.8 TB/s HBM3e from the same source.
# B200: HGX B200 PCF summary (8 GPUs) divided by 8 --
#   https://images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf
#   (FP4 72/8 = 9 PF, FP8 36/8 = 4.5, BF16 18/8 = 2.25, TF32 9/8 = 1.125,
#   FP32 600/8 = 75, FP64 296/8 = 37; sparse marketing halved). 8 TB/s HBM3e.
# RTX 5090, RTX 5070: NVIDIA RTX Blackwell GPU Architecture whitepaper --
#   https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
#   Table 3 (5090) and Table 6 (5070), the dense figure of each "x/y" pair, at
#   "with FP32 Accumulate". GeForce Blackwell halves FP16/FP8 tensor throughput
#   under FP32 accumulate (5090 FP16: 419 fp16-acc, 209.5 fp32-acc), so the
#   table's bf16 is half the "AI TOPS"-style number. INT8 has no accumulate
#   split and equals the FP8 fp16-acc rate. FP4 is only published at FP32 acc.
# RTX PRO 6000 Blackwell Workstation Edition: NVIDIA RTX PRO Blackwell GPU
#   Architecture whitepaper Table 4 --
#   https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf
#   Dense figures; RTX PRO does not halve under FP32 accumulate (503.8 both).
# RTX 5050: not in either whitepaper. NVIDIA's compare page --
#   https://www.nvidia.com/en-us/geforce/graphics-cards/compare/ -- gives 2560
#   CUDA cores (20 SMs), 2.57 GHz boost, 320 GB/s, 421 AI TOPS (FP4 sparse).
#   Every tensor row is the RTX 5070 row times the SM x clock ratio
#   20 x 2.57 / (48 x 2.512) = 0.4263: FP4 493.9 x 0.4263 = 210.5, twice which
#   is the published 421, so the scaling is consistent. FP32 is
#   2 x 2560 x 2.57 GHz = 13.2.
_DEVICES: Final[Mapping[Device, tuple[float, Mapping[torch.dtype, float], float]]] = {
    "a100": (
        2.039,
        {
            torch.float64: 19.5,
            torch.float32: 156,
            torch.bfloat16: 312,
            torch.float16: 312,
            torch.int8: 624,
        },
        19.5,
    ),
    "h100": (
        3.35,
        {
            torch.float64: 34,
            torch.float32: 494,
            torch.bfloat16: 989,
            torch.float16: 989,
            torch.float8_e4m3fn: 1979,
            torch.float8_e5m2: 1979,
            torch.int8: 1979,
        },
        67,
    ),
    "h200": (
        4.8,
        {
            torch.float64: 34,
            torch.float32: 494,
            torch.bfloat16: 989,
            torch.float16: 989,
            torch.float8_e4m3fn: 1979,
            torch.float8_e5m2: 1979,
            torch.int8: 1979,
        },
        67,
    ),
    "b200": (
        8.0,
        {
            torch.float64: 37,
            torch.float32: 1125,
            torch.bfloat16: 2250,
            torch.float16: 2250,
            torch.float8_e4m3fn: 4500,
            torch.float8_e5m2: 4500,
            torch.int8: 4500,
            torch.float4_e2m1fn_x2: 9000,
        },
        75,
    ),
    "rtx5050": (
        0.320,
        {
            torch.float32: 13.2,
            torch.bfloat16: 26.3,
            torch.float16: 26.3,
            torch.float8_e4m3fn: 52.6,
            torch.float8_e5m2: 52.6,
            torch.int8: 105.3,
            torch.float4_e2m1fn_x2: 210.5,
        },
        13.2,
    ),
    "rtx5070": (
        0.672,
        {
            torch.float32: 30.9,
            torch.bfloat16: 61.7,
            torch.float16: 61.7,
            torch.float8_e4m3fn: 123.5,
            torch.float8_e5m2: 123.5,
            torch.int8: 246.9,
            torch.float4_e2m1fn_x2: 493.9,
        },
        30.9,
    ),
    "rtx5090": (
        1.792,
        {
            torch.float32: 104.8,
            torch.bfloat16: 209.5,
            torch.float16: 209.5,
            torch.float8_e4m3fn: 419,
            torch.float8_e5m2: 419,
            torch.int8: 838,
            torch.float4_e2m1fn_x2: 1676,
        },
        104.8,
    ),
    "rtxpro6000": (
        1.792,
        {
            torch.float32: 251.9,
            torch.bfloat16: 503.8,
            torch.float16: 503.8,
            torch.float8_e4m3fn: 1007.6,
            torch.float8_e5m2: 1007.6,
            torch.int8: 1007.6,
            torch.float4_e2m1fn_x2: 2015.2,
        },
        126.0,
    ),
}
"""``(TB/s, {dtype: tensor-core TFLOP/s}, CUDA-core TFLOP/s)`` per device."""


def peak() -> Report:
    """Return floating device ceilings and their arithmetic-intensity ridges.

    Keyed ``(device, dtype, measure, kernel)`` with measures ``flops`` (FLOP/s),
    ``bytes`` (B/s), and ``intensity`` (FLOP/B: the ridge). A ``matmul`` cell
    is the dense tensor-core peak for its dtype; every other silo runs on the
    CUDA cores at the FP32 vector rate regardless of dtype, index dtypes
    included, since a gather or a sort moves ``int64`` at the same rate.

    ``peak()["h100", torch.bfloat16]`` is one device at one dtype, measures by
    kernel. Sums across devices or dtypes rank alternatives and mean nothing.

    Returns:
      peak: The floating datasheet table, owning nothing.

    """
    cells: dict[tuple[object, ...], float] = {}
    for device, (bandwidth, tensor, vector) in _DEVICES.items():
        for dtype in (*tensor, torch.int64, torch.int32, torch.bool):
            for kernel in KERNELS:
                rate = tensor.get(dtype) if kernel == "matmul" else vector
                if rate is None:
                    continue
                cells[(device, dtype, "flops", kernel)] = rate * 1e12
                cells[(device, dtype, "bytes", kernel)] = bandwidth * 1e12
                cells[(device, dtype, "intensity", kernel)] = rate / bandwidth
    return Report(cells=cells)


def utilization(
    cost: Cost,
    *,
    device: Device,
    duration_sec: float = math.inf,
) -> Report:
    """Report how well each whole invocation uses ``device``.

    Without ``duration_sec`` the answer is the invocation's intensity over the
    device ridge. With ``duration_sec`` the answer is achieved FLOP/s over the
    roofline ceiling that applies -- the lower of compute peak and intensity
    times bandwidth. This report intentionally returns float ratios.

    Args:
      cost: Whole-invocation cost at one concrete geometry.
      device: Which datasheet to read.
      duration_sec: Wall seconds the invocation took, accelerator work
        complete; omit for the shape limit alone.

    Returns:
      ratio: At ``(phase, kernel, dtype)`` keys, owning nothing; ``inf`` where
        the device lists no peak for the cell.

    """
    ceiling = peak()[device].cells
    ratios = intensity(cost).cells
    cells: dict[tuple[object, ...], float] = {}
    for key, flops in cost["flops"].cells.items():
        _, kernel, dtype = key
        compute = ceiling.get((dtype, "flops", kernel), 0.0)
        bandwidth = ceiling.get((dtype, "bytes", kernel), 0.0)
        if math.isinf(duration_sec):
            ridge = ceiling.get((dtype, "intensity", kernel), 0.0)
            cells[key] = _div(ratios.get(key, math.inf), ridge)
            continue
        memory = ratios.get(key, math.inf) * bandwidth
        achieved = flops / duration_sec
        cells[key] = _div(achieved, min(compute, memory))
    return Report(cells=cells)


def _axis_of(key: tuple[object, ...], value: object) -> int:
    """Return the position in ``key`` whose axis holds values like ``value``."""
    for i, have in enumerate(key):
        if _kind(have) == _kind(value):
            return i
    raise KeyError(f"{value!r} names no axis of a {len(key)}-axis table.")


def _canonical(value: object) -> object:
    """Spell a dtype given by name or alias as the ``torch.dtype`` the keys hold."""
    if isinstance(value, str):
        named: object = DTYPE_ALIASES.get(value, getattr(torch, value, None))
        if isinstance(named, torch.dtype):
            return named
    return value


def _kind(value: object) -> str:
    if isinstance(value, torch.dtype):
        return "dtype"
    if value in _REPORT_MEASURES:
        return "measure"
    if value in PHASES:
        return "phase"
    if value in KERNELS:
        return "kernel"
    if value in _DEVICES:
        return "device"
    raise KeyError(f"{value!r} is not a measure, phase, kernel, dtype or device.")


def _validate_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer; got {value!r}.")


def _validate_nonnegative_integer(name: str, value: object) -> None:
    _validate_integer(name, value)
    assert isinstance(value, int)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative; got {value}.")


def _validate_rows(rows: int) -> None:
    _validate_integer("rows", rows)
    if rows < 1:
        raise ValueError("rows must be at least one.")


# The ``measure`` axis, wherever it sits, becomes the column group; the last other
# axis joins it as columns (or, with only one other axis, labels the rows); every
# axis between labels the rows.
def _grid(cells: Mapping[tuple[object, ...], int | float]) -> str:
    """Render a table as an aligned grid with a totals row."""
    if not cells:
        return "(empty)"
    width = len(next(iter(cells)))
    if width == 0:
        return _si(next(iter(cells.values())))
    axis = next(
        (i for i in range(width) if all(k[i] in _REPORT_MEASURES for k in cells)),
        None,
    )
    measures: list[object] = (
        [m for m in _REPORT_MEASURES if any(k[axis] == m for k in cells)]
        if axis is not None
        else [""]
    )
    others = [i for i in range(width) if i != axis]
    rows_only = axis is not None and len(others) == 1
    column_axis = None if rows_only or not others else others[-1]
    row_axes = others if rows_only else others[:-1]
    columns_axis: list[object] = (
        [()]
        if column_axis is None
        else sorted({k[column_axis] for k in cells}, key=_column_order)
    )
    order = (*PHASES, *KERNELS)
    labels = sorted(
        {tuple(str(k[i]) for i in row_axes) for k in cells},
        key=lambda label: tuple(order.index(x) if x in order else -1 for x in label),
    )
    depth = max(1, *(len(label) for label in labels))
    # The measure and the column axis stack as two header lines rather than
    # widening every column to ``flops[bfloat16]``.
    if column_axis is None:
        header = [[_axis_name(m) for m in measures]]
    elif axis is None:
        header = [[_axis_name(d) for d in columns_axis]]
    else:
        header = [
            [str(m) for m in measures for _ in columns_axis],
            [_axis_name(d) for _ in measures for d in columns_axis],
        ]
    lookup = {
        tuple(str(x) if i in row_axes else x for i, x in enumerate(k)): v
        for k, v in cells.items()
    }
    rows: list[list[str]] = [[*[""] * depth, *line] for line in header]
    totals = [0.0] * (len(measures) * len(columns_axis))
    for label in labels:
        fixed = dict(zip(row_axes, label, strict=True))
        values = [
            lookup.get(_key(width, fixed, (axis, m), (column_axis, d)), 0.0)
            for m in measures
            for d in columns_axis
        ]
        totals = [t + v for t, v in zip(totals, values, strict=True)]
        rows.append(_line(label, values, depth=depth))
    if len(labels) > 1 and "intensity" not in measures:
        rows.append(_line(("total",), totals, depth=depth))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        " ".join(
            cell.ljust(widths[i]) if i < depth else cell.rjust(widths[i])
            for i, cell in enumerate(row)
        ).rstrip()
        for row in rows
    )


def _key(
    width: int,
    fixed: Mapping[int, object],
    *placed: tuple[int | None, object],
) -> tuple[object, ...]:
    """Assemble a cell key from row labels plus the measure and column axes."""
    parts = {
        **fixed,
        **{index: value for index, value in placed if index is not None},
    }
    return tuple(parts[i] for i in range(width))


def _column_order(axis: object) -> tuple[int, str]:
    if isinstance(axis, torch.dtype):
        return (axis.itemsize, str(axis))
    order = (*_REPORT_MEASURES, *PHASES, *KERNELS)
    return (order.index(axis) if axis in order else -1, str(axis))


def _line(
    label: tuple[str, ...],
    values: list[int | float],
    *,
    depth: int,
) -> list[str]:
    """Format one grid row."""
    return [*label, *[""] * (depth - len(label)), *(_si(v) for v in values)]


def _axis_name(axis: object) -> str:
    for alias, dtype in DTYPE_ALIASES.items():
        if axis == dtype:
            return alias
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
