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
- ``Bytes`` and ``bytes_state`` hold bytes, not element counts. Helpers use
  a uniform ``itemsize=4`` unless the caller specifies another width. This is
  an accounting assumption, not mixed-dtype profiling.
- Traffic is the analytical unfused algorithm's minimum tensor operand I/O:
  each primitive reads its inputs and writes its outputs once. Intermediates
  between primitives count even when a fused implementation keeps them on chip.
  This does not predict HBM traffic, cache reuse, or physical memory transactions.
  Traffic geometry is explicit and never inferred from FLOP counts. Nonlinear
  tensor operators move operands once, not once per internal scalar operation.

``cost(**kwargs)`` takes ``forward``'s open keyword bus. A leaf names what it
reads and ``del``s the rest; a container forwards the bus to every child.

:func:`matmul_cost` (``weight=False`` for ``QK^T``) and
:func:`elementwise_cost` are the two primitives that own parameters. The other
silos are ``Cost`` literals. A module-level function assigned as a class
attribute (``cost = attention_kernel_cost``) binds as the method.

A differentiation mode is a ``Compute`` field on ``Cost``: ``primal`` and
``adjoint`` (VJP) today; a ``tangent`` (JVP) or ``hessian_vector`` is one
more field and one line in each operator when a consumer needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, Self, overload, runtime_checkable

import math


__all__ = [
    "Bytes",
    "Compute",
    "Cost",
    "Flops",
    "HasCost",
    "KernelStats",
    "cost",
    "elementwise_cost",
    "matmul_cost",
    "mbu",
    "mfu",
    "reduction_cost",
    "utilization",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class KernelStats:
    """One number per kernel silo."""

    matmul: float = 0
    elementwise: float = 0
    reduction: float = 0
    selection: float = 0
    sort: float = 0

    @property
    def total(self) -> float:
        """Sum over every silo."""
        return (
            self.matmul + self.elementwise + self.reduction + self.selection + self.sort
        )

    def __add__(self, other: Self) -> Self:
        return replace(
            self,
            matmul=self.matmul + other.matmul,
            elementwise=self.elementwise + other.elementwise,
            reduction=self.reduction + other.reduction,
            selection=self.selection + other.selection,
            sort=self.sort + other.sort,
        )

    @overload
    def __mul__(self, other: float) -> Self: ...
    @overload
    def __mul__(self, other: KernelStats) -> KernelStats: ...
    def __mul__(self, other: float | KernelStats) -> Self | KernelStats:
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, KernelStats):
            return KernelStats(
                matmul=self.matmul * other.matmul,
                elementwise=self.elementwise * other.elementwise,
                reduction=self.reduction * other.reduction,
                selection=self.selection * other.selection,
                sort=self.sort * other.sort,
            )
        return self._scale(other)

    __rmul__ = __mul__

    @overload
    def __truediv__(self, other: float) -> Self: ...
    @overload
    def __truediv__(self, other: KernelStats) -> KernelStats: ...
    def __truediv__(self, other: float | KernelStats) -> Self | KernelStats:
        """Divide per silo; zero over zero is NaN, nonzero over zero is signed infinity."""
        if isinstance(other, KernelStats):
            return KernelStats(
                matmul=_div(self.matmul, other.matmul),
                elementwise=_div(self.elementwise, other.elementwise),
                reduction=_div(self.reduction, other.reduction),
                selection=_div(self.selection, other.selection),
                sort=_div(self.sort, other.sort),
            )
        return replace(
            self,
            matmul=_div(self.matmul, other),
            elementwise=_div(self.elementwise, other),
            reduction=_div(self.reduction, other),
            selection=_div(self.selection, other),
            sort=_div(self.sort, other),
        )

    def _scale(self, factor: float) -> Self:
        return replace(
            self,
            matmul=self.matmul * factor,
            elementwise=self.elementwise * factor,
            reduction=self.reduction * factor,
            selection=self.selection * factor,
            sort=self.sort * factor,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Flops(KernelStats):
    """Floating operations per token."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Bytes(KernelStats):
    """Bytes moved per token under the analytical tensor-I/O convention."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Total:
    """A pass's FLOPs and bytes summed over every silo."""

    flops: float
    bytes: float


@dataclass(frozen=True, slots=True, kw_only=True)
class Compute:
    """What one differentiation pass computes and moves."""

    flops: Flops = field(default_factory=Flops)
    bytes: Bytes = field(default_factory=Bytes)

    @property
    def intensity(self) -> KernelStats:
        """Return FLOPs per byte moved, by silo: the roofline's x-axis.

        Its ``total`` sums the silo ratios and means nothing; the pass as a
        whole is memory-bound or not by ``total.flops / total.bytes``.
        """
        return self.flops / self.bytes

    @property
    def total(self) -> Total:
        """Sum every silo: all FLOPs and all bytes of this pass."""
        return Total(flops=self.flops.total, bytes=self.bytes.total)

    def __add__(self, other: Compute) -> Compute:
        return Compute(flops=self.flops + other.flops, bytes=self.bytes + other.bytes)

    def __mul__(self, other: int) -> Compute:
        if isinstance(other, bool) or not isinstance(other, int):  # pyright: ignore[reportUnnecessaryIsInstance] -- Python operator dispatch must reject non-integers from untyped callers.
            return NotImplemented
        return Compute(flops=self.flops * other, bytes=self.bytes * other)

    __rmul__ = __mul__


@dataclass(frozen=True, slots=True, kw_only=True)
class Cost:
    """Per-token cost of one module, additive over children."""

    primal: Compute = field(default_factory=Compute)
    adjoint: Compute = field(default_factory=Compute)

    params: int = 0
    """Parameters the module owns."""

    params_active: int = 0
    """Parameters one token reads; fewer than ``params`` only when routed."""

    bytes_state: int = 0
    """Bytes of per-token state carried across a decode step."""

    @property
    def training(self) -> Compute:
        """Primal plus adjoint: one training step's work."""
        return self.primal + self.adjoint

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            primal=self.primal + other.primal,
            adjoint=self.adjoint + other.adjoint,
            params=self.params + other.params,
            params_active=self.params_active + other.params_active,
            bytes_state=self.bytes_state + other.bytes_state,
        )

    def __mul__(self, other: int) -> Cost:
        # A Cost is repeated over modules, never scaled by a fraction.
        if isinstance(other, bool) or not isinstance(other, int):  # pyright: ignore[reportUnnecessaryIsInstance] -- Python operator dispatch must reject non-integers from untyped callers.
            return NotImplemented
        return Cost(
            primal=self.primal * other,
            adjoint=self.adjoint * other,
            params=self.params * other,
            params_active=self.params_active * other,
            bytes_state=self.bytes_state * other,
        )

    __rmul__ = __mul__

    def tile(self, rows: int, *, copies: int = 1) -> Cost:
        """Run ``rows`` times per token, owned ``copies`` times.

        A norm over every head row of one shared weight runs ``rows`` times
        but is owned once; ``rows * cost`` would multiply its parameters too.

        Args:
          rows: Times the work runs per token.
          copies: Times the parameters exist.

        Returns:
          tiled: Work and all traffic scaled by ``rows``; ownership by ``copies``.

        """
        return replace(
            rows * self,
            params=copies * self.params,
            params_active=copies * self.params_active,
        )


@runtime_checkable
class HasCost(Protocol):
    """A config that prices the module it builds."""

    def cost(self, **kwargs: object) -> Cost:
        """Price one token through the module ``self`` builds.

        Args:
          **kwargs: The open message bus (``seq_len``, ``rows``, ...).

        Returns:
          cost: Per-token cost.

        """
        ...


def cost(config: object, **kwargs: object) -> Cost:
    """Price a config per token for ``batch_size`` sequences of ``seq_len`` tokens.

    A caller states the batch geometry, never the sharing rows. Every token of
    the batch reads the same weights, so ``rows`` -- what one parameter and
    its gradient reduction are amortized over -- is ``seq_len * batch_size``
    unless a container already set it for a child with its own geometry (a
    puzzle, an image position, an expert's occupancy). Both default to one,
    so ``seq_len`` alone prices one sequence of that length.

    Args:
      config: A config with ``cost``.
      **kwargs: The open message bus. ``seq_len`` is tokens per sequence,
        attention's reach before any window; ``batch_size`` is sequences per
        step and amortizes weights only, since attention never reuses
        another sequence's keys. Anything else is forwarded unchanged.

    Returns:
      cost: The config's per-token cost.

    Raises:
      TypeError: ``config`` has no ``cost``, or ``seq_len``/``batch_size``
        is not an integer.

    """
    if not isinstance(config, HasCost):
        raise TypeError(
            f"{type(config).__qualname__} has no cost(); every config under a "
            "priced container must implement HasCost.",
        )
    seq_len = kwargs.setdefault("seq_len", 1)
    batch_size = kwargs.setdefault("batch_size", 1)
    if not isinstance(seq_len, int) or not isinstance(batch_size, int):
        raise TypeError("seq_len and batch_size must be integers.")
    kwargs.setdefault("rows", seq_len * batch_size)
    return config.cost(**kwargs)


def matmul_cost(
    *,
    channels_in: int,
    channels_out: int,
    bias: bool = False,
    weight: bool = True,
    rows: float = 1,
    itemsize: int = 4,
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
      itemsize: Uniform bytes per operand element, including gradients.

    Returns:
      cost: Per-row FLOPs and unfused tensor I/O. The primal moves
        ``itemsize * (K + N + K*N/M)`` bytes; each adjoint product moves the
        same amount. Bias traffic belongs to elementwise and reduction silos.

    Raises:
      ValueError: ``rows`` is nonfinite or below one, or ``itemsize``
        is not positive.

    """
    _validate_geometry(rows=rows, itemsize=itemsize)
    products = channels_in * channels_out
    biases = channels_out if bias else 0
    params = (products if weight else 0) + biases
    moved = itemsize * (channels_in + channels_out + products / rows)
    return Cost(
        primal=Compute(
            flops=Flops(matmul=2 * products, elementwise=biases),
            bytes=Bytes(
                matmul=moved,
                elementwise=itemsize * (2 * biases + biases / rows),
            ),
        ),
        adjoint=Compute(
            flops=Flops(
                matmul=4 * products,
                reduction=biases * (rows - 1) / rows,
            ),
            bytes=Bytes(
                matmul=2 * moved,
                reduction=itemsize * (biases + biases / rows),
            ),
        ),
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
    itemsize: int = 4,
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
      itemsize: Uniform bytes per operand element, including gradients.
      inputs: Primal input operands, excluding owned parameters.
      outputs: Primal output operands.
      adjoint_inputs: Adjoint input operands, excluding owned parameters.
      adjoint_outputs: Adjoint outputs, excluding parameter-gradient temporaries.

    Returns:
      cost: Explicit operand I/O with parameter reductions in the adjoint.

    Raises:
      ValueError: ``rows`` is nonfinite or below one, or ``itemsize``
        is not positive.

    """
    _validate_geometry(rows=rows, itemsize=itemsize)
    return Cost(
        primal=Compute(
            flops=Flops(elementwise=primal),
            bytes=Bytes(
                elementwise=itemsize * (channels * (inputs + outputs) + params / rows),
            ),
        ),
        adjoint=Compute(
            flops=Flops(
                elementwise=adjoint,
                reduction=params * (rows - 1) / rows,
            ),
            bytes=Bytes(
                elementwise=itemsize
                * (
                    channels * (adjoint_inputs + adjoint_outputs)
                    + params / rows
                    + params
                ),
                reduction=itemsize * (params + params / rows),
            ),
        ),
        params=params,
        params_active=params,
    )


def reduction_cost(
    *,
    input_elements: float,
    output_groups: float = 1,
    rows: float = 1,
    itemsize: int = 4,
) -> Compute:
    """Price one reduction's explicit tensor geometry, amortized over tokens.

    Args:
      input_elements: Total elements read across all output groups.
      output_groups: Reduced elements written; each group uses n-1 operations.
      rows: Tokens sharing this reduction's work and traffic.
      itemsize: Uniform bytes per input and output element.

    Returns:
      compute: Reduction FLOPs and minimum unfused operand I/O. Singleton
        groups copy their elements; empty groups write the identity. Both use
        zero FLOPs. This describes one reduction, not its derivative or HBM.

    Raises:
      ValueError: ``rows`` is nonfinite or below one, or ``itemsize``
        is not positive.

    """
    _validate_geometry(rows=rows, itemsize=itemsize)
    return Compute(
        flops=Flops(reduction=max(0, input_elements - output_groups) / rows),
        bytes=Bytes(reduction=itemsize * (input_elements + output_groups) / rows),
    )


def utilization(cost: Cost, *, tokens_per_sec: float, peak: KernelStats) -> KernelStats:
    """Achieved fraction of each silo's ceiling on a training step; ``.matmul`` is MFU.

    Args:
      cost: Per-token cost at the batch's sequence length.
      tokens_per_sec: Measured ``tokens_per_step / step_time``.
      peak: Per-silo ceiling summed over every device the tokens spanned.
        ``matmul`` is the dense FLOP/s for the dtype.

    Returns:
      achieved: Fraction of peak per silo.

    """
    return cost.training.flops * tokens_per_sec / peak


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
    return cost.training.flops.matmul * tokens_per_sec / peak_flops_per_sec


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


def _validate_geometry(*, rows: float, itemsize: int) -> None:
    if not math.isfinite(rows) or rows < 1:
        raise ValueError("rows must be finite and at least one.")
    if itemsize <= 0:
        raise ValueError("itemsize must be positive.")


def _div(numerator: float, denominator: float) -> float:
    if denominator:
        return numerator / denominator
    if numerator == 0 or math.isnan(numerator):
        return math.nan
    return math.copysign(math.inf, numerator) * math.copysign(1, denominator)
