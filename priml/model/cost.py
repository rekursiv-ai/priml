"""Analytical per-token cost of a model, read from its config.

Work is attributed to five kernel silos, one per roofline regime:

- ``matmul``: mm, bmm, conv, sdpa. Tensor-core work; the MFU numerator and
  what ``torch.utils.flop_counter`` counts.
- ``elementwise``: one output per input element.
- ``reduction``: many inputs to one output; a scan is a reduction that keeps
  its prefixes.
- ``selection``: gather in the primal, scatter-add in the adjoint.
- ``sort``: argsort, top-k. Traffic is n log n.

The silo follows the kernel dispatched, not the algebra: ``sum(x)`` is a
reduction, ``ones @ x`` is a matmul. Softmax is elementwise, reduction,
elementwise.

Counting policy:

- A multiply-accumulate is two FLOPs; any other floating op is one. A
  reduction over n is n-1. A gather is zero FLOPs, one element moved; its
  scatter-add is one add per element.
- Analytical training algorithm with saved primal values, not a fused
  kernel. The adjoint includes local derivatives and parameter-gradient
  reductions; not the optimizer.
- ``num_tokens`` is the rows sharing one parameter. Its gradient is summed
  over them, (n-1)/n per row, in ``adjoint.flops.reduction``. Only the two
  primitives write that term.
- Attention is counted over ``min(window, seq_len)`` keys with no causal
  discount (PaLM convention). Recompute excluded: MFU, not HFU.
- Bytes are elements; multiply by itemsize at use.

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
        """Divide per silo; a zero denominator is ``inf`` (nothing moved is compute-bound)."""
        if isinstance(other, KernelStats):
            return KernelStats(
                matmul=_div(self.matmul, other.matmul),
                elementwise=_div(self.elementwise, other.elementwise),
                reduction=_div(self.reduction, other.reduction),
                selection=_div(self.selection, other.selection),
                sort=_div(self.sort, other.sort),
            )
        return self._scale(1 / other)

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
    """Elements moved per token."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Compute:
    """What one differentiation pass computes and moves."""

    flops: Flops = field(default_factory=Flops)
    bytes: Bytes = field(default_factory=Bytes)

    @property
    def intensity(self) -> KernelStats:
        """FLOPs per element moved, by silo: the roofline's x-axis."""
        return self.flops / self.bytes

    def __add__(self, other: Compute) -> Compute:
        return Compute(flops=self.flops + other.flops, bytes=self.bytes + other.bytes)

    def __mul__(self, other: int) -> Compute:
        if isinstance(other, bool) or not isinstance(other, int):  # pyright: ignore[reportUnnecessaryIsInstance] -- the runtime check is the contract
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
    """Per-token state carried across a decode step."""

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
        if isinstance(other, bool) or not isinstance(other, int):  # pyright: ignore[reportUnnecessaryIsInstance] -- the runtime check is the contract
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
          tiled: Work scaled by ``rows``; parameters and their reads by ``copies``.

        """
        repeated = rows * self
        return replace(
            repeated,
            params=copies * self.params,
            params_active=copies * self.params_active,
            primal=replace(
                repeated.primal,
                bytes=replace(
                    repeated.primal.bytes,
                    matmul=copies * self.primal.bytes.matmul,
                ),
            ),
            adjoint=replace(
                repeated.adjoint,
                bytes=replace(
                    repeated.adjoint.bytes,
                    matmul=copies * self.adjoint.bytes.matmul,
                ),
            ),
        )


@runtime_checkable
class HasCost(Protocol):
    """A config that prices the module it builds."""

    def cost(self, **kwargs: object) -> Cost:
        """Price one token through the module ``self`` builds.

        Args:
          **kwargs: The open message bus (``seq_len``, ``num_tokens``, ...).

        Returns:
          cost: Per-token cost.

        """
        ...


def cost(config: object, **kwargs: object) -> Cost:
    """Price a child config; a child without ``cost`` raises rather than pricing zero.

    Args:
      config: A child config slot's value.
      **kwargs: The open message bus, forwarded unchanged.

    Returns:
      cost: The child's per-token cost.

    Raises:
      TypeError: ``config`` has no ``cost``.

    """
    if not isinstance(config, HasCost):
        raise TypeError(
            f"{type(config).__qualname__} has no cost(); every config under a "
            "priced container must implement HasCost.",
        )
    return config.cost(**kwargs)


def matmul_cost(
    *,
    channels_in: int,
    channels_out: int,
    bias: bool = False,
    weight: bool = True,
    num_tokens: int = 1,
) -> Cost:
    """Price a ``[channels_in] -> [channels_out]`` matmul on one token.

    Args:
      channels_in: Input width.
      channels_out: Output width.
      bias: Add a bias vector to each output row.
      weight: The other operand is a learned matrix. ``False`` prices an
        activation-activation product: same FLOPs, no parameters.
      num_tokens: Rows sharing the bias.

    Returns:
      cost: Two FLOPs per product in the primal, four in the adjoint; the
        weight read once each way, the output row written once each way.

    """
    products = channels_in * channels_out
    weights = products if weight else 0
    biases = channels_out if bias else 0
    params = weights + biases
    return Cost(
        primal=Compute(
            flops=Flops(matmul=2 * products, elementwise=biases),
            bytes=Bytes(matmul=params, elementwise=channels_out),
        ),
        adjoint=Compute(
            flops=Flops(
                matmul=4 * products,
                reduction=biases * (num_tokens - 1) / num_tokens,
            ),
            bytes=Bytes(matmul=params, elementwise=channels_out),
        ),
        params=params,
        params_active=params,
    )


def elementwise_cost(
    *,
    primal: float,
    adjoint: float,
    channels: int = 0,
    params: int = 0,
    num_tokens: int = 1,
) -> Cost:
    """Price elementwise work plus the gradient of the parameters it owns.

    Args:
      primal: Operations per token evaluating the map.
      adjoint: Operations per token pulling a gradient back, excluding the
        parameter gradient's reduction over rows, which is added here.
      channels: Width of the output row; zero writes nothing new.
      params: Parameters owned, all read per primal.
      num_tokens: Rows sharing the parameters.

    Returns:
      cost: Elementwise FLOPs both ways; the parameters' reduction in the adjoint.

    """
    return Cost(
        primal=Compute(
            flops=Flops(elementwise=primal),
            bytes=Bytes(elementwise=channels + params),
        ),
        adjoint=Compute(
            flops=Flops(
                elementwise=adjoint,
                reduction=params * (num_tokens - 1) / num_tokens,
            ),
            bytes=Bytes(elementwise=channels + params),
        ),
        params=params,
        params_active=params,
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
      itemsize: Bytes per element of the weight and state dtype.
      peak_bytes_per_sec: Datasheet HBM bandwidth.

    Returns:
      achieved: Fraction of peak bandwidth.

    """
    moved = (cost.params_active + batch * context_len * cost.bytes_state) * itemsize
    return moved * steps_per_sec / peak_bytes_per_sec


def _div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("inf")
