"""Validate a config's analytical ``cost`` against what torch actually dispatches.

Builds the module, runs one forward and backward under
:class:`torch.utils.flop_counter.FlopCounterMode` and a dispatch-mode byte
tally, and compares ``training.flops.matmul`` and ``training.bytes.matmul``
per token to what torch counted, and ``params`` to ``sum(p.numel())``.

The FLOP counter's registry is exactly the matmul silo (mm, bmm, addmm,
convolution, sdpa). The byte tally weighs every operand an aten op read or
wrote, attributed to a silo by the op's name; only the matmul silo is gated
on it, because that is the one whose dispatched operands ARE the analytical
convention (each product reads its two inputs and writes its output once).
The other silos are inspectable through :func:`measured_traffic` but differ
from the analytical count by torch's own copies -- ``ones_like`` seeding the
backward, ``clone``/``cat`` around a fused kernel -- that the unfused
algorithm does not have.

A documented estimate passes ``expected_ratio``; the harness then holds the
estimate to exactly that factor of the measurement, for FLOPs and bytes alike.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Final, cast, override

import math

from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from torch.utils.flop_counter import FlopCounterMode

import torch

from priml.cost import (
    KERNELS,
    Cost,
    cost,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from configgle import Makeable
    from torch._ops import OpOverload

    from priml.cost import Kernel


def assert_cost_matches_torch[I: (Tensor, tuple[Tensor, ...])](
    config: Makeable[nn.Module],
    *,
    build_input: Callable[[], I],
    num_tokens: int,
    run: Callable[[nn.Module, I], Tensor] | None = None,
    expected_ratio: float = 1.0,
    check_bytes: bool = True,
    seed: int = 0,
    **bus: object,
) -> Cost:
    """Build ``config``, measure one forward+backward, compare to its ``cost``.

    Args:
      config: Finalized on a copy; the caller's is untouched.
      build_input: Produces the forward input, a tensor or a tuple of them.
      num_tokens: Tokens the built input holds, so torch's step total becomes
        a per-token figure comparable to ``cost``.
      run: Applies the module to the built input; defaults to
        ``module(*inputs)``. Must return a tensor to reduce for backward.
      expected_ratio: ``analytical / measured`` to hold for matmul FLOPs and
        matmul bytes alike; ``1.0`` is exact.
      check_bytes: Also gate matmul bytes. ``False`` only where the analytical
        traffic convention is known to differ from the dispatched operands
        (convolution, attention kernels); see Issue#20739.
      seed: For ``build_input`` and any random init.
      **bus: Named arguments ``cost`` takes (``seq_len``, ``batch_size``,
        ``dtype``, ...), forwarded unchanged.

    Returns:
      analytical: The config's own cost, for further assertions.

    Raises:
      ValueError: Matmul FLOPs per token, matmul bytes per token, or the
        parameter count disagree.

    """
    finalized = config.copy_tree().finalize()
    analytical = cost(finalized, **bus)

    torch.manual_seed(seed)
    module = finalized.make()
    module.train()
    params = sum(p.numel() for p in module.parameters())
    if analytical.params != params:
        raise ValueError(
            f"cost.params={analytical.params} but the module owns {params}",
        )

    torch.manual_seed(seed)
    inputs = build_input()
    traffic = _TrafficMode()
    with FlopCounterMode(display=False) as counter, traffic:
        _forward_backward(module, inputs, run)
    measured = counter.get_total_flops() / num_tokens
    matmul = analytical["flops", "matmul"].sum()
    if matmul != expected_ratio * measured:
        raise ValueError(
            f"cost reports {matmul} matmul FLOPs/token; torch measured {measured} "
            f"(ratio {matmul / measured if measured else float('inf'):.4f}, "
            f"expected {expected_ratio})",
        )
    if not check_bytes:
        return analytical
    moved = traffic.bytes["matmul"] / num_tokens
    matmul_bytes = analytical["bytes", "matmul"].sum()
    # Bytes are sums of amortized fractions (``K*N/rows``), so the two sides
    # meet only to rounding; FLOPs above are integers and compare exactly.
    if not math.isclose(matmul_bytes, expected_ratio * moved, rel_tol=1e-9):
        raise ValueError(
            f"cost reports {matmul_bytes} matmul bytes/token; torch moved {moved} "
            f"(ratio {matmul_bytes / moved if moved else float('inf'):.4f}, "
            f"expected {expected_ratio})",
        )
    return analytical


def measured_traffic[I: (Tensor, tuple[Tensor, ...])](
    config: Makeable[nn.Module],
    *,
    build_input: Callable[[], I],
    num_tokens: int,
    run: Callable[[nn.Module, I], Tensor] | None = None,
    seed: int = 0,
) -> Mapping[Kernel, float]:
    """Tally the bytes every aten op moved in one forward+backward, per silo.

    An op's traffic is the size of every tensor it read plus every tensor it
    wrote; views and allocations move nothing. The silo is the op's name
    (see :data:`_SILO_OPS`); anything unlisted is elementwise.

    Args:
      config: Finalized on a copy; the caller's is untouched.
      build_input: Produces the forward input, a tensor or a tuple of them.
      num_tokens: Tokens the built input holds; the tally is divided by it.
      run: Applies the module to the built input; defaults to
        ``module(*inputs)``.
      seed: For ``build_input`` and any random init.

    Returns:
      traffic: Bytes per token, one entry per silo in :data:`KERNELS`.

    """
    finalized = config.copy_tree().finalize()
    torch.manual_seed(seed)
    module = finalized.make()
    module.train()
    torch.manual_seed(seed)
    inputs = build_input()
    traffic = _TrafficMode()
    with traffic:
        _forward_backward(module, inputs, run)
    return {kernel: traffic.bytes[kernel] / num_tokens for kernel in KERNELS}


def _forward_backward[I: (Tensor, tuple[Tensor, ...])](
    module: nn.Module,
    inputs: I,
    run: Callable[[nn.Module, I], Tensor] | None,
) -> None:
    """Run one forward and, when there is a graph, one backward."""
    if run is None:
        args = inputs if isinstance(inputs, tuple) else (inputs,)
        output = cast(Tensor, module(*args))
    else:
        output = run(module, inputs)
    # A module with no parameters fed an integer input (a rotary table on
    # positions) has no graph to pull a gradient through; its adjoint is
    # zero and the forward count is the whole measurement.
    if output.requires_grad:
        output.sum().backward()


# Aten overload-packet names by silo. The matmul set mirrors ``FlopCounterMode``'s
# registry, so the two measurements gate the same ops. Anything absent is
# elementwise, the silo with no distinguishing structure.
_SILO_OPS: Final[Mapping[str, Kernel]] = {
    **dict.fromkeys(
        (
            "mm",
            "bmm",
            "addmm",
            "baddbmm",
            "addbmm",
            "convolution",
            "_convolution",
            "convolution_backward",
            "_scaled_dot_product_flash_attention_for_cpu",
            "_scaled_dot_product_flash_attention_for_cpu_backward",
            "_scaled_dot_product_efficient_attention",
            "_scaled_dot_product_efficient_attention_backward",
            "_scaled_dot_product_flash_attention",
            "_scaled_dot_product_flash_attention_backward",
            "_scaled_dot_product_cudnn_attention",
            "_scaled_dot_product_cudnn_attention_backward",
        ),
        "matmul",
    ),
    **dict.fromkeys(
        (
            "sum",
            "mean",
            "max",
            "min",
            "amax",
            "amin",
            "argmax",
            "argmin",
            "var",
            "var_mean",
            "std",
            "norm",
            "linalg_vector_norm",
            "logsumexp",
            "_softmax",
            "_log_softmax",
            "_softmax_backward_data",
            "_log_softmax_backward_data",
            "cumsum",
            "cumprod",
            "prod",
            "any",
            "all",
            "max_pool2d_with_indices",
            "max_pool2d_with_indices_backward",
            "avg_pool2d",
            "avg_pool2d_backward",
        ),
        "reduction",
    ),
    **dict.fromkeys(
        (
            "index",
            "index_select",
            "gather",
            "scatter",
            "scatter_add",
            "index_add",
            "index_put",
            "index_put_",
            "_index_put_impl_",
            "index_add_",
            "scatter_",
            "scatter_add_",
            "embedding",
            "embedding_dense_backward",
            "embedding_backward",
            "take",
            "masked_select",
            "masked_scatter",
        ),
        "selection",
    ),
    **dict.fromkeys(("sort", "argsort", "topk"), "sort"),
}


_FUSED_BIAS_OPS: Final = frozenset({"addmm", "baddbmm", "addbmm"})


class _TrafficMode(TorchDispatchMode):
    """Sum the bytes of every tensor operand each aten op reads or writes.

    A view op moves nothing, and neither does an allocation that fills no
    values (``empty``); both are skipped by torch's own flags rather than a
    list, so an op added to torch is classified the way torch classifies it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bytes: defaultdict[Kernel, float] = defaultdict(float)

    @override
    def __torch_dispatch__(
        self,
        func: OpOverload[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if func.is_view or (
            func.name().split("::")[-1].split(".")[0]
            in {
                "empty",
                "empty_like",
                "empty_strided",
                "new_empty",
                "new_empty_strided",
                "empty_permuted",
            }
        ):
            return out
        name = func.name().split("::")[-1].split(".")[0]
        moved = _tensor_bytes((args, kwargs)) + _tensor_bytes(out)
        # ``addmm(bias, a, b)`` fuses the bias add into the product; the
        # analytical convention files that add, and its operand, under
        # elementwise. Split the fused op the way the unfused algorithm does.
        if name in _FUSED_BIAS_OPS:
            bias = _tensor_bytes(args[0])
            moved -= bias
            self.bytes["elementwise"] += bias
        self.bytes[_SILO_OPS.get(name, "elementwise")] += moved
        return out


def _tensor_bytes(tree: object) -> float:
    """Total bytes of every tensor leaf in ``tree``."""
    return float(
        sum(
            leaf.numel() * leaf.element_size()
            for leaf in cast(list[object], tree_leaves(tree))
            if isinstance(leaf, Tensor)
        ),
    )
