"""Validate a config's analytical ``cost`` against torch's own FLOP counter.

Builds the module, runs one forward and backward under
:class:`torch.utils.flop_counter.FlopCounterMode`, and compares
``training.flops.matmul`` per token to what torch counted, and ``params`` to
``sum(p.numel())``. The counter's registry is exactly the matmul silo (mm,
bmm, addmm, convolution, sdpa), so nothing else is validated: the other four
silos stay analytical.

A documented estimate passes ``expected_ratio``; the harness then holds the
estimate to exactly that factor of the measurement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode

import torch

from priml.cost import (
    Cost,
    cost,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from configgle import Makeable


def assert_cost_matches_torch[I: (Tensor, tuple[Tensor, ...])](
    config: Makeable[nn.Module],
    *,
    build_input: Callable[[], I],
    num_tokens: int,
    run: Callable[[nn.Module, I], Tensor] | None = None,
    expected_ratio: float = 1.0,
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
      expected_ratio: ``analytical / measured`` to hold; ``1.0`` is exact.
      seed: For ``build_input`` and any random init.
      **bus: Named arguments ``cost`` takes (``seq_len``, ``batch_size``,
        ``dtype``, ...), forwarded unchanged.

    Returns:
      analytical: The config's own cost, for further assertions.

    Raises:
      ValueError: Matmul FLOPs per token or parameter count disagree.

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
    with FlopCounterMode(display=False) as counter:
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
    measured = counter.get_total_flops() / num_tokens

    matmul = analytical["flops", "matmul"].sum()
    if matmul != expected_ratio * measured:
        raise ValueError(
            f"cost reports {matmul} matmul FLOPs/token; torch measured {measured} "
            f"(ratio {matmul / measured if measured else float('inf'):.4f}, "
            f"expected {expected_ratio})",
        )
    return analytical
