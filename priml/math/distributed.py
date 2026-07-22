"""Distributed reductions in log-space."""

from __future__ import annotations

from collections.abc import Sequence

import math

from torch import Tensor

import torch
import torch.distributed

from priml.math.custom_types import Tensorable, convert_to_tensor


def logsumexp_all_to_all(
    x: Tensorable,
    dim: int | Sequence[int] = -1,
    keepdim: bool = False,
    world_size: int | None = None,
) -> Tensor:
    """Distributed logsumexp via all_gather + local reduction.

    Returns:
      result: Global logsumexp over the specified dimensions.

    """
    return _logsumexp_all_to_all(x, dim, keepdim, world_size, mean=False)


def logmeanexp_all_to_all(
    x: Tensorable,
    dim: int | Sequence[int] = -1,
    keepdim: bool = False,
    world_size: int | None = None,
) -> Tensor:
    """Distributed logmeanexp via all_gather + local reduction.

    Returns:
      result: Global logmeanexp over the specified dimensions.

    """
    return _logsumexp_all_to_all(x, dim, keepdim, world_size, mean=True)


def _logsumexp_all_to_all(
    x: Tensorable,
    dim: int | Sequence[int] = -1,
    keepdim: bool = False,
    world_size: int | None = None,
    mean: bool = False,
) -> Tensor:
    """Distributed logsumexp via all_gather + local reduction.

    Each rank computes a local logsumexp, then all ranks exchange their
    partial results via all_gather and apply a second logsumexp to get
    the global result. Analogous to jax.lax.psum over log-space
    reductions and tfp.math.reduce_logmeanexp with cross-replica
    reduction.
    """
    x = convert_to_tensor(x)
    partial_result = torch.logsumexp(x, dim=dim, keepdim=keepdim)
    reduced = partial_result.numel()
    # An empty reduction (e.g. a zero-length batch dim) leaves nothing to
    # average; return the empty partial result instead of dividing by zero.
    if reduced == 0:
        return partial_result
    local_n = x.numel() // reduced
    if not torch.distributed.is_initialized():
        return (partial_result - math.log(local_n)) if mean else partial_result
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    gathered = [torch.empty_like(partial_result) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, partial_result)
    global_lse = torch.logsumexp(torch.stack(gathered), dim=0)
    total_n = local_n * world_size
    return (global_lse - math.log(total_n)) if mean else global_lse
