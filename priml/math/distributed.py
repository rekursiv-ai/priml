"""Distributed reductions in log-space."""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

from torch import Tensor

import torch
import torch.distributed as dist

from priml.math.basic import reduction_dims
from priml.memory import convert_to_tensor


if TYPE_CHECKING:
    from collections.abc import Sequence

    from priml.math.custom_types import Tensorable


def collective_device(group: dist.ProcessGroup | None = None) -> torch.device:
    """Return the device a collective on ``group`` must run on.

    NCCL reduces only CUDA tensors, on this rank's CURRENT device -- index 0
    would put every rank's tensor on one GPU. Every other backend (gloo) takes
    CPU tensors.

    Args:
      group: Process group whose backend decides; ``None`` is the default group.

    Returns:
      device: ``torch.device("cuda", current)`` for NCCL, else the CPU.

    Raises:
      ValueError: The backend is NCCL but this host has no CUDA device.

    """
    if dist.get_backend(group) != "nccl":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise ValueError(
            "NCCL backend declared but no CUDA devices are available; "
            "use gloo for CPU-only distributed training.",
        )
    return torch.device("cuda", torch.cuda.current_device())


def logsumexp_all_to_all(
    x: Tensorable,
    dim: int | Sequence[int] | None = -1,
    keepdim: bool = False,
    world_size: int | None = None,
) -> Tensor:
    """Distributed logsumexp via all_gather + local reduction.

    Args:
      x: Local tensor to reduce.
      dim: Dimension(s) over which to sum in log-space; None for all.
      keepdim: If True, reduced dimensions are kept with size 1.
      world_size: Number of ranks; if None, auto-detect from distributed setup.

    Returns:
      result: Global logsumexp over the specified dimensions.

    """
    return _logsumexp_all_to_all(x, dim, keepdim, world_size)


def logmeanexp_all_to_all(
    x: Tensorable,
    dim: int | Sequence[int] | None = -1,
    keepdim: bool = False,
    world_size: int | None = None,
) -> Tensor:
    """Distributed logmeanexp via all_gather + local reduction.

    Args:
      x: Local tensor to reduce.
      dim: Dimension(s) over which to average in log-space; None for all.
      keepdim: If True, reduced dimensions are kept with size 1.
      world_size: Number of ranks; if None, auto-detect from distributed setup.

    Returns:
      result: Global logmeanexp over the specified dimensions.

    """
    return _logsumexp_all_to_all(x, dim, keepdim, world_size, mean=True)


# Each rank computes a local logsumexp, then all ranks exchange their partial results
# via all_gather and apply a second logsumexp to get the global result. Analogous to
# jax.lax.psum over log-space reductions and tfp.math.reduce_logmeanexp with cross-
# replica reduction.
def _logsumexp_all_to_all(
    x: Tensorable,
    dim: int | Sequence[int] | None = -1,
    keepdim: bool = False,
    world_size: int | None = None,
    mean: bool = False,
) -> Tensor:
    """Distributed logsumexp via all_gather + local reduction."""
    x = convert_to_tensor(x)
    partial_result = torch.logsumexp(
        x,
        dim=reduction_dims(dim, ndim=x.ndim),
        keepdim=keepdim,
    )
    reduced = partial_result.numel()
    # An empty reduction (e.g. a zero-length batch dim) leaves nothing to
    # average; return the empty partial result instead of dividing by zero.
    if reduced == 0:
        return partial_result
    local_n = x.numel() // reduced
    if not dist.is_initialized():
        return (partial_result - math.log(local_n)) if mean else partial_result
    if world_size is None:
        world_size = dist.get_world_size()
    gathered = [torch.empty_like(partial_result) for _ in range(world_size)]
    dist.all_gather(gathered, partial_result)
    global_lse = torch.logsumexp(torch.stack(gathered), dim=0)
    total_n = local_n * world_size
    return (global_lse - math.log(total_n)) if mean else global_lse
