"""Distributed-correct global gradient-norm clipping.

A single helper, :func:`clip_grad_norm_`, that clips the *global* gradient norm
across the whole model and -- crucially -- across the parallelism mesh, using
PyTorch's ``get_total_norm`` / ``clip_grads_with_norm_`` primitives and the
DTensor ``full_tensor()`` all-reduce. This follows torchtitan's reference
``clip_grad_norm_`` (``torchtitan/distributed/utils.py``).

Why not plain ``torch.nn.utils.clip_grad_norm_``: that computes the norm over
whatever tensors it is handed. For sharded parameters (TP/FSDP -> DTensor
grads) the per-rank norm is only a *shard* of the true global norm, so clipping
on it is wrong. The DTensor total-norm carries a ``_NormPartial`` placement;
``full_tensor()`` performs the cross-mesh reduction that yields the true global
norm. For pure DDP (replicated, already-all-reduced grads) the grads are plain
tensors and the local norm already equals the global norm -- this helper
degrades to the standard computation, so it is correct in both regimes.

Pipeline (PP) and expert (EP) parallelism need extra cross-stage reductions
(see torchtitan); we do not use PP/EP, so they are intentionally omitted. If
they are ever added, port torchtitan's ``pp_mesh`` / EP branches here.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor
from torch.distributed.tensor import DTensor

import torch


@torch.no_grad()
def clip_grad_norm_(
    parameters: Tensor | Iterable[Tensor],
    max_norm: float,
    *,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = True,
) -> Tensor:
    """Clip the global gradient norm across the whole model and mesh.

    Computes one total norm over every parameter gradient, reduces it across the
    parallelism mesh when grads are sharded (DTensor), and rescales all grads
    in-place so the global norm is at most ``max_norm``.

    Args:
      parameters: A parameter/tensor or iterable thereof whose ``.grad`` is
        clipped. Tensors without a gradient are skipped for the norm but still
        passed to the rescale (a no-op for them).
      max_norm: Maximum allowed global gradient norm.
      norm_type: p-norm order (``float('inf')`` for the max norm).
      error_if_nonfinite: Raise if the total norm is non-finite.
      foreach: Use the fused foreach kernels (faster). ``True`` (default) forces
        them; ``None`` lets PyTorch pick per device type; ``False`` disables
        them. Set this to ``None`` or ``False`` on the MPS backend, which has no
        foreach kernels (``True`` raises "can't use the foreach API on mps
        tensors"). CUDA/CPU keep the fast ``True`` path.

    Returns:
      total_norm: The global pre-clip gradient norm as a plain (mesh-reduced)
        tensor, suitable for ``.item()`` logging.

    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    else:
        # Materialize: the generator is consumed twice (norm, then rescale).
        parameters = list(parameters)

    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )

    # Sharded grads (TP/FSDP) yield a DTensor total norm with _NormPartial
    # placement; full_tensor() all-reduces it across the mesh to the true global
    # norm. Replicated DDP grads yield a plain tensor and skip this.
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def total_grad_norm(
    parameters: Tensor | Iterable[Tensor],
    *,
    norm_type: float = 2.0,
    foreach: bool | None = True,
) -> Tensor:
    """Return the global gradient norm without clipping (mesh-reduced).

    Same distributed-correct computation as :func:`clip_grad_norm_` but
    measure-only -- for logging ``grad_norm`` on steps where clipping is off.

    Args:
      parameters: A parameter/tensor or iterable thereof.
      norm_type: p-norm order.
      foreach: Use the fused foreach kernels. Set ``None``/``False`` on MPS
        (no foreach kernels there); ``True`` is the fast CUDA/CPU path.

    Returns:
      total_norm: The global gradient norm as a plain (mesh-reduced) tensor.

    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(grads, norm_type, False, foreach)
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()
    return total_norm
