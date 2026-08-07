"""SignSGD optimizer for the per-puzzle sparse embedding.

Single-process port of reference's
``CastedSparseEmbeddingSignSGD_Distributed`` (`TinyRecursiveModels/
models/sparse_embedding.py`). Per-row update rule for sparse embedding
``weights`` of shape ``[N, D]``:

    if row i received gradient this step:
        p[i] <- p[i] * (1 - lr * weight_decay) - lr * sign(grad[i])
    else:
        p[i] <- p[i]                                       # no decay, no update

Reference does this from three buffers: persistent global ``weights``,
non-persistent ``local_weights`` that receives gradients, and non-persistent
``local_ids``. Only rows present in ``local_ids`` are read/decayed/written.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, overload, override

from configgle import Fig
from torch import Tensor
from torch.optim import Optimizer

import torch
import torch.distributed as dist


_ParamLike = Iterable[Tensor] | Iterable[dict[str, Any]]


class SignSGD(Optimizer):
    """SignSGD with decoupled weight decay."""

    class Config(Fig["Callable[..., SignSGD]"]):
        """SignSGD hyperparameters.

        ``make()`` yields a ``partial``, not the optimizer: a config tree has
        no parameters to hand it. Call the result with them::

            optimizer = SignSGD.Config().make()(model.parameters())
        """

        lr: float = 1e-2
        """Step size."""

        weight_decay: float = 0.0
        """Decoupled weight decay."""

        aggregate_distributed: bool = True
        """All-gather sparse-embedding gradient rows across ranks.

        Set False for TASK-parallel use, where each rank optimizes an
        independent problem: see :class:`SignSGD` for why a per-step gather
        then desyncs the job."""

        @override
        def make(self) -> Callable[..., SignSGD]:
            """Return a constructor awaiting the parameters to optimize."""
            final = (
                self.copy_tree()
                if getattr(self, "_finalized", False)
                else self.copy_tree().finalize()
            )
            return partial(
                SignSGD,
                lr=final.lr,
                weight_decay=final.weight_decay,
                aggregate_distributed=final.aggregate_distributed,
            )

    def __init__(
        self,
        params: _ParamLike,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        *,
        aggregate_distributed: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}.")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}.")
        defaults: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        # Whether the sparse-embedding step all-gathers gradient rows across
        # ranks (the data-parallel default). MUST be False for TASK-PARALLEL use
        # (e.g. per-task TTT) where each rank optimizes an INDEPENDENT problem and
        # steps a DIFFERENT number of times: a per-step ``all_gather_into_tensor``
        # then desyncs (one rank finishes while another keeps stepping with no
        # peer) and the NCCL watchdog aborts the job. False = purely local update.
        self.aggregate_distributed = aggregate_distributed

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], Tensor | float]) -> Tensor | float: ...

    @override
    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], Tensor | float] | None = None,
    ) -> Tensor | float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr: float = group["lr"]
            wd: float = group["weight_decay"]
            params = list(group["params"])
            if group.get("sparse_embedding", False):
                sparse_parts = _sparse_embedding_parts(params)
                if sparse_parts is None:
                    raise ValueError(
                        "sparse_embedding=True requires params=[weights (2D), "
                        "local_weights (2D, requires_grad), local_ids (1D)].",
                    )
                local_weights_grad, local_ids, weights = sparse_parts
                if local_weights_grad is not None:
                    _sparse_embedding_step(
                        local_weights_grad,
                        local_ids,
                        weights,
                        lr=lr,
                        weight_decay=wd,
                        aggregate_distributed=self.aggregate_distributed,
                    )
                continue
            for p in params:
                grad = p.grad
                if grad is None:
                    continue
                if self.aggregate_distributed and _is_distributed() and p.ndim >= 2:
                    _sparse_distributed_step(p, grad, lr=lr, weight_decay=wd)
                    continue
                if wd != 0.0:
                    if p.ndim >= 2:
                        # Sparse-row WD: only decay rows whose gradient has
                        # any non-zero element. ``reshape(N, -1).any(-1)``
                        # avoids materialising a full ``grad != 0`` mask
                        # (which would allocate a model-sized bool tensor).
                        touched = grad.reshape(grad.shape[0], -1).any(dim=-1)
                        # Per-row decay multiplier: (1 - lr*wd) where touched,
                        # 1.0 elsewhere. Allocate only an [N]-shaped tensor.
                        decay = touched.to(p.dtype).mul_(-lr * wd).add_(1.0)
                        p.mul_(decay.view(-1, *([1] * (p.ndim - 1))))
                    else:
                        # 1D params (biases etc.): plain decoupled WD.
                        # No "untouched row" notion in 1D.
                        p.mul_(1.0 - lr * wd)
                # ``torch.sign(grad)`` allocates a same-shape temp so the
                # caller's ``.grad`` survives intact -- downstream readers
                # (logging, clipping, a second optimizer on shared params)
                # need it. ``sign(0) = 0`` → untouched rows receive no
                # update.
                p.add_(torch.sign(grad), alpha=-lr)
        return loss

    @torch.no_grad()
    def step_sparse_embedding(
        self,
        local_weights_grad: Tensor,
        local_ids: Tensor,
    ) -> None:
        """Apply one sparse embedding update from externally accumulated rows.

        The caller's explicit invocation of this method is itself a
        signal that the matching group is a sparse-embedding group; the
        ``sparse_embedding=True`` flag is recommended but not required
        here (it IS required for routing via ``step()``).
        """
        for group in self.param_groups:
            params = list(group["params"])
            sparse_parts = _sparse_embedding_parts(params)
            if sparse_parts is None:
                continue
            _, _, weights = sparse_parts
            _sparse_embedding_step(
                local_weights_grad,
                local_ids,
                weights,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                aggregate_distributed=self.aggregate_distributed,
            )


def _is_distributed() -> bool:
    """Return whether distributed row aggregation should run."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _sparse_embedding_parts(
    params: list[Tensor],
) -> tuple[Tensor | None, Tensor, Tensor] | None:
    """Return reference sparse embedding buffers if this is that optimizer group."""
    if len(params) != 3:
        return None
    local_weights_grad: Tensor | None = None
    local_ids: Tensor | None = None
    weights: Tensor | None = None
    for p in params:
        if p.requires_grad:
            local_weights_grad = p.grad
        elif p.ndim == 1:
            local_ids = p
        elif p.ndim == 2:
            weights = p
    if local_ids is None or weights is None:
        return None
    return local_weights_grad, local_ids, weights


def _sparse_embedding_step(
    local_weights_grad: Tensor,
    local_ids: Tensor,
    weights: Tensor,
    *,
    lr: float,
    weight_decay: float,
    aggregate_distributed: bool = True,
) -> None:
    """Apply reference-style sparse embedding SignSGD.

    When ``aggregate_distributed`` is False the cross-rank gradient-row gather is
    skipped and the update is purely local -- required for task-parallel use
    (per-task TTT) where ranks step independently and a per-step collective would
    desync and trip the NCCL watchdog.
    """
    n, d = local_weights_grad.shape
    all_weights_grad = local_weights_grad
    all_ids = local_ids
    if aggregate_distributed and _is_distributed():
        world_size = dist.get_world_size()
        all_weights_grad = torch.empty(
            world_size * n,
            d,
            dtype=local_weights_grad.dtype,
            device=local_weights_grad.device,
        )
        all_ids = torch.empty(
            world_size * n,
            dtype=local_ids.dtype,
            device=local_ids.device,
        )
        grad_work = dist.all_gather_into_tensor(
            all_weights_grad,
            local_weights_grad,
            async_op=True,
        )
        ids_work = dist.all_gather_into_tensor(all_ids, local_ids, async_op=True)
        assert grad_work is not None
        assert ids_work is not None
        grad_work.wait()
        ids_work.wait()

    grad_ids, inv = all_ids.unique(return_inverse=True)
    grad = torch.zeros(
        grad_ids.shape[0],
        d,
        dtype=all_weights_grad.dtype,
        device=all_weights_grad.device,
    )
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, d), all_weights_grad)

    index_ids = grad_ids.to(torch.long)
    rows = weights[index_ids]
    rows.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)
    weights[index_ids] = rows


def _sparse_distributed_step(
    p: Tensor,
    grad: Tensor,
    *,
    lr: float,
    weight_decay: float,
) -> None:
    """Apply reference-style touched-row SignSGD across all ranks."""
    if not p.is_contiguous():
        raise RuntimeError(
            "SignSGD distributed sparse step requires a contiguous params "
            "tensor; reshape on non-contiguous storage returns a copy and "
            "the index-write below would be silently discarded.",
        )
    grad_flat = grad.reshape(grad.shape[0], -1)
    p_flat = p.reshape(p.shape[0], -1)
    touched = grad_flat.any(dim=-1)
    local_ids = touched.nonzero(as_tuple=False).flatten().to(torch.long)
    world_size = dist.get_world_size()
    count = torch.tensor([local_ids.numel()], device=p.device, dtype=torch.long)
    counts = torch.empty(world_size, device=p.device, dtype=torch.long)
    dist.all_gather_into_tensor(counts, count)
    max_count = int(counts.max().item())
    if max_count == 0:
        return

    ids_padded = torch.zeros(max_count, device=p.device, dtype=torch.long)
    grads_padded = torch.zeros(
        max_count,
        grad_flat.shape[1],
        device=p.device,
        dtype=grad.dtype,
    )
    if local_ids.numel() > 0:
        ids_padded[: local_ids.numel()] = local_ids
        grads_padded[: local_ids.numel()] = grad_flat[local_ids]

    all_ids = torch.empty(
        world_size * max_count,
        device=p.device,
        dtype=torch.long,
    )
    all_grads = torch.empty(
        world_size * max_count,
        grad_flat.shape[1],
        device=p.device,
        dtype=grad.dtype,
    )
    dist.all_gather_into_tensor(all_ids, ids_padded)
    dist.all_gather_into_tensor(all_grads, grads_padded)

    valid = torch.arange(max_count, device=p.device).expand(world_size, -1)
    valid = valid < counts.view(-1, 1)
    valid = valid.flatten()
    grad_ids, inv = all_ids[valid].unique(return_inverse=True)
    grad_rows = torch.zeros(
        grad_ids.shape[0],
        grad_flat.shape[1],
        device=p.device,
        dtype=grad.dtype,
    )
    grad_rows.scatter_add_(
        0,
        inv.unsqueeze(-1).expand(-1, grad_flat.shape[1]),
        all_grads[valid],
    )

    rows = p_flat[grad_ids]
    if weight_decay != 0.0:
        rows = rows * (1.0 - lr * weight_decay)
    rows = rows.add(torch.sign(grad_rows).to(rows.dtype), alpha=-lr)
    p_flat[grad_ids] = rows
