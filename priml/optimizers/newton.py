"""Custom optimizers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, ClassVar, override

from configgle import Fig
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import Optimizer

import torch
import torch.linalg


def compute_hessian(
    loss: torch.Tensor,
    params: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient and Hessian via autograd.

    Args:
        loss: Scalar loss tensor (must have grad enabled).
        params: List of parameter tensors.

    Returns:
        grad_tensor: Flattened gradient vector.
        hessian: Hessian matrix (n_params x n_params).

    """
    # Get gradients with create_graph=True for second derivatives
    grads_list = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grad_tensor = torch.cat([g.view(-1) for g in grads_list])
    n_params = grad_tensor.shape[0]

    # Compute Hessian: d(grad)/d(params). Match the gradient dtype so the
    # subsequent linear solve does not fail on float64 (or other) parameters.
    hessian = torch.zeros(
        n_params,
        n_params,
        device=grad_tensor.device,
        dtype=grad_tensor.dtype,
    )
    for i in range(n_params):
        # Compute gradient of grad[i] w.r.t. all params
        retain = i < n_params - 1
        hess_row = torch.autograd.grad(
            grad_tensor[i],
            params,
            retain_graph=retain,
            allow_unused=True,
        )
        hess_row_flat: Tensor = torch.cat(
            [
                h.view(-1) if h is not None else torch.zeros_like(p.view(-1))  # pyright: ignore[reportUnnecessaryComparison]
                for h, p in zip(hess_row, params, strict=True)
            ],
        )
        hessian[i] = hess_row_flat

    return grad_tensor, hessian


class Newton(Optimizer):
    """Newton's method optimizer using exact Hessian.

    Computes Hessian via autograd and uses Newton step: x -= lr * H^{-1} @ g
    Only suitable for small models where Hessian computation is tractable.

    Requires a closure to recompute the loss with a fresh graph;
    ``TrainStep.train_step`` builds and forwards one automatically (via
    ``TrainStep.step`` -> ``optimizer.step(closure)``), so Newton trains
    through the normal path with no hand-rolled loop.

    Memory requirements: O(n^2) where n is number of parameters.
    For 10k parameters: ~400MB. For 100k parameters: ~40GB. For 1M parameters: ~4TB.
    Only practical for very small models (< 10k parameters).
    """

    requires_closure: ClassVar[bool] = True
    """Newton needs the loss-recomputing closure to build the Hessian."""

    class Config(Fig["Callable[..., Newton]"]):
        """Newton hyperparameters; see :class:`Newton` for what each one does.

        ``make()`` yields a ``partial``, not a ``Newton``: a config tree has no
        parameters to hand an optimizer. Call the result with them::

            optimizer = Newton.Config().make()(model.parameters())
        """

        lr: float = 1.0
        """Step size multiplier on the Newton direction."""

        damping: float = 1e-5
        """Added to the Hessian diagonal so a near-singular solve stays stable."""

        @override
        def make(self) -> Callable[..., Newton]:
            """Return a constructor awaiting the parameters to optimize."""
            final = (
                self.copy_tree()
                if getattr(self, "_finalized", False)
                else self.copy_tree().finalize()
            )
            return partial(Newton, lr=final.lr, damping=final.damping)

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1.0,
        *,
        damping: float = 1e-5,
    ) -> None:
        """Initialize Newton optimizer.

        Args:
            params: Parameters to optimize, or parameter groups.
            lr: Learning rate (step size multiplier).
            damping: Damping factor added to Hessian diagonal for stability.

        """
        super().__init__(params, {"lr": lr, "damping": damping})

    @override
    def step(  # ty: ignore[invalid-method-override] -- single-signature override of torch's overloaded `Optimizer.step`; the union return matches `OptimizerProtocol`
        self,
        closure: Callable[[], Tensor | float] | None = None,
    ) -> Tensor | float | None:
        """Perform single Newton step.

        Newton requires the loss-recomputing ``closure`` (``requires_closure``
        is set): ``autograd.grad`` needs a graph-bearing loss to build the
        Hessian, so the training loop always forwards one. The optional
        signature matches ``OptimizerProtocol``; a missing closure is a caller
        error, not a no-op.

        Args:
            closure: Callable that recomputes the loss (without backward).

        Returns:
            loss: Loss value after the step.

        Raises:
            ValueError: If ``closure`` is not provided.

        """
        if closure is None:
            raise ValueError(
                "Newton.step requires a loss-recomputing closure to build the "
                "Hessian; the training loop forwards one for optimizers whose "
                "requires_closure is set.",
            )
        loss = None
        for group in self.param_groups:
            lr = group["lr"]
            damping = group["damping"]

            # Compute Hessian via autograd. The closure must return a
            # graph-bearing Tensor (a float loss carries no graph for
            # ``autograd.grad`` to differentiate).
            with torch.enable_grad():
                loss = closure()
                if not isinstance(loss, Tensor):
                    raise TypeError(
                        "Newton.step requires a closure returning a graph-bearing "
                        f"Tensor loss; got {type(loss).__name__}.",
                    )
                param_list = [p for p in group["params"] if p.requires_grad]
                # A sharded DTensor would flatten to its LOCAL shard inside
                # compute_hessian, building a global Hessian from inconsistent
                # partial vectors. Newton is an O(n^2)-Hessian small-model
                # optimizer that cannot be correct under sharding, so refuse
                # rather than silently corrupt the step.
                if any(isinstance(p, DTensor) for p in param_list):
                    raise NotImplementedError(
                        "Newton does not support sharded (DTensor) parameters: "
                        "the exact-Hessian step is global and cannot be computed "
                        "from local shards. Use a single-device (replicated) "
                        "parameter set.",
                    )
                grad_tensor, hessian = compute_hessian(loss, param_list)

            # Add damping for numerical stability.
            n_params = grad_tensor.shape[0]
            hessian = hessian + damping * torch.eye(
                n_params,
                device=grad_tensor.device,
                dtype=grad_tensor.dtype,
            )

            # Solve H @ delta = -grad_tensor for Newton direction. Only a
            # singular Hessian (LinAlgError) warrants the gradient-descent
            # fallback; other RuntimeErrors are genuine bugs and must surface.
            delta: Tensor
            try:
                delta = torch.linalg.solve(hessian, -grad_tensor)
            except torch.linalg.LinAlgError:
                delta = -grad_tensor

            # Update parameters (outside autograd)
            with torch.no_grad():
                offset = 0
                for p in param_list:
                    numel = p.numel()
                    p.add_(delta[offset : offset + numel].view_as(p), alpha=lr)
                    offset += numel

        if loss is None:
            raise RuntimeError("Newton optimizer requires at least one parameter group")
        return loss
