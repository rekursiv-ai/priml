"""Muon optimizer: SGD with momentum in the orthogonal manifold."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal, override

import math

from torch import Tensor, nn

import torch

from priml.math.numeric import matrix_signum_via_newtonschulz


class Muon(torch.optim.Optimizer):
    """Muon optimizer: SGD with momentum in the orthogonal manifold.

    Uses Newton-Schulz iteration to project gradients onto the
    orthogonal group, achieving weight-scale invariance.

    Only defined for parameters with ndim >= 2 (linear operators).

    References:
      Bernstein & Wan 2024, "Old Optimizer, New Norm."
      https://kellerjordan.github.io/posts/muon/
      https://github.com/KellerJordan/Muon

    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        weight_decay: float | None = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: Literal[
            "original",
            "match_rms_adamw",
            "conv_heuristic",
        ] = "original",
        ensemble_dims: int = 0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay is not None and weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires momentum > 0")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
            "ensemble_dims": ensemble_dims,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    @override
    def step(self, closure: Callable[[], float] | None = None) -> None:  # ty: ignore[invalid-method-override] -- narrows Optimizer.step return to None; ty rejects the narrowing
        del closure
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_coefficients = group["ns_coefficients"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            adjust_lr_fn = group["adjust_lr_fn"]
            ensemble_dims = group["ensemble_dims"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.ndim < 2:
                    raise ValueError(
                        f"Muon requires ndim >= 2, got shape {p.shape}.",
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - momentum)
                g = g.lerp(buf, momentum) if nesterov else buf

                msgn_g = matrix_signum_via_newtonschulz(
                    g.reshape(
                        math.prod(g.shape[:ensemble_dims]),
                        g.shape[ensemble_dims],
                        -1,
                    ),
                    coefficients=ns_coefficients,
                    steps=ns_steps,
                    eps=eps,
                ).reshape(g.shape)

                if weight_decay is not None:
                    p.data.mul_(1 - lr * weight_decay)

                adjusted_lr = _adjust_lr(lr, p, adjust_lr_fn, ensemble_dims)
                if isinstance(adjusted_lr, Tensor):
                    p.data.add_(msgn_g * (-adjusted_lr))
                else:
                    p.data.add_(msgn_g, alpha=-adjusted_lr)


def _adjust_lr(
    lr: float,
    param: nn.Parameter,
    adjust_lr_fn: str,
    ensemble_dims: int = 0,
) -> float | Tensor:
    """Per-parameter learning rate adjustment for Muon.

    Returns float for static adjustments, Tensor for data-dependent
    ones (conv_heuristic) to avoid GPU→CPU sync.
    """
    c_out = param.shape[ensemble_dims]
    c_in = math.prod(param.shape[ensemble_dims + 1 :])
    if adjust_lr_fn == "original":
        return lr * max(1, c_out / c_in) ** 0.5
    if adjust_lr_fn == "match_rms_adamw":
        return lr * 0.2 * max(c_out, c_in) ** 0.5
    if adjust_lr_fn == "conv_heuristic":
        # Keep on GPU: norm() returns 0-dim tensor, avoid .item() sync.
        return lr * param.data.norm() / c_out**0.5
    raise ValueError(f"Unknown adjust_lr_fn: {adjust_lr_fn}")
