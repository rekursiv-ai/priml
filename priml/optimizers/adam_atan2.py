"""AdamATan2 optimizer (vendored).

Reference-compatible port of the `adam-atan2` PyPI package
(`adam-atan2==0.0.3`). The core difference from AdamW: replace the
``m / (sqrt(v) + eps)`` ratio with ``atan2(m, sqrt(v))``, which removes
the ``eps`` hyperparameter and bounds the per-element update.

Decoupled weight decay and Adam bias corrections match the reference package.

References:
  Everett et al. 2024, "Scaling Exponents Across Parameterizations
  and Optimizers." arxiv.org/abs/2407.05872 (Section 4.2 introduces
  the atan2 ratio.)

"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, cast, overload, override

from configgle import Fig
from torch import Tensor
from torch.optim import Optimizer

import torch


_ParamLike = Iterable[Tensor] | Iterable[dict[str, Any]]


class AdamATan2(Optimizer):
    """Adam variant using atan2 normalization in place of m/(sqrt(v)+eps)."""

    class Config(Fig["Callable[..., AdamATan2]"]):
        """AdamATan2 hyperparameters.

        ``make()`` yields a ``partial``, not the optimizer: a config tree has
        no parameters to hand it. Call the result with them::

            optimizer = AdamATan2.Config().make()(model.parameters())
        """

        lr: float = 1e-3
        """Step size."""

        betas: tuple[float, float] = (0.9, 0.999)
        """Decay rates for the first and second moment estimates."""

        weight_decay: float = 1e-2
        """Decoupled weight decay."""

        @override
        def make(self) -> Callable[..., AdamATan2]:
            """Return a constructor awaiting the parameters to optimize."""
            final = (
                self.copy_tree()
                if getattr(self, "_finalized", False)
                else self.copy_tree().finalize()
            )
            return partial(
                AdamATan2,
                lr=final.lr,
                betas=final.betas,
                weight_decay=final.weight_decay,
            )

    def __init__(
        self,
        params: _ParamLike,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-2,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}.")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}.")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}.")
        defaults: dict[str, Any] = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

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
        loss: Tensor | float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr: float = group["lr"]
            wd: float = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    # ``step`` was previously stored as a 0-dim CUDA tensor
                    # for foreach/grad-scaler compatibility. ``int(.item())``
                    # at every step sync'd GPU->CPU once per parameter and
                    # dominated wall-clock (~85% per profile traces). Plain
                    # Python int avoids the sync; foreach kernels aren't used
                    # here so the tensor form bought us nothing.
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                state["step"] = cast(int, state["step"]) + 1
                step = cast(int, state["step"])
                m = cast(Tensor, state["exp_avg"])
                v = cast(Tensor, state["exp_avg_sq"])
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                step_size = lr / (1.0 - beta1**step)
                bias_correction2_sqrt = (1.0 - beta2**step) ** 0.5
                update = torch.atan2(m, v.sqrt() / bias_correction2_sqrt)
                p.add_(update, alpha=-step_size)
        return loss
