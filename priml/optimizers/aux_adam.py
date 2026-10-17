"""AdamW with a step counter shared by each parameter group."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, cast, overload, override

import math

from configgle import Fig
from torch import Tensor
from torch.optim import Optimizer
from treekle import from_plain

import torch


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class AuxAdam(Optimizer):
    """AdamW whose bias correction follows one step counter per group.

    A group advances on every optimizer call, even when some or all of its
    parameters have no gradient. Parameter moments are allocated only on the
    first call that supplies a gradient for that parameter. Updates divide the
    first moment by ``sqrt(second_moment) + eps`` before applying the scalar
    bias-correction factor ``sqrt(1 - beta2**step) / (1 - beta1**step)``.
    """

    class Config(Fig["Callable[..., AuxAdam]"]):
        """Hyperparameters; ``make`` returns a constructor awaiting parameters."""

        lr: float = 1e-4
        """Step size."""

        betas: tuple[float, float] = (0.9, 0.95)
        """First- and second-moment decay coefficients."""

        eps: float = 1e-8
        """Denominator floor."""

        weight_decay: float = 0.1
        """Decoupled weight decay."""

        @override
        def make(self) -> Callable[..., AuxAdam]:
            """Return an optimizer constructor."""
            final = self.finalized()
            return partial(
                AuxAdam,
                lr=final.lr,
                betas=final.betas,
                eps=final.eps,
                weight_decay=final.weight_decay,
            )

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, object]],
        lr: float = 1e-4,
        *,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ) -> None:
        _validate(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "step": 0,
            },
        )

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
        """Advance each group's counter and update its parameters."""
        loss: Tensor | float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for raw_group in self.param_groups:
            self._step_group(cast(dict[str, object], raw_group))
        return loss

    def _step_group(self, group: dict[str, object]) -> None:
        """Apply one update, counting every call and bias-correcting last."""
        step = from_plain(group["step"], int) + 1
        group["step"] = step
        lr = from_plain(group["lr"], float)
        beta1, beta2 = from_plain(group["betas"], tuple[float, float])
        eps = from_plain(group["eps"], float)
        weight_decay = from_plain(group["weight_decay"], float)
        for parameter in cast(list[Tensor], group["params"]):
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("AuxAdam does not support sparse gradients.")
            state = cast(dict[str, object], self.state[parameter])
            if "adamw_exp_avg" not in state:
                state["adamw_exp_avg"] = torch.zeros_like(gradient)
                state["adamw_exp_avg_sq"] = torch.zeros_like(gradient)
            first = cast(Tensor, state["adamw_exp_avg"])
            second = cast(Tensor, state["adamw_exp_avg_sq"])
            first.lerp_(gradient, 1 - beta1)
            second.lerp_(gradient.square(), 1 - beta2)
            update = first / (second.sqrt() + eps)
            correction1 = 1 - beta1**step
            correction2 = 1 - beta2**step
            parameter.mul_(1 - lr * weight_decay)
            parameter.add_(
                update,
                alpha=-(lr * correction2**0.5 / correction1),
            )


def _validate(
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> None:
    """Validate scalar hyperparameters before constructing optimizer state."""
    if not math.isfinite(lr) or lr < 0.0:
        raise ValueError(f"Learning rate must be finite and nonnegative: {lr}.")
    if not all(math.isfinite(beta) and 0.0 <= beta < 1.0 for beta in betas):
        raise ValueError(f"Betas must be finite and lie in [0, 1): {betas}.")
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError(f"Epsilon must be finite and nonnegative: {eps}.")
    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError(
            f"Weight decay must be finite and nonnegative: {weight_decay}.",
        )
