"""AdamW as a single compiled kernel, with the bias correction folded.

Two departures from :class:`torch.optim.AdamW`, both consequences of the step
being one fused graph rather than a Python loop:

* **The second-moment correction divides before the root.** ``sqrt(v / bias2)``
  rather than ``sqrt(v) / sqrt(bias2)``: one fewer operation in the kernel, and
  the same value in exact arithmetic. In floating point the two differ in the
  last bits, so a run that swaps them is NOT bit-comparable with one that does
  not -- which is why this exists as its own optimizer instead of a flag.
* **Every scheduled scalar is a 0-D tensor.** A learning rate that arrives as a
  Python float is baked into the compiled graph, so the first step a scheduler
  changes it triggers a recompilation -- and a budgeted run that anneals its
  rate every step would then spend its budget compiling. Held as tensors, the
  graph is compiled once and the values are written into it.

The gain is wall-clock: on a run scored against a fixed time budget, optimizer
overhead comes straight out of the step count, so a fused step buys score.

References:
  https://github.com/karpathy/autoresearch
    Karpathy. autoresearch, ``adamw_step_fused`` in train.py.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import cache, partial
from typing import Any, overload, override

from configgle import Fig
from torch import Tensor
from torch.optim import Optimizer

import torch


__all__ = ["FusedAdamW"]


def _adamw_update(
    parameter: Tensor,
    gradient: Tensor,
    first_moment: Tensor,
    second_moment: Tensor,
    *,
    step: Tensor,
    lr: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    eps: Tensor,
    weight_decay: Tensor,
) -> None:
    """Apply one AdamW step in place, as a single compiled graph.

    Args:
      parameter: Weight to update, modified in place.
      gradient: Its gradient.
      first_moment: Running mean of the gradient, modified in place.
      second_moment: Running mean of its square, modified in place.
      step: 1-based step count, as a 0-D tensor.
      lr: Learning rate, as a 0-D tensor.
      beta1: Decay of the first moment, as a 0-D tensor.
      beta2: Decay of the second moment, as a 0-D tensor.
      eps: Denominator floor, as a 0-D tensor.
      weight_decay: Decoupled decay coefficient, as a 0-D tensor.

    """
    # Decoupled: the decay multiplies the weight rather than entering the
    # gradient, so it does not accumulate into either moment.
    parameter.mul_(1 - lr * weight_decay)
    first_moment.lerp_(gradient, 1 - beta1)
    second_moment.lerp_(gradient.square(), 1 - beta2)
    bias1 = 1 - beta1**step
    bias2 = 1 - beta2**step
    # Divided INSIDE the root: see the module docstring. Splitting it into
    # ``second_moment.sqrt() / bias2.sqrt()`` is the same value in exact
    # arithmetic and a different one in floating point.
    denominator = (second_moment / bias2).sqrt() + eps
    # Subtracted rather than passed through ``alpha``: the step size is a
    # tensor here, and ``alpha`` takes a Python number.
    parameter.sub_(first_moment / denominator * (lr / bias1))


@cache
def _compiled_update() -> Callable[..., None]:
    """Compile the step once, on first use.

    Deferred rather than decorated at module scope: compiling at import makes
    every importer pay for a kernel it may never step.
    """
    return torch.compile(_adamw_update, dynamic=False)


class FusedAdamW(Optimizer):
    """AdamW whose step is one compiled kernel.

    Args:
      params: Parameters or parameter groups to optimize.
      lr: Learning rate.
      betas: Decay coefficients of the first and second moments.
      eps: Denominator floor.
      weight_decay: Decoupled decay coefficient.

    Raises:
      ValueError: A hyperparameter lies outside its valid range.

    """

    class Config(Fig["Callable[..., FusedAdamW]"]):
        """Hyperparameters; see :class:`FusedAdamW` for what each one does.

        ``make()`` yields a constructor, not an optimizer: a config tree has no
        parameters to hand one. Call the result with them.
        """

        lr: float = 1e-3
        """Learning rate."""

        betas: tuple[float, float] = (0.9, 0.999)
        """Decay coefficients of the first and second moments."""

        eps: float = 1e-8
        """Denominator floor."""

        weight_decay: float = 0.0
        """Decoupled decay coefficient."""

        compile: bool = True
        """Fuse the step into one compiled graph.

        The point of this optimizer, so on by default. Turning it off is NOT
        free: a compiled step and an eager one differ in the last bits, so a
        run with this off is not bit-comparable with one that has it on. It
        exists for a run too short to amortize the compile, which is charged to
        the first step that uses it."""

        @override
        def make(self) -> Callable[..., FusedAdamW]:
            """Return a constructor awaiting the parameters to optimize."""
            final = (
                self.copy_tree()
                if getattr(self, "_finalized", False)
                else self.copy_tree().finalize()
            )
            return partial(
                FusedAdamW,
                lr=final.lr,
                betas=final.betas,
                eps=final.eps,
                weight_decay=final.weight_decay,
                compile=final.compile,
            )

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        compile: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}.")
        if not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"betas must lie in [0, 1); got {betas}.")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}.")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}.")
        super().__init__(
            params,
            {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay},
        )
        # One set of 0-D CPU tensors reused by every group and every step. A
        # Python float would be a compile-time constant, so a scheduler moving
        # the rate would retrigger compilation on the step it moved -- and a
        # budgeted run anneals its rate on EVERY step.
        self._scalars: dict[str, Tensor] = {
            name: torch.zeros((), dtype=torch.float32)
            for name in ("step", "lr", "beta1", "beta2", "eps", "weight_decay")
        }
        self._update: Callable[..., None] = (
            _compiled_update() if compile else _adamw_update
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
        """Apply one update to every parameter holding a gradient."""
        loss: Tensor | float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                self._step_parameter(parameter, gradient, group)
        return loss

    def _step_parameter(
        self,
        parameter: Tensor,
        gradient: Tensor,
        group: dict[str, Any],
    ) -> None:
        """Update one parameter, filling the shared scalars from its group."""
        state = self.state[parameter]
        if not state:
            state["step"] = 0
            state["first_moment"] = torch.zeros_like(parameter)
            state["second_moment"] = torch.zeros_like(parameter)
        state["step"] += 1
        beta1, beta2 = group["betas"]
        for name, value in (
            ("step", state["step"]),
            ("lr", group["lr"]),
            ("beta1", beta1),
            ("beta2", beta2),
            ("eps", group["eps"]),
            ("weight_decay", group["weight_decay"]),
        ):
            self._scalars[name].fill_(value)
        self._update(
            parameter,
            gradient,
            state["first_moment"],
            state["second_moment"],
            **self._scalars,
        )
