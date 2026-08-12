"""Muon optimizer: SGD with momentum in the orthogonal manifold."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, override

import math

from configgle import Fig
from torch import Tensor, nn
from torch.optim import Optimizer

import torch

from priml.math.numeric import matrix_signum_via_newtonschulz


if TYPE_CHECKING:
    from torch.nn import Parameter


class Muon(Optimizer):
    """Muon optimizer: SGD with momentum in the orthogonal manifold.

    Uses Newton-Schulz iteration to project gradients onto the
    orthogonal group, achieving weight-scale invariance.

    Only defined for parameters with ndim >= 2 (linear operators).

    References:
      Bernstein & Wan 2024, "Old Optimizer, New Norm."
      https://kellerjordan.github.io/posts/muon/
      https://github.com/KellerJordan/Muon

    """

    @classmethod
    def eligible_tensor(cls, name: str, parameter: Parameter) -> bool:
        """Whether Muon is defined on this parameter.

        Muon orthogonalizes each update, which is meaningful for a linear
        operator and undefined for the 1-D scales and biases beside it --
        :meth:`step` raises on anything lower-rank. This is the algorithm's own
        constraint; a recipe that also wants the classifier head left out
        composes ``excluding(Muon.eligible_tensor, "head")``.

        Args:
          name: Qualified parameter name, as ``named_parameters`` reports it.
          parameter: The parameter itself.

        Returns:
          eligible: True for a rank >= 2 weight.

        """
        del cls, name
        return parameter.ndim >= 2

    class Config(Fig["Callable[..., Muon]"]):
        """Muon hyperparameters; see :class:`Muon` for what each one does.

        ``make()`` yields a ``partial``, not a ``Muon``: a config tree has no
        parameters to hand an optimizer. Call the result with them::

            optimizer = Muon.Config().make()(model.parameters())
        """

        lr: float = 1e-3
        """Step size."""

        weight_decay: float | None = 0.1
        """Decoupled weight decay; None disables it."""

        momentum: float = 0.95
        """Momentum coefficient on the update."""

        nesterov: bool = True
        """Look ahead by the momentum term before orthogonalizing."""

        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
        """Newton-Schulz polynomial coefficients."""

        eps: float = 1e-7
        """Floor on the gradient norm used to normalize before iterating."""

        ns_steps: int = 5
        """Newton-Schulz iterations per update."""

        adjust_lr_fn: Literal["original", "match_rms_adamw", "conv_heuristic"] = (
            "original"
        )
        """How the step is rescaled for a parameter's shape."""

        ensemble_dims: int = 0
        """Leading dimensions treated as an ensemble, not as matrix axes."""

        @override
        def make(self) -> Callable[..., Muon]:
            """Return a constructor awaiting the parameters to optimize."""
            final = (
                self.copy_tree()
                if getattr(self, "_finalized", False)
                else self.copy_tree().finalize()
            )
            return partial(
                Muon,
                lr=final.lr,
                weight_decay=final.weight_decay,
                momentum=final.momentum,
                nesterov=final.nesterov,
                ns_coefficients=final.ns_coefficients,
                eps=final.eps,
                ns_steps=final.ns_steps,
                adjust_lr_fn=final.adjust_lr_fn,
                ensemble_dims=final.ensemble_dims,
            )

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        *,
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
        return lr * math.sqrt(max(1, c_out / c_in))
    if adjust_lr_fn == "match_rms_adamw":
        return lr * 0.2 * math.sqrt(max(c_out, c_in))
    if adjust_lr_fn == "conv_heuristic":
        # Keep on GPU: norm() returns 0-dim tensor, avoid .item() sync.
        return lr * param.data.norm() / math.sqrt(c_out)
    raise ValueError(f"Unknown adjust_lr_fn: {adjust_lr_fn}")
