"""NorMuon: orthogonalized momentum with a per-row second-moment correction.

Muon replaces a weight matrix's update with the nearest orthogonal matrix, so
the step is invariant to the weight's scale. Two modifications matter enough at
language-model scale to be worth their own optimizer rather than arguments to
:class:`~priml.optimizers.muon.Muon`:

* **Polar express.** The orthogonalization is a fixed five-term polynomial
  iteration with per-iteration coefficients, rather than one repeated
  Newton-Schulz triple. The coefficients are tuned so the first iterations move
  fast and the last ones land accurately, which reaches a usable orthogonal
  factor in five steps where the repeated triple needs more.
* **NorMuon.** An orthogonal update spends the same magnitude on every row,
  which is wrong when rows differ in how much signal they carry. A second
  moment per row rescales them, then the whole matrix is renormalized back to
  the orthogonal update's norm -- so the correction redistributes the step
  without changing its size.

Weight decay is CAUTIOUS: it applies only where the update and the weight point
the same way, so decay never fights an update that is already shrinking a
weight.

Parameters of one shape are stacked and stepped as a single batched tensor. A
language model has many identically-shaped matrices, and one batched
orthogonalization over them is far faster than a loop -- which is why the
grouping is by shape and happens here rather than in the caller's recipe.

References:
    https://arxiv.org/abs/2505.16932
      Amsel et al. The Polar Express: Optimal Matrix Sign Methods.
    https://arxiv.org/abs/2510.05491
      Zhang et al. NorMuon: Making Muon more efficient and scalable.
    https://kellerjordan.github.io/posts/muon/
      Jordan et al. 2024. Muon: an optimizer for hidden layers.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, overload, override

from configgle import Fig
from torch import Tensor
from torch.optim import Optimizer

import torch


if TYPE_CHECKING:
    from torch.nn import Parameter


class NorMuon(Optimizer):
    """Orthogonalized momentum with row-wise second-moment rescaling.

    Defined only for parameters of rank at least two -- orthogonalizing a
    vector is meaningless -- which :meth:`eligible_tensor` states so a recipe
    can route by it.

    Args:
      params: Parameters, or parameter groups, to optimize.
      lr: Step size, before the tall-matrix correction.
      momentum: Coefficient on the momentum buffer.
      beta2: Decay of the per-row second moment.
      ns_steps: Polynomial iterations, at most ``len(coefficients)``.
      weight_decay: Decoupled decay, applied only where it agrees with the
        update.
      coefficients: Polynomial iteration coefficients.

    """

    @classmethod
    def eligible_tensor(cls, name: str, parameter: Parameter) -> bool:
        """Whether this optimizer is defined on the given parameter.

        Args:
          name: Qualified parameter name, as ``named_parameters`` reports it.
          parameter: The parameter itself.

        Returns:
          eligible: True for a rank >= 2 weight.

        """
        del cls, name
        return parameter.ndim >= 2

    class Config(Fig["Callable[..., NorMuon]"]):
        """Hyperparameters; see :class:`NorMuon` for what each one does.

        ``make()`` yields a constructor, not an optimizer: a config tree has no
        parameters to hand one. Call the result with them.
        """

        lr: float = 0.04
        """Step size, before the tall-matrix correction."""

        momentum: float = 0.95
        """Coefficient on the momentum buffer."""

        beta2: float = 0.95
        """Decay of the per-row second moment."""

        ns_steps: int = 5
        """Polynomial iterations per update."""

        weight_decay: float = 0.2
        """Decoupled decay, applied only where it agrees with the update."""

        coefficients: tuple[tuple[float, float, float], ...] = (
            (8.156554524902461, -22.48329292557795, 15.878769915207462),
            (4.042929935166739, -2.808917465908714, 0.5000178451051316),
            (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
            (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
            (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
        )
        """Per-iteration ``(a, b, c)`` of ``a x + b x G + c x G^2``, ``G = x^T x``.

        A property of the polynomial iteration rather than a tunable: the five
        triples are jointly optimized for a five-step schedule, so running a
        PREFIX of them is meaningful (that is ``ns_steps``) while editing one is
        not. Immutable, and its length is part of the schedule's shape."""

        @override
        def make(self) -> Callable[..., NorMuon]:
            """Return a constructor awaiting the parameters to optimize."""
            final = (
                self.copy_tree()
                if getattr(self, "_finalized", False)
                else self.copy_tree().finalize()
            )
            return partial(
                NorMuon,
                lr=final.lr,
                momentum=final.momentum,
                beta2=final.beta2,
                ns_steps=final.ns_steps,
                weight_decay=final.weight_decay,
                coefficients=final.coefficients,
            )

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 0.04,
        *,
        momentum: float = 0.95,
        beta2: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.2,
        coefficients: tuple[tuple[float, float, float], ...] = (
            (8.156554524902461, -22.48329292557795, 15.878769915207462),
            (4.042929935166739, -2.808917465908714, 0.5000178451051316),
            (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
            (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
            (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
        ),
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must lie in [0, 1); got {momentum}.")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must lie in [0, 1); got {beta2}.")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}.")
        if not 1 <= ns_steps <= len(coefficients):
            raise ValueError(
                f"ns_steps must lie in [1, {len(coefficients)}]; got {ns_steps}.",
            )
        super().__init__(
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "beta2": beta2,
                "ns_steps": ns_steps,
                "weight_decay": weight_decay,
                "coefficients": coefficients,
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
        """Apply one update to every parameter holding a gradient."""
        loss: Tensor | float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for shape_params in _by_shape(group["params"]):
                self._step_shape(shape_params, group)
        return loss

    def _step_shape(self, params: list[Tensor], group: dict[str, Any]) -> None:
        """Update one batch of identically-shaped parameters together."""
        shape = params[0].shape
        if len(shape) < 2:
            raise ValueError(f"NorMuon requires ndim >= 2; got shape {tuple(shape)}.")
        state = self.state[params[0]]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(
                len(params),
                *shape,
                dtype=params[0].dtype,
                device=params[0].device,
            )
            # The second moment is per ROW of the update when the matrix is
            # tall and per column otherwise, so its buffer collapses whichever
            # axis the mean reduces.
            tall = shape[-2] >= shape[-1]
            state["second_moment"] = torch.zeros(
                (len(params), shape[-2], 1) if tall else (len(params), 1, shape[-1]),
                dtype=params[0].dtype,
                device=params[0].device,
            )
        stacked_grads = torch.stack([_gradient(p) for p in params])
        stacked_params = torch.stack(list(params))
        _norvmuon_update(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_moment"],
            momentum=group["momentum"],
            # A tall matrix's orthogonal update has more rows than it has
            # independent directions, so its step is scaled to match a square
            # one's per-element magnitude.
            lr=group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5,
            weight_decay=group["weight_decay"],
            beta2=group["beta2"],
            ns_steps=group["ns_steps"],
            coefficients=group["coefficients"],
        )
        torch._foreach_copy_(list(params), list(stacked_params.unbind(0)))


def _by_shape(params: list[Tensor]) -> list[list[Tensor]]:
    """Bucket parameters by shape, in first-appearance order."""
    buckets: dict[tuple[int, ...], list[Tensor]] = {}
    for parameter in params:
        if parameter.grad is None:
            continue
        buckets.setdefault(tuple(parameter.shape), []).append(parameter)
    return list(buckets.values())


def _gradient(parameter: Tensor) -> Tensor:
    """Return a parameter's gradient, which ``_by_shape`` guaranteed exists."""
    grad = parameter.grad
    assert grad is not None
    return grad


def _norvmuon_update(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_moment: Tensor,
    *,
    momentum: float,
    lr: float,
    weight_decay: float,
    beta2: float,
    ns_steps: int,
    coefficients: tuple[tuple[float, float, float], ...],
) -> None:
    """Apply momentum, orthogonalize, rescale rows, and decay, in place.

    Args:
      stacked_grads: ``[N, R, C]`` gradients; consumed destructively.
      stacked_params: ``[N, R, C]`` weights, updated in place.
      momentum_buffer: ``[N, R, C]`` running gradient average.
      second_moment: ``[N, R, 1]`` or ``[N, 1, C]`` running row energy.
      momentum: Coefficient on the momentum buffer.
      lr: Step size, already corrected for a tall matrix.
      weight_decay: Decoupled decay, applied only where it agrees.
      beta2: Decay of the second moment.
      ns_steps: Polynomial iterations to run.
      coefficients: Polynomial iteration coefficients.

    """
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    update = stacked_grads.lerp_(momentum_buffer, momentum)

    # Orthogonalization runs in bfloat16: the iteration is self-correcting, so
    # its intermediate precision does not reach the result, and the matmuls
    # dominate the step's cost.
    x = update.bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if x.size(-2) > x.size(-1):
        for a, b, c in coefficients[:ns_steps]:
            gram = x.mT @ x
            x = a * x + x @ (b * gram + c * (gram @ gram))
    else:
        for a, b, c in coefficients[:ns_steps]:
            gram = x @ x.mT
            x = a * x + (b * gram + c * (gram @ gram)) @ x

    reduce_dim = -1 if x.size(-2) >= x.size(-1) else -2
    row_energy = x.float().square().mean(dim=reduce_dim, keepdim=True)
    width = x.size(reduce_dim)
    before = (row_energy.sum(dim=(-2, -1), keepdim=True) * width).sqrt()
    second_moment.lerp_(row_energy.to(second_moment.dtype), 1 - beta2)
    scale = second_moment.clamp_min(1e-10).rsqrt()
    after = (
        ((row_energy * width) * scale.float().square())
        .sum(dim=(-2, -1), keepdim=True)
        .sqrt()
    )
    # Renormalize to the orthogonal update's own norm: the row rescaling is
    # meant to REDISTRIBUTE the step, not to resize it.
    x = x * (scale * (before / after.clamp_min(1e-10))).to(x.dtype)

    agrees = (x * stacked_params) >= 0
    stacked_params.sub_(lr * x + lr * weight_decay * stacked_params * agrees)
