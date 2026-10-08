"""Adam behind a clip of the global gradient norm, as Craftax_Baselines chains them.

The reference builds its optimizer as ``optax.chain(clip_by_global_norm(max),
adam(lr, eps))``: every gradient is scaled by ``min(1, max / norm)``, the norm
taken over all of them at once, before Adam's bias-corrected step.
:class:`ClippedAdam` is torch's Adam with that clip in front of its step.

References:
    https://github.com/MichaelTMatthews/Craftax_Baselines
        Matthews et al. Craftax_Baselines (MIT license), ``ppo.py``, commit
        ``7ce36fa``.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload, override

import math

import torch


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import Tensor


class ClippedAdam(torch.optim.Adam):
    """Adam whose step first clips every gradient by their global L2 norm.

    On a CUDA device it is capturable -- its step count lives on the device
    and it takes a device-tensor rate -- so a train step can record its step
    in a CUDA graph and refill the rate between replays. Torch's Adam refuses
    capturable parameters on the CPU, where it runs uncaptured.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float,
        max_grad_norm: float,
        eps: float = 1e-8,
        betas: tuple[float, float] = (0.9, 0.999),
    ) -> None:
        """Build Adam over ``params`` with the clip in front.

        Args:
          params: The weights, in the order the global norm reduces them.
          lr: The step size; a schedule may rewrite it per group.
          max_grad_norm: The global norm the gradients are clipped to.
          eps: Adam's floor, added to the root of the second moment.
          betas: Adam's moment decays.

        Raises:
          ValueError: ``max_grad_norm`` is not positive and finite.

        """
        if math.isnan(max_grad_norm) or math.isinf(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be positive and finite, not {max_grad_norm}",
            )
        parameters = list(params)
        super().__init__(
            parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            capturable=all(parameter.is_cuda for parameter in parameters),
        )
        # The clip is the run's code, not its state: it stays off the groups a
        # checkpoint serializes, as FusedMuon keeps its own.
        self.max_grad_norm = max_grad_norm

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], Tensor | float]) -> Tensor | float: ...

    @override
    def step(
        self,
        closure: Callable[[], Tensor | float] | None = None,
    ) -> Tensor | float | None:
        """Clip every gradient by the global norm, then take Adam's step.

        Args:
          closure: Re-evaluates the loss, as ``torch.optim.Adam.step`` takes it.

        Returns:
          loss: The closure's loss, or None without one.

        """
        loss = None
        if closure is not None:
            # The loss and its gradients first: the clip reads those gradients.
            with torch.enable_grad():
                loss = closure()
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for group in self.param_groups
                for parameter in cast("list[Tensor]", group["params"])
            ],
            self.max_grad_norm,
        )
        super().step()
        return loss
