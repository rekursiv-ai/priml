"""Eager moving averages of every persistent floating model state tensor."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, override

import copy
import math

from configgle import Fig

import torch


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn


class StateEMA:
    """Own a forward-capable shadow with averaged parameters and floating buffers."""

    class Config(Fig["Callable[..., StateEMA]"]):
        decay: float = 0.99
        """Fraction of each previous floating state value retained per update."""

        @override
        def make(self) -> Callable[..., StateEMA]:
            """Return a constructor awaiting the initial live model."""
            return partial(StateEMA, self.finalized())

    def __init__(self, config: Config, model: nn.Module) -> None:
        if not math.isfinite(config.decay) or config.decay < 0 or config.decay > 1:
            raise ValueError("decay must be finite and between zero and one.")
        self.config = config
        self.module = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Average floating state and copy integer state after a successful step.

        Nonpersistent buffers retain their initial snapshot. Unlike the standard
        parameter EMA, this policy also averages persistent floating buffers.

        Args:
          model: Live module with the same persistent state topology.

        """
        live = model.state_dict()
        for name, shadow in self.module.state_dict().items():
            value = live[name].detach()
            if shadow.is_floating_point():
                shadow.mul_(self.config.decay).add_(
                    value,
                    alpha=1.0 - self.config.decay,
                )
            else:
                shadow.copy_(value)
