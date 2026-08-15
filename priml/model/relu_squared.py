"""Feed-forward with a squared-ReLU nonlinearity and no gate.

One matrix in and one out, where :class:`~priml.model.swiglu.SwiGLU` is
three: the nonlinearity carries what a gate otherwise would. Cheaper per
parameter, and the shape the speedrun recipes settled on.

References:
    https://arxiv.org/abs/2109.08668
      So et al. Primer: Searching for Efficient Transformer for Language
      Modeling.

"""

from __future__ import annotations

from typing import Any, override

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

from priml.model.init import InitFn, unit_fan_in_uniform
from priml.model.linear import Linear


class ReluSquared(nn.Module):
    """Feed-forward with a squared-ReLU nonlinearity and no gate.

    The output projection is zero-initialized, so a fresh block is the identity
    on its residual stream: the stack starts as shallow as the task needs and
    deepens as training proceeds.
    """

    class Config(Fig["ReluSquared"]):
        """Width, expansion, and initialization."""

        channels_in: int = -1
        """Model width; -1 inherits from the block."""

        expansion: int = 4
        """Hidden width as a multiple of ``channels_in``."""

        bias: bool = False
        """Include bias in both projections."""

        init_weight: InitFn = unit_fan_in_uniform
        """Initialization for the input projection."""

        depth: int = -1
        """Block depth index, accepted for the priml block contract."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.channels_in <= 0:
            raise ValueError(
                f"channels_in must be positive; got {config.channels_in}. It is "
                "normally inherited from the block during finalize.",
            )
        hidden = config.expansion * config.channels_in
        self.up_proj = Linear.Config(
            channels_in=config.channels_in,
            channels_out=hidden,
            bias=config.bias,
            init_weight=config.init_weight,
        ).make()
        self.down_proj = Linear.Config(
            channels_in=hidden,
            channels_out=config.channels_in,
            bias=config.bias,
            init_weight=nn.init.zeros_,
        ).make()

    def reset_parameters(self) -> None:
        """Re-initialize both projections."""
        self.up_proj.reset_parameters()
        self.down_proj.reset_parameters()

    @override
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        return self.down_proj(functional.relu(self.up_proj(x)).square())
