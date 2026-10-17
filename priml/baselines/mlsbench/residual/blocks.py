"""Residual blocks of the MLS-Bench ``dl-residual-connection`` baselines.

Each fills
:attr:`~priml.baselines.mlsbench.pytorch_vision.model.ResNet.Config.block`,
the slot the task's editable ``CustomBlock`` becomes. The pristine block is
:class:`~priml.baselines.mlsbench.pytorch_vision.model.BasicBlock`.

Every block registers its layers in the reference's order and every forward
keeps the reference's operations: registration order fixes the order
initialization consumes the generator, and an algebraically equal rewrite
rounds differently, so neither is a refactor here.
"""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import TYPE_CHECKING, Self, cast, override

import math

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn

import torch

from priml.baselines.mlsbench.pytorch_vision.model import (
    BasicBlock,
    BatchNorm,
    NormFn,
    built_activation,
    conv2d_cost,
    norm_cost,
)
from priml.cost import Cost, cost, elementwise_cost
from priml.model.custom_types import ActivationFn
from priml.model.swiglu import relu


if TYPE_CHECKING:
    from collections.abc import Sequence


class PreActivationBlock(nn.Module):
    """Normalize and activate before each convolution; scale the branch by a learned scalar.

    The projection shortcut reads the block's input, not its pre-activation.

    References:
      https://arxiv.org/abs/1603.05027
        He et al. 2016, "Identity Mappings in Deep Residual Networks."

    """

    class Config(Fig["PreActivationBlock"]):
        """Width, stride, activation, normalization and the branch scale's start."""

        channels_in: int = -1
        """Input channels (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        stride: int = 1
        """Stride of the first convolution and of the projection shortcut."""

        activation: ActivationFn = relu
        """Activation after each of the two normalizations."""

        norm: NormFn = BatchNorm
        """Builds each normalization: before each convolution and in the projection."""

        scale: float = 0.1
        """Value the learned multiplier of the residual branch starts at."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the normalizations, activations, convolutions, scale and sum.

            Args:
              seq_len: Positions of the block's input, a square grid.
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes and ownership.

            """
            side, side_out = _sides(seq_len, self.stride)
            c_in, c_out = self.channels_in, self.channels_out
            rows_in, rows_out = batch_size * side**2, batch_size * side_out**2
            norm_out = norm_cost(
                self.norm,
                channels=c_out,
                side=side_out,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            total = (
                norm_cost(
                    self.norm,
                    channels=c_in,
                    side=side,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + cost(self.activation, channels=c_in * rows_in, dtype=dtype)
                + conv2d_cost(
                    c_in,
                    c_out,
                    kernel_size=3,
                    side=side,
                    stride=self.stride,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + norm_out
                + cost(self.activation, channels=c_out * rows_out, dtype=dtype)
                + conv2d_cost(
                    c_out,
                    c_out,
                    kernel_size=3,
                    side=side_out,
                    stride=1,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + _scale_cost(c_out, rows_out, dtype)
                + _sum_cost(c_out, rows_out, dtype)
            )
            if self.stride != 1 or c_in != c_out:
                total += (
                    conv2d_cost(
                        c_in,
                        c_out,
                        kernel_size=1,
                        side=side,
                        stride=self.stride,
                        batch_size=batch_size,
                        dtype=dtype,
                    )
                    + norm_out
                )
            return total

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm1 = config.norm(config.channels_in)
        self.conv1 = nn.Conv2d(
            config.channels_in,
            config.channels_out,
            3,
            stride=config.stride,
            padding=1,
            bias=False,
        )
        self.norm2 = config.norm(config.channels_out)
        self.conv2 = nn.Conv2d(
            config.channels_out,
            config.channels_out,
            3,
            padding=1,
            bias=False,
        )
        self.scale = nn.Parameter(torch.tensor(config.scale))
        self.act = built_activation(config.activation)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    config.channels_in,
                    config.channels_out,
                    1,
                    stride=config.stride,
                    bias=False,
                ),
                config.norm(config.channels_out),
            )
            if config.stride != 1 or config.channels_in != config.channels_out
            else nn.Identity()
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(self.act(cast(Tensor, self.norm1(x))))
        h = self.conv2(self.act(cast(Tensor, self.norm2(h))))
        return self.shortcut(x) + self.scale * h


class GatedResidualBlock(BasicBlock):
    """A basic block whose residual branch is multiplied by one learned scalar before the sum.

    The gate starts at one, so the block starts as the basic block.

    References:
      https://arxiv.org/abs/2003.04887
        Bachlechner et al. 2021, "ReZero is All You Need: Fast Convergence at
        Large Depth" (a gate started at zero rather than one).

    """

    class Config(Makes["GatedResidualBlock"], BasicBlock.Config):
        """The basic block's fields and the gate's start."""

        gate: float = 1.0
        """Value the learned multiplier of the residual branch starts at."""

        @override
        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the basic block and the gate's multiply.

            Args:
              seq_len: Positions of the block's input, a square grid.
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes and ownership.

            """
            _, side_out = _sides(seq_len, self.stride)
            return super().cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + _scale_cost(self.channels_out, batch_size * side_out**2, dtype)

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.gate = nn.Parameter(torch.full((1,), config.gate))

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = cast(
            Tensor,
            self.norm2(self.conv2(self.act(cast(Tensor, self.norm1(self.conv1(x)))))),
        )
        return self.act(self.shortcut(x) + self.gate * h)


class StochasticDepthBlock(BasicBlock):
    """A basic block whose residual branch is dropped at random while training.

    The branch survives a training forward with probability ``1 - (depth /
    num_layers) * (1 - survival_last)``, decided by one ``torch.rand(1)``
    draw from the global generator after the shortcut runs, so every training
    forward takes exactly one draw. In evaluation the branch is scaled by that
    probability instead. The cost is the basic block's: the surviving path. As
    a ``ResNet`` template it numbers its own copies through
    :meth:`Config.place`, so one config serves any depth.

    References:
      https://arxiv.org/abs/1603.09382
        Huang et al. 2016, "Deep Networks with Stochastic Depth."

    """

    class Config(Makes["StochasticDepthBlock"], BasicBlock.Config):
        """The basic block's fields and the block's place in the survival schedule."""

        depth: int = 1
        """Position on the survival schedule, counted from 1."""

        num_layers: int = 1
        """Position that survives with ``survival_last``; usually the deepest."""

        survival_last: float = 0.5
        """Survival probability of the deepest block."""

        reset_width: int = 16
        """Width at which a stride-1 block from and to it restarts the position count.

        Upstream's own constant: its block counter resets on every block from
        16 to 16 channels at stride 1, the first block of a CIFAR ResNet."""

        def place(self, blocks: Sequence[Makeable[nn.Module]]) -> None:
            """Set each sized copy's position as upstream's block counter counts it.

            The count restarts at every stride-1 copy from and to
            ``reset_width`` channels, so in a first stage of that width every
            block is at position 1. The count's last value is every copy's
            ``num_layers``, as upstream reads it in the forward.

            Args:
              blocks: The copies, sized, in network order.

            """
            copies = [b for b in blocks if isinstance(b, StochasticDepthBlock.Config)]
            depth = 0
            for block in copies:
                if block.stride == 1 and (
                    block.channels_in == block.channels_out == self.reset_width
                ):
                    depth = 0
                depth += 1
                block.depth = depth
            for block in copies:
                block.num_layers = depth

        @override
        def finalize(self) -> Self:
            if self.num_layers < 1:
                raise ValueError(
                    f"num_layers must be at least 1; got {self.num_layers}.",
                )
            if self.depth < 1:
                raise ValueError(f"depth must be at least 1; got {self.depth}.")
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.survival = 1.0 - (config.depth / config.num_layers) * (
            1.0 - config.survival_last
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut(x)
        if not self.training:
            h = cast(
                Tensor,
                self.norm2(
                    self.conv2(self.act(cast(Tensor, self.norm1(self.conv1(x))))),
                ),
            )
            return self.act(self.survival * h + shortcut)
        # Compared as Python numbers, a float32 draw against a float64
        # probability, as the reference does; a tensor comparison would round
        # the probability to float32 first and flip draws that land between.
        if torch.rand(1).item() < self.survival:
            h = cast(
                Tensor,
                self.norm2(
                    self.conv2(self.act(cast(Tensor, self.norm1(self.conv1(x))))),
                ),
            )
            return self.act(h + shortcut)
        return self.act(shortcut)


def _sides(seq_len: int, stride: int) -> tuple[int, int]:
    """Return the side of a square input grid and of the strided output grid."""
    side = math.isqrt(seq_len)
    if side * side != seq_len:
        raise ValueError(f"A block needs a square input grid; got {seq_len}.")
    return side, (side - 1) // stride + 1


def _scale_cost(channels: int, rows: int, dtype: torch.dtype | None) -> Cost:
    """Cost multiplying a ``(rows, channels)`` branch by one learned scalar."""
    return elementwise_cost(
        primal=channels * rows,
        adjoint=2 * channels * rows,
        channels=channels,
        params=1,
        rows=rows,
        dtype=dtype,
    )


def _sum_cost(channels: int, rows: int, dtype: torch.dtype | None) -> Cost:
    """Cost adding the shortcut to the branch; the gradient passes through unchanged."""
    return elementwise_cost(
        primal=channels * rows,
        adjoint=0,
        channels=channels,
        inputs=2,
        adjoint_inputs=0,
        adjoint_outputs=0,
        rows=rows,
        dtype=dtype,
    )
