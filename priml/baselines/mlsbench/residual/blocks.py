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

from priml.baselines.mlsbench.pytorch_vision.model import BasicBlock, NormFn
from priml.cost import Cost, cost, elementwise_cost
from priml.math.custom_types import TensorFn
from priml.model.conv import conv_cost
from priml.model.custom_types import ActivationFn
from priml.model.norm import BatchNorm2d
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

        norm: NormFn = nn.BatchNorm2d
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
            norm_out = _norm_cost(c_out, side_out, batch_size, dtype, **kwargs)
            total = (
                _norm_cost(c_in, side, batch_size, dtype, **kwargs)
                + cost(self.activation, channels=c_in * rows_in, dtype=dtype)
                + _conv_cost(
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
                + _conv_cost(
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
                    _conv_cost(
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
        self.act = _built(config.activation)
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
    probability instead. The cost is the basic block's: the surviving path.

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


def stochastic_depth(
    channels_hidden: Sequence[int],
    blocks_per_stage: Sequence[int],
    *,
    survival_last: float,
) -> list[Makeable[nn.Module]]:
    """Return one stochastic-depth block per position of a network, in order.

    Positions are numbered as the reference numbers them: the count restarts
    at every stride-1 block whose input and output both have the stem's width,
    so in a first stage as wide as the stem every block is at depth 1, and
    counting resumes from 2 in the second stage. The count's last value sets
    every block's ``num_layers``, as the reference reads it in the forward.

    Args:
      channels_hidden: Width of each stage; the first is the stem's.
      blocks_per_stage: Blocks in each stage.
      survival_last: Survival probability of the deepest block.

    Returns:
      blocks: ``sum(blocks_per_stage)`` configs for ``ResNet.Config.block``.

    """
    depths: list[int] = []
    depth = 0
    channels = channels_hidden[0]
    for index, (width, count) in enumerate(
        zip(channels_hidden, blocks_per_stage, strict=True),
    ):
        for position in range(count):
            # The stride ResNet gives this position: 2 on each later stage's first.
            stride = 2 if index and not position else 1
            if stride == 1 and channels == width == channels_hidden[0]:
                depth = 0
            depth += 1
            depths.append(depth)
            channels = width
    blocks: list[Makeable[nn.Module]] = []
    for position_depth in depths:
        block = StochasticDepthBlock.Config()
        block.depth = position_depth
        block.num_layers = depth
        block.survival_last = survival_last
        blocks.append(block)
    return blocks


def _built(activation: ActivationFn) -> TensorFn:
    """Build an activation from a config, or pass a callable through."""
    if isinstance(activation, Makeable):
        # ``Makeable`` is runtime-checkable, so isinstance erases its type
        # parameter: ``make`` reads as returning ``object`` without the cast.
        return cast(TensorFn, activation.make())
    return activation


def _sides(seq_len: int, stride: int) -> tuple[int, int]:
    """Return the side of a square input grid and of the strided output grid."""
    side = math.isqrt(seq_len)
    if side * side != seq_len:
        raise ValueError(f"A block needs a square input grid; got {seq_len}.")
    return side, (side - 1) // stride + 1


def _conv_cost(
    channels_in: int,
    channels_out: int,
    *,
    kernel_size: int,
    side: int,
    stride: int,
    batch_size: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost one bias-free, same-padded convolution over a square grid."""
    return conv_cost(
        channels_in=channels_in,
        channels_out=channels_out,
        kernel_size=kernel_size,
        ndim=2,
        groups=1,
        bias=False,
        input_grid=(side, side),
        batch_size=batch_size,
        stride=stride,
        padding=kernel_size // 2,
        dtype=dtype,
    )


def _norm_cost(
    channels: int,
    side: int,
    batch_size: int,
    dtype: torch.dtype | None,
    **kwargs: object,
) -> Cost:
    """Cost one affine batch norm over a square grid."""
    return cost(
        BatchNorm2d.Config(channels, elementwise_affine=True),
        seq_len=side**2,
        batch_size=batch_size,
        dtype=dtype,
        **kwargs,
    )


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
