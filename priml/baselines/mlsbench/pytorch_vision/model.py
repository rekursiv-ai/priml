"""CIFAR-style residual network with post-activation basic blocks.

The architecture family of the MLS-Bench vision tasks: a 3x3 stem, three
stages that double the width and halve the resolution, global average
pooling and a linear head. Depth is the number of blocks per stage, so
``(3, 3, 3)`` is ResNet-20 and ``(9, 9, 9)`` is ResNet-56.

Layers are registered in the order of the reference network. Registration
order fixes both the order the generator is consumed by initialization and
the order parameters are summed in, so reordering is not a refactor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, field
from typing import Self, cast, override

import math

from configgle import Fig, Makeable
from configgle.walk import copy_tree
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, cost, elementwise_cost, matmul_cost
from priml.math.custom_types import TensorFn
from priml.model.conv import conv_cost
from priml.model.custom_types import (
    ActivationFn,
    ChannelsIn,
    ChannelsOut,
    propagate_attr,
)
from priml.model.init import InitFn, call_init
from priml.model.norm import BatchNorm2d
from priml.model.pool import avg_pool_cost
from priml.model.swiglu import relu


class GlobalAvgPool(nn.Module):
    """Average every channel over its whole spatial grid: ``(B, C, H, W) -> (B, C)``."""

    class Config(Fig["GlobalAvgPool"]):
        """No parameters; a config so the pool is a slot a fork can fill."""

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return functional.adaptive_avg_pool2d(x, 1).flatten(1)


type NormFn = Callable[[int], nn.Module]
"""Builds a normalization layer for a channel count; a class such as ``nn.BatchNorm2d``."""

type ModelInitFn = Callable[[nn.Module], object]
"""Initializes a built network in place, from its whole structure."""


def keep_initialization(model: nn.Module) -> None:
    """Leave the network as the per-tensor initializers left it."""
    del model


def kaiming_normal_fan_out(w: Tensor) -> None:
    """Draw from a normal with variance ``2 / fan_out``, He's rule for ReLU convolutions.

    Args:
      w: Weight to overwrite in place.

    """
    nn.init.kaiming_normal_(w, mode="fan_out", nonlinearity="relu")


def kaiming_normal_fan_in(w: Tensor) -> None:
    """Draw from a normal with variance ``2 / fan_in``, He's rule for ReLU layers.

    Args:
      w: Weight to overwrite in place.

    """
    nn.init.kaiming_normal_(w, mode="fan_in", nonlinearity="relu")


class BasicBlock(nn.Module):
    """Two 3x3 convolutions and a skip connection, activated after the sum."""

    class Config(Fig["BasicBlock"]):
        """Width, stride and activation of one block."""

        channels_in: int = -1
        """Input channels (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        stride: int = 1
        """Stride of the first convolution and of the projection shortcut."""

        activation: ActivationFn = relu
        """Activation after the first normalization and after the sum."""

        norm: NormFn = nn.BatchNorm2d
        """Builds each normalization: after each convolution and in the projection."""

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
            """Cost the convolutions, normalizations, activations and the sum.

            Args:
              seq_len: Positions of the block's input, a square grid.
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes and ownership.

            Raises:
              ValueError: ``seq_len`` is not a perfect square.

            """
            side = math.isqrt(seq_len)
            if side * side != seq_len:
                raise ValueError(f"A block needs a square input grid; got {seq_len}.")
            c_in, c_out, stride = self.channels_in, self.channels_out, self.stride
            side_out = (side - 1) // stride + 1
            rows_out = batch_size * side_out**2
            norm = _norm_cost(c_out, side_out, batch_size, dtype, **kwargs)
            act = cost(self.activation, channels=c_out * rows_out, dtype=dtype)
            total = (
                _conv_cost(
                    c_in,
                    c_out,
                    kernel_size=3,
                    side=side,
                    stride=stride,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + norm
                + act
                + _conv_cost(
                    c_out,
                    c_out,
                    kernel_size=3,
                    side=side_out,
                    stride=1,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + norm
                + act
                + elementwise_cost(
                    primal=c_out * rows_out,
                    adjoint=c_in * batch_size * seq_len,
                    channels=c_out,
                    inputs=2,
                    adjoint_inputs=0,
                    adjoint_outputs=0,
                    rows=rows_out,
                    dtype=dtype,
                )
            )
            if stride != 1 or c_in != c_out:
                total += (
                    _conv_cost(
                        c_in,
                        c_out,
                        kernel_size=1,
                        side=side,
                        stride=stride,
                        batch_size=batch_size,
                        dtype=dtype,
                    )
                    + norm
                )
            return total

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            config.channels_in,
            config.channels_out,
            3,
            stride=config.stride,
            padding=1,
            bias=False,
        )
        self.norm1 = config.norm(config.channels_out)
        self.conv2 = nn.Conv2d(
            config.channels_out,
            config.channels_out,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = config.norm(config.channels_out)
        # Before the shortcut, as upstream registers it: an activation with
        # parameters draws its initialization in this position.
        self.act = _activation(config.activation)
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
        h = self.act(cast(Tensor, self.norm1(self.conv1(x))))
        return self.act(cast(Tensor, self.norm2(self.conv2(h))) + self.shortcut(x))


class ResNet(nn.Module):
    """Residual network for small images, ending in a linear classifier."""

    class Config(Fig["ResNet"]):
        """Width, depth and initialization of the network."""

        channels_in: int = -1
        """Input image channels; the experiment sets it."""

        channels_out: int = -1
        """Output logits, one per class; the experiment sets it."""

        _: KW_ONLY

        image_size: int = 32
        """Side of the square input image, which the cost is spread over."""

        channels_hidden: tuple[int, ...] = (16, 32, 64)
        """Width of each stage; the first also sizes the stem."""

        blocks_per_stage: tuple[int, ...] = (3, 3, 3)
        """Blocks in each stage, one entry per entry of ``channels_hidden``."""

        block: Makeable[nn.Module] | list[Makeable[nn.Module]] = field(
            default_factory=BasicBlock.Config,
        )
        """Block template copied per position, or one config per position in order.

        Either way each is sized from its stage. A list gives a block its own
        position-dependent values (stochastic depth's survival)."""

        activation: ActivationFn = relu
        """Activation of the stem and, pushed down, of every block."""

        norm: NormFn = nn.BatchNorm2d
        """Builds the stem's normalization and, pushed down, every block's."""

        pool: Makeable[nn.Module] = field(default_factory=GlobalAvgPool.Config)
        """Reduces the last stage's ``(B, C, H, W)`` to ``(B, C)`` for the head."""

        init_conv: InitFn = kaiming_normal_fan_out
        """Initializer for every convolution weight."""

        init_head: InitFn = kaiming_normal_fan_in
        """Initializer for the classifier weight."""

        init_bias: InitFn = nn.init.zeros_
        """Initializer for the classifier bias."""

        init: ModelInitFn = keep_initialization
        """Runs on the built network after the per-tensor initializers."""

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the stem, every block, the pool and the head.

            Args:
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes and ownership.

            """
            side = self.image_size
            width = self.channels_hidden[0]
            total = (
                _conv_cost(
                    self.channels_in,
                    width,
                    kernel_size=3,
                    side=side,
                    stride=1,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + _norm_cost(width, side, batch_size, dtype, **kwargs)
                + cost(
                    self.activation,
                    channels=width * batch_size * side**2,
                    dtype=dtype,
                )
            )
            for stage in _stages(self):
                for block, stride in stage:
                    total += cost(
                        block,
                        seq_len=side**2,
                        batch_size=batch_size,
                        dtype=dtype,
                        **kwargs,
                    )
                    side = (side - 1) // stride + 1
            channels = self.channels_hidden[-1]
            return (
                total
                + avg_pool_cost(
                    channels=channels,
                    positions=side**2,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + matmul_cost(
                    channels_in=channels,
                    channels_out=self.channels_out,
                    bias=True,
                    rows=batch_size,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.stem = nn.Conv2d(
            config.channels_in,
            config.channels_hidden[0],
            3,
            padding=1,
            bias=False,
        )
        self.norm = config.norm(config.channels_hidden[0])
        self.act = _activation(config.activation)
        self.stages = nn.Sequential(
            *(
                nn.Sequential(*(block.make() for block, _ in stage))
                for stage in _stages(config)
            ),
        )
        # Registered between the stages and the head, as upstream's ``CustomPool``
        # is: a pool with parameters draws its initialization in this position.
        self.pool = config.pool.make()
        self.head = nn.Linear(config.channels_hidden[-1], config.channels_out)
        # Construction above already drew torch's default initialization for every
        # layer. Those draws are overwritten here but stay in the stream: skipping
        # them, say by building on the meta device, moves every later draw and the
        # network no longer reproduces the reference from the same seed.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                call_init(config.init_conv, module.weight)
        call_init(config.init_head, self.head.weight)
        call_init(config.init_bias, cast(Tensor, self.head.bias))
        config.init(self)

    @override
    def forward(self, media: Tensor) -> Tensor:
        """Classify a batch of images.

        Args:
          media: ``(B, channels_in, H, W)`` normalized images.

        Returns:
          logits: ``(B, channels_out)`` class scores.

        """
        x = self.stages(self.act(cast(Tensor, self.norm(self.stem(media)))))
        return self.head(self.pool(x))


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


# An ``nn.Module`` activation satisfies ``TensorFn``; naming the callable type rather
# than the union keeps ``self.act(x)`` inferring ``Tensor``.
def _activation(activation: ActivationFn) -> TensorFn:
    """Build an activation from a config, or pass a callable through."""
    if isinstance(activation, Makeable):
        # ``Makeable`` is runtime-checkable, so isinstance erases its type
        # parameter: ``make`` reads as returning ``object`` without the cast.
        return cast(TensorFn, activation.make())
    return activation


# A shared template would be mutated once per block and every block would end up
# carrying the last stage's width, so each position gets its own copy.
def _stages(
    config: ResNet.Config,
) -> list[list[tuple[Makeable[nn.Module], int]]]:
    """Size one block config per position and pair it with its stride."""
    if min(config.blocks_per_stage) < 1:
        raise ValueError(f"Every stage needs a block; got {config.blocks_per_stage}.")
    total = sum(config.blocks_per_stage)
    templates = (
        config.block if isinstance(config.block, list) else [config.block] * total
    )
    if len(templates) != total:
        raise ValueError(
            f"block lists {len(templates)} configs; blocks_per_stage needs {total}.",
        )
    configs = iter(templates)
    stages: list[list[tuple[Makeable[nn.Module], int]]] = []
    channels = config.channels_hidden[0]
    for index, (width, count) in enumerate(
        zip(config.channels_hidden, config.blocks_per_stage, strict=True),
    ):
        stage: list[tuple[Makeable[nn.Module], int]] = []
        for position in range(count):
            block = copy_tree(next(configs))
            propagate_attr(block, "channels_in", channels, protocol=ChannelsIn)
            propagate_attr(block, "channels_out", width, protocol=ChannelsOut)
            channels = width
            # Each stage after the first halves the resolution once, on its
            # first block; 32x32 is already small enough to keep at full size.
            stride = 2 if index and not position else 1
            propagate_attr(block, "stride", stride)
            propagate_attr(block, "activation", config.activation)
            propagate_attr(block, "norm", config.norm)
            stage.append((block, stride))
        stages.append(stage)
    return stages
