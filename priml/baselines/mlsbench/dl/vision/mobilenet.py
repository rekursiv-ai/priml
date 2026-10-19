"""MobileNetV2 in the CIFAR variant the MLS-Bench vision tasks use.

A stride-1 3x3 stem, seventeen inverted residual blocks in seven stages, a 1x1
convolution to 1280 channels, a global pool and a linear head. Layers are
registered in the order of the reference network: registration order fixes the
order initialization consumes the generator.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import TYPE_CHECKING, Self, cast, override

import math

from configgle import Fig, Makeable
from configgle.walk import copy_tree
from torch import Tensor, nn

import torch

from priml.baselines.mlsbench.dl.vision.model import (
    BatchNorm,
    GlobalAvgPool,
    ModelInitFn,
    NormFn,
    activation_layer,
    init_layers,
    kaiming_normal_fan_in,
    kaiming_normal_fan_out,
    keep_initialization,
    norm_cost,
)
from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    map_cost,
    matmul_cost,
    set_cost,
)
from priml.model.conv import conv_cost
from priml.model.custom_types import (
    ActivationFn,
    ChannelsIn,
    ChannelsOut,
    propagate_attr,
)
from priml.model.init import InitFn


if TYPE_CHECKING:
    from collections.abc import Mapping


# Two compares forward, one mask multiply back.
@set_cost(map_cost(primal=2, adjoint=1))
def relu6(x: Tensor) -> Tensor:
    """Apply ``min(max(0, x), 6)``.

    Args:
      x: Pre-activation values, any shape.

    Returns:
      activated: ``x`` clipped to ``[0, 6]``, elementwise.

    """
    return nn.functional.relu6(x)


class InvertedResidual(nn.Module):
    """Expand with a 1x1, filter depthwise, project back with a linear 1x1.

    References:
      https://arxiv.org/abs/1801.04381
        Sandler et al. 2018, "MobileNetV2: Inverted Residuals and Linear
        Bottlenecks."

    """

    class Config(Fig["InvertedResidual"]):
        """Widths, stride, expansion, activation and normalization of one block."""

        channels_in: int = -1
        """Input channels (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        stride: int = 1
        """Stride of the depthwise convolution."""

        expand_ratio: int = 1
        """Hidden width over input width; 1 skips the expanding convolution."""

        activation: ActivationFn = relu6
        """Activation after the expansion and after the depthwise convolution."""

        norm: NormFn = BatchNorm
        """Builds the normalization after every convolution."""

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
            """Cost the three convolutions, their norms and activations, and the sum.

            Args:
              seq_len: Positions of the block's input, a square grid.
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every norm.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes and ownership.

            Raises:
              ValueError: ``seq_len`` is not a perfect square.

            """
            side = math.isqrt(seq_len)
            if side * side != seq_len:
                raise ValueError(f"A block needs a square input grid; got {seq_len}.")
            c_in, c_out = self.channels_in, self.channels_out
            hidden = hidden_width(self)
            side_out = (side - 1) // self.stride + 1
            total = Cost()
            if self.expand_ratio != 1:
                total += _conv_norm_act(
                    self,
                    c_in,
                    hidden,
                    side=side,
                    batch_size=batch_size,
                    dtype=dtype,
                    bus=kwargs,
                )
            total += _conv_norm_act(
                self,
                hidden,
                hidden,
                side=side,
                kernel_size=3,
                stride=self.stride,
                groups=hidden,
                batch_size=batch_size,
                dtype=dtype,
                bus=kwargs,
            ) + _conv_norm_act(
                self,
                hidden,
                c_out,
                side=side_out,
                batch_size=batch_size,
                dtype=dtype,
                activated=False,
                bus=kwargs,
            )
            if residual(self):
                rows = batch_size * seq_len
                total += elementwise_cost(
                    primal=c_out * rows,
                    adjoint=0,
                    channels=c_out,
                    inputs=2,
                    adjoint_inputs=0,
                    adjoint_outputs=0,
                    rows=rows,
                    dtype=dtype,
                )
            return total

    def __init__(self, config: Config) -> None:
        super().__init__()
        hidden = hidden_width(config)
        self.use_res = residual(config)
        layers: list[nn.Module] = []
        if config.expand_ratio != 1:
            layers += [
                nn.Conv2d(config.channels_in, hidden, 1, bias=False),
                config.norm(hidden),
                activation_layer(config.activation),
            ]
        layers += [
            nn.Conv2d(
                hidden,
                hidden,
                3,
                stride=config.stride,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            config.norm(hidden),
            activation_layer(config.activation),
            nn.Conv2d(hidden, config.channels_out, 1, bias=False),
            config.norm(config.channels_out),
        ]
        self.conv = nn.Sequential(*layers)

    @override
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        return x + out if self.use_res else out


class MobileNetV2(nn.Module):
    """MobileNetV2 for small images, ending in a linear classifier."""

    class Config(Fig["MobileNetV2"]):
        """Stages, widths and initialization of the network."""

        channels_in: int = -1
        """Input image channels; the experiment sets it."""

        channels_out: int = -1
        """Output logits, one per class; the experiment sets it."""

        _: KW_ONLY

        image_size: int = 32
        """Side of the square input image, which the cost is spread over."""

        channels_stem: int = 32
        """Width of the stem convolution."""

        channels_hidden: tuple[int, ...] = (16, 24, 32, 64, 96, 160, 320)
        """Output width of every block of each stage."""

        blocks_per_stage: tuple[int, ...] = (1, 2, 3, 4, 3, 3, 1)
        """Blocks in each stage."""

        expand_ratios: tuple[int, ...] = (1, 6, 6, 6, 6, 6, 6)
        """Expansion of every block of each stage."""

        strides: tuple[int, ...] = (1, 1, 2, 2, 1, 2, 1)
        """Stride of each stage's first block; the rest stride 1."""

        channels_last: int = 1280
        """Width of the 1x1 convolution after the last stage."""

        block: Makeable[nn.Module] = field(default_factory=InvertedResidual.Config)
        """Block template, copied and sized per position."""

        activation: ActivationFn = relu6
        """Activation of the stem, the last convolution and, pushed down, every block."""

        norm: NormFn = BatchNorm
        """Builds every normalization: the stem's, the last one and every block's."""

        pool: Makeable[nn.Module] = field(default_factory=GlobalAvgPool.Config)
        """Reduces the last convolution's ``(B, C, H, W)`` to ``(B, C)`` for the head."""

        init_conv: InitFn = kaiming_normal_fan_out
        """Initializer for every convolution weight."""

        init_head: InitFn = kaiming_normal_fan_in
        """Initializer for the classifier weight."""

        init_bias: InitFn = nn.init.zeros_
        """Initializer for the classifier bias."""

        init: ModelInitFn = keep_initialization
        """Runs on the built network after the per-tensor initializers."""

        @override
        def finalize(self) -> Self:
            lengths = {
                len(self.channels_hidden),
                len(self.blocks_per_stage),
                len(self.expand_ratios),
                len(self.strides),
            }
            if len(lengths) != 1:
                raise ValueError(
                    "channels_hidden, blocks_per_stage, expand_ratios and strides "
                    "need one entry per stage.",
                )
            if min(self.blocks_per_stage, default=0) < 1:
                raise ValueError(
                    f"Every stage needs a block; got {self.blocks_per_stage}.",
                )
            return super().finalize()

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the stem, every block, the last convolution, the pool and the head.

            Args:
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes and ownership.

            """
            side = self.image_size
            total = _conv_norm_act(
                self,
                self.channels_in,
                self.channels_stem,
                side=side,
                kernel_size=3,
                batch_size=batch_size,
                dtype=dtype,
                bus=kwargs,
            )
            for block, stride in sized_blocks(self):
                total += cost(
                    block,
                    seq_len=side**2,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                side = (side - 1) // stride + 1
            return (
                total
                + _conv_norm_act(
                    self,
                    self.channels_hidden[-1],
                    self.channels_last,
                    side=side,
                    batch_size=batch_size,
                    dtype=dtype,
                    bus=kwargs,
                )
                + cost(
                    self.pool,
                    channels=self.channels_last,
                    seq_len=side**2,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + self.head_cost(
                    channels_out=self.channels_out,
                    batch_size=batch_size,
                    dtype=dtype,
                )
            )

        def head(self, channels_out: int) -> nn.Linear:
            """Build a linear classifier over the pooled features, ``channels_out`` wide."""
            return nn.Linear(self.channels_last, channels_out)

        def head_cost(
            self,
            *,
            channels_out: int,
            batch_size: int,
            dtype: torch.dtype | None,
        ) -> Cost:
            """Cost one :meth:`head` of width ``channels_out``.

            Args:
              channels_out: Logits of the head.
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes and ownership.

            """
            return matmul_cost(
                channels_in=self.channels_last,
                channels_out=channels_out,
                bias=True,
                rows=batch_size,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                config.channels_in,
                config.channels_stem,
                3,
                padding=1,
                bias=False,
            ),
            config.norm(config.channels_stem),
            activation_layer(config.activation),
        )
        self.blocks = nn.Sequential(
            *(block.make() for block, _ in sized_blocks(config)),
        )
        self.last = nn.Sequential(
            nn.Conv2d(config.channels_hidden[-1], config.channels_last, 1, bias=False),
            config.norm(config.channels_last),
            activation_layer(config.activation),
        )
        # Between the last convolution and the head, as upstream's ``CustomPool``
        # is: a pool with parameters draws its initialization in this position.
        self.pool = config.pool.make()
        self.head = config.head(config.channels_out)
        # Construction already drew torch's default initialization; those draws
        # stay in the stream, so the re-initialization lands where upstream's does.
        init_layers(
            self,
            init_conv=config.init_conv,
            init_head=config.init_head,
            init_bias=config.init_bias,
        )
        config.init(self)

    @override
    def forward(self, media: Tensor) -> Tensor:
        """Classify a batch of images.

        Args:
          media: ``(B, channels_in, H, W)`` normalized images.

        Returns:
          logits: ``(B, channels_out)`` class scores.

        """
        return self.head(self.features(media))

    def features(self, media: Tensor) -> Tensor:
        """Return the pooled ``(B, channels_last)`` features the head reads."""
        x = self.last(self.blocks(self.stem(media)))
        return cast(Tensor, self.pool(x))


def hidden_width(config: InvertedResidual.Config) -> int:
    """Return a block's expanded width, rounded as the reference rounds it."""
    return round(config.channels_in * config.expand_ratio)


def residual(config: InvertedResidual.Config) -> bool:
    """Return whether a block adds its input: same shape in and out."""
    return config.stride == 1 and config.channels_in == config.channels_out


# A shared template would be mutated once per block and every block would end up
# carrying the last stage's width, so each position gets its own copy.
def sized_blocks(config: MobileNetV2.Config) -> list[tuple[Makeable[nn.Module], int]]:
    """Size one copy of the block template per position and pair it with its stride.

    Args:
      config: The network.

    Returns:
      blocks: Each position's block config and stride, in network order.

    """
    blocks: list[tuple[Makeable[nn.Module], int]] = []
    channels = config.channels_stem
    for width, count, ratio, first in zip(
        config.channels_hidden,
        config.blocks_per_stage,
        config.expand_ratios,
        config.strides,
        strict=True,
    ):
        for position in range(count):
            block = copy_tree(config.block)
            stride = first if position == 0 else 1
            propagate_attr(block, "channels_in", channels, protocol=ChannelsIn)
            propagate_attr(block, "channels_out", width, protocol=ChannelsOut)
            propagate_attr(block, "stride", stride)
            propagate_attr(block, "expand_ratio", ratio)
            propagate_attr(block, "activation", config.activation)
            propagate_attr(block, "norm", config.norm)
            blocks.append((block, stride))
            channels = width
    return blocks


# ``bus`` is the caller's open cost bus, forwarded to the norm; passed as a mapping so
# it cannot collide with this function's own keywords.
def _conv_norm_act(
    config: InvertedResidual.Config | MobileNetV2.Config,
    channels_in: int,
    channels_out: int,
    *,
    side: int,
    batch_size: int,
    dtype: torch.dtype | None,
    kernel_size: int = 1,
    stride: int = 1,
    groups: int = 1,
    activated: bool = True,
    bus: Mapping[str, object],
) -> Cost:
    """Cost a bias-free convolution, the norm after it and, if activated, the activation."""
    side_out = (side - 1) // stride + 1
    total = conv_cost(
        channels_in=channels_in,
        channels_out=channels_out,
        kernel_size=kernel_size,
        ndim=2,
        groups=groups,
        bias=False,
        input_grid=(side, side),
        batch_size=batch_size,
        stride=stride,
        padding=kernel_size // 2,
        dtype=dtype,
    ) + norm_cost(
        config.norm,
        channels=channels_out,
        side=side_out,
        batch_size=batch_size,
        dtype=dtype,
        **bus,
    )
    if activated:
        total += cost(
            config.activation,
            channels=channels_out * batch_size * side_out**2,
            dtype=dtype,
        )
    return total
