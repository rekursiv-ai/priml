"""CIFAR-10 classification networks.

Two architectures, one per experiment family:

* :class:`ResNet` -- the ``exp000`` baseline. A pre-activation residual network
  sized for 32x32 inputs: a 3x3 stem, three stages that halve resolution and
  double width, global average pooling, and a linear head.
* :class:`SpeedNet` -- the architecture the CIFAR-10 speedrun literature
  converged on. Wider and only eight convolutions deep, fronted by a frozen
  PCA-whitening layer. Used from ``exp001`` on.

Both take their block as a config: one template broadcast across the stack, or
an explicit list whose length IS the block count.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, cast, override

import math

from configgle import Fig, Makeable
from configgle.walk import copy_tree
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
)
from priml.math.custom_types import TensorFn
from priml.math.stats import PcaDecompose, pca_eigh
from priml.model.conv import conv_cost
from priml.model.custom_types import (
    ActivationFn,
    ChannelsIn,
    ChannelsOut,
    TensorModule,
    propagate_attr,
)
from priml.model.init import InitFn, call_init
from priml.model.norm import BatchNorm2d
from priml.model.pool import avg_pool_cost, max_pool_cost
from priml.model.swiglu import relu, silu
from priml.model.whitening import PCAWhiteningConv2d


class ResidualBlock(nn.Module):
    """Pre-activation residual block, projecting the skip when shape changes."""

    class Config(Fig["ResidualBlock"]):
        """Width, stride, and normalization of one residual block."""

        channels_in: int = -1
        """Input channels (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        stride: int = 1
        """Spatial stride of the first convolution; 2 halves resolution."""

        activation: ActivationFn = relu
        """Activation applied after each normalization."""

        norm_momentum: float = 0.1
        """BatchNorm running-statistic momentum."""

        image_size: tuple[int, int] = (0, 0)
        """``(height, width)`` of the network's input image: the token grid
        the cost is spread over. The network sets it; ``(0, 0)`` alone means
        the block stands alone and its own grid is the image."""

        grid: tuple[int, int] = (0, 0)
        """``(height, width)`` this block reads. The network sets it from the
        strides before it; ``(0, 0)`` alone means the block reads the image."""

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
            """Cost norm-act-conv twice and the shortcut for one invocation.

            ``grid`` is this block's concrete input grid. ``norm1`` and its
            activation run there; ``conv1`` and the shortcut read it and write
            the strided grid, where everything else runs. The adjoint
            accumulates the shortcut's and branch's gradients into the
            activated input, one add per element.

            Args:
              seq_len: Rows per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            c_in, c_out = self.channels_in, self.channels_out
            grid = _block_grids(self.image_size, self.grid, seq_len=seq_len)[1]
            dt = dtype
            strided = _grid(grid, kernel_size=3, stride=self.stride, padding=1)
            rows_input = batch_size * math.prod(grid)
            rows_strided = batch_size * math.prod(strided)
            at_input = (
                cost(
                    BatchNorm2d.Config(c_in, elementwise_affine=True),
                    seq_len=math.prod(grid),
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + cost(self.activation, channels=c_in * rows_input, dtype=dt)
                + elementwise_cost(
                    primal=0,
                    adjoint=c_in * rows_input,
                    channels=c_in,
                    inputs=0,
                    outputs=0,
                    rows=rows_input,
                    dtype=dt,
                )
            )
            at_output = (
                _conv2d_cost(
                    c_in,
                    c_out,
                    kernel_size=3,
                    input_grid=grid,
                    batch_size=batch_size,
                    stride=self.stride,
                    padding=1,
                    dtype=dt,
                )
                + cost(
                    BatchNorm2d.Config(c_out, elementwise_affine=True),
                    seq_len=math.prod(strided),
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + cost(self.activation, channels=c_out * rows_strided, dtype=dt)
                + _conv2d_cost(
                    c_out,
                    c_out,
                    kernel_size=3,
                    input_grid=strided,
                    batch_size=batch_size,
                    stride=1,
                    padding=1,
                    dtype=dt,
                )
                + elementwise_cost(
                    primal=c_out * rows_strided,
                    adjoint=0,
                    channels=c_out,
                    inputs=2,
                    adjoint_inputs=0,
                    adjoint_outputs=0,
                    rows=rows_strided,
                    dtype=dt,
                )
            )
            if self.stride != 1 or c_in != c_out:
                at_output += _conv2d_cost(
                    c_in,
                    c_out,
                    kernel_size=1,
                    input_grid=grid,
                    batch_size=batch_size,
                    stride=self.stride,
                    padding=0,
                    dtype=dt,
                )
            return at_input + at_output

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(
            config.channels_in,
            momentum=config.norm_momentum,
        )
        self.conv1 = nn.Conv2d(
            config.channels_in,
            config.channels_out,
            3,
            stride=config.stride,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(
            config.channels_out,
            momentum=config.norm_momentum,
        )
        self.conv2 = nn.Conv2d(
            config.channels_out,
            config.channels_out,
            3,
            padding=1,
            bias=False,
        )
        self.act = _activation(config.activation)
        self.shortcut = (
            nn.Conv2d(
                config.channels_in,
                config.channels_out,
                1,
                stride=config.stride,
                bias=False,
            )
            if config.stride != 1 or config.channels_in != config.channels_out
            else nn.Identity()
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        # The shortcut reads the POST-activation tensor, not the block input:
        # a projection applied to un-normalized activations diverges at depth.
        h = self.act(self.norm1(x))
        return self.shortcut(h) + self.conv2(self.act(self.norm2(self.conv1(h))))


def _conv2d_cost(
    channels_in: int,
    channels_out: int,
    *,
    kernel_size: int,
    input_grid: tuple[int, int],
    batch_size: int,
    stride: int = 1,
    padding: int | str = 1,
    dtype: torch.dtype | None,
    weight_grad: bool = True,
) -> Cost:
    """Cost one complete bias-free, ungrouped 2-d convolution invocation."""
    return conv_cost(
        channels_in=channels_in,
        channels_out=channels_out,
        kernel_size=kernel_size,
        ndim=2,
        groups=1,
        bias=False,
        input_grid=input_grid,
        batch_size=batch_size,
        stride=stride,
        padding=padding,
        dtype=dtype,
        weight_grad=weight_grad,
    )


def _grid(
    size: tuple[int, int],
    *,
    kernel_size: int,
    stride: int,
    padding: int,
) -> tuple[int, int]:
    """Return the grid a window sweep writes, as torch's conv and pool shape it."""
    height, width = ((s + 2 * padding - kernel_size) // stride + 1 for s in size)
    return height, width


# A block inside a network has both set by the network's finalize. A block standing
# alone reads the image, whose positions are ``seq_len``; a square is assumed, since a
# length alone cannot say otherwise.
def _block_grids(
    image_size: tuple[int, int],
    grid: tuple[int, int],
    *,
    seq_len: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Resolve a block's image and input grids."""
    if image_size == (0, 0):
        side = math.isqrt(seq_len)
        if side * side != seq_len:
            raise ValueError(
                f"A block costed alone needs a square image; seq_len={seq_len} "
                "is not a square. Set image_size on the block.",
            )
        image_size = (side, side)
    if grid == (0, 0):
        grid = image_size
    return image_size, grid


class ConvBlock(nn.Module):
    """Convolution block for :class:`SpeedNet`: conv, pool, norm, activate."""

    class Config(Fig["ConvBlock"]):
        """Width and convolution count of one block."""

        channels_in: int = -1
        """Input channels (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        num_convs: int = 3
        """Convolutions in this block; 3 adds a residual connection."""

        activation: ActivationFn = silu
        """Activation applied after each normalization."""

        norm_momentum: float = 0.4
        """BatchNorm momentum; high because runs are only a few epochs long."""

        image_size: tuple[int, int] = (0, 0)
        """``(height, width)`` of the network's input image: the token grid
        the cost is spread over. The network sets it; ``(0, 0)`` alone means
        the block stands alone and its own grid is the image."""

        grid: tuple[int, int] = (0, 0)
        """``(height, width)`` this block reads. The network sets it from the
        pools before it; ``(0, 0)`` alone means the block reads the image."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            if self.num_convs not in (1, 2, 3):
                raise ValueError(f"num_convs must be 1, 2, or 3; got {self.num_convs}.")
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the first convolution on the block's grid, the rest on the pooled one.

            The first convolution keeps ``grid`` (``padding="same"``); the
            pool halves it, and every norm, activation, and later convolution
            runs there. Three convolutions add a residual: one add forward and
            one gradient accumulation into the skip.

            Args:
              seq_len: Rows per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            c_in, c_out = self.channels_in, self.channels_out
            grid = _block_grids(self.image_size, self.grid, seq_len=seq_len)[1]
            pooled_grid = _grid(grid, kernel_size=2, stride=2, padding=0)
            rows_pooled = batch_size * math.prod(pooled_grid)
            norm_act = cost(
                BatchNorm2d.Config(c_out),
                seq_len=math.prod(pooled_grid),
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + cost(
                self.activation,
                channels=c_out * rows_pooled,
                dtype=dtype,
            )
            pooled = (
                max_pool_cost(
                    channels=c_out,
                    kernel_size=2,
                    rows=rows_pooled,
                    dtype=dtype,
                )
                + norm_act
            )
            for _ in range(self.num_convs - 1):
                pooled += _conv2d_cost(
                    c_out,
                    c_out,
                    kernel_size=3,
                    input_grid=pooled_grid,
                    batch_size=batch_size,
                    stride=1,
                    padding=1,
                    dtype=dtype,
                )
                pooled += norm_act
            if self.num_convs == 3:
                pooled += elementwise_cost(
                    primal=c_out * rows_pooled,
                    adjoint=c_out * rows_pooled,
                    channels=c_out,
                    inputs=2,
                    rows=rows_pooled,
                    dtype=dtype,
                )
            return (
                _conv2d_cost(
                    c_in,
                    c_out,
                    kernel_size=3,
                    input_grid=grid,
                    batch_size=batch_size,
                    stride=1,
                    padding=1,
                    dtype=dtype,
                )
                + pooled
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_convs = config.num_convs
        self.act = _activation(config.activation)
        self.pool = nn.MaxPool2d(2)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    config.channels_in if index == 0 else config.channels_out,
                    config.channels_out,
                    3,
                    padding="same",
                    bias=False,
                )
                for index in range(config.num_convs)
            ],
        )
        self.norms = nn.ModuleList(
            [
                # affine=False: the following convolution can absorb any
                # per-channel scale, so the parameters are redundant here and
                # measurably hurt at this training length.
                nn.BatchNorm2d(
                    config.channels_out,
                    eps=1e-12,
                    momentum=config.norm_momentum,
                    affine=False,
                )
                for _ in range(config.num_convs)
            ],
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.norms[0](self.pool(self.convs[0](x))))
        if self.num_convs == 1:
            return x
        skip = x
        x = self.act(self.norms[1](self.convs[1](x)))
        if self.num_convs == 2:
            return x
        x = self.act(self.norms[2](self.convs[2](x)))
        return x + skip


class ScaledLinear(nn.Linear):
    """Linear projection that scales its output by ``1 / fan_in``.

    Folding the scale into the projection keeps it with the weights it
    divides, so a caller swapping in a plain ``nn.Linear`` gets an unscaled
    head without a second flag to clear.
    """

    class Config(Fig["ScaledLinear"]):
        """Width and scaling of the output projection."""

        channels_in: int = -1
        """Input features."""

        channels_out: int = -1
        """Output features."""

        _: KW_ONLY

        scale: float = 0.0
        """Output multiplier; 0 means ``1 / channels_in``."""

        bias: bool = False
        """Whether the projection learns an additive bias."""

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
            """Cost the projection plus one scale multiply per output each way.

            The scale is a Python float, not a parameter, so it adds no
            gradient of its own.

            Args:
              seq_len: Rows per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            del kwargs
            rows = seq_len * batch_size
            return matmul_cost(
                channels_in=self.channels_in,
                channels_out=self.channels_out,
                bias=self.bias,
                rows=rows,
                dtype=dtype,
            ) + elementwise_cost(
                primal=self.channels_out * rows,
                adjoint=self.channels_out * rows,
                channels=self.channels_out,
                adjoint_inputs=1,
                rows=rows,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__(config.channels_in, config.channels_out, bias=config.bias)
        self.scale = config.scale if config.scale > 0 else 1.0 / config.channels_in

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Project and scale.

        Args:
          input: ``(..., channels_in)`` features.

        Returns:
          y: ``(..., channels_out)`` scaled projections.

        """
        return super().forward(input) * self.scale


# An ``nn.Module`` activation satisfies ``TensorFn``; naming the callable type rather
# than the union keeps ``self.act(x)`` inferring ``Tensor``.
def _activation(activation: ActivationFn) -> TensorFn:
    """Build an activation from a config, or pass a callable through."""
    if isinstance(activation, Makeable):
        # ``Makeable`` is runtime-checkable, so isinstance erases its type
        # parameter: ``make`` reads as returning ``object`` without the cast.
        return cast(TensorFn, activation.make())
    return activation


class ResNet(nn.Module):
    """Pre-activation residual network for 32x32 images.

    Pre-activation ordering (norm, activation, convolution) leaves an
    unbroken identity path from input to loss, which is what lets the
    network train at this depth without warmup tricks.
    """

    class Config(Fig["ResNet"]):
        """Width and block composition of the residual stack."""

        channels_in: int = -1
        """Input image channels."""

        channels_out: int = -1
        """Output logits, one per class."""

        _: KW_ONLY

        image_size: tuple[int, int] = (32, 32)
        """``(height, width)`` of the input image: the token grid every cost
        is spread over. CIFAR's 32x32 by default."""

        channels_hidden: tuple[int, ...] = (64, 128, 256)
        """Width of each stage; the first also sizes the stem."""

        block: Makeable[nn.Module] | list[Makeable[nn.Module]] = field(
            default_factory=ResidualBlock.Config,
        )
        """Block template, repeated ``blocks_per_stage`` times per stage, or an
        explicit per-stage list of lists flattened in order."""

        blocks_per_stage: int = 2
        """Repeats of the template within each stage. Ignored for a list."""

        activation: ActivationFn = relu
        """Activation used throughout the network.

        A ``Makeable`` when the activation carries state (``InlineConfig(nn.PReLU)``
        registers its learnable slope); a bare function otherwise.
        """

        norm_momentum: float = 0.1
        """BatchNorm running-statistic momentum."""

        proj_out: Makeable[TensorModule] | None = None
        """Output projection. None builds an ``nn.Linear`` over the last width."""

        @override
        def finalize(self) -> Self:
            if not self.channels_hidden:
                raise ValueError("channels_hidden must name at least one stage.")
            if self.blocks_per_stage < 1:
                raise ValueError(
                    f"blocks_per_stage must be positive; got {self.blocks_per_stage}.",
                )
            if self.proj_out is not None:
                propagate_attr(
                    self.proj_out,
                    "channels_in",
                    self.channels_hidden[-1],
                    protocol=ChannelsIn,
                )
                propagate_attr(
                    self.proj_out,
                    "channels_out",
                    self.channels_out,
                    protocol=ChannelsOut,
                )
            return super().finalize()

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the stem, every block, final norm, pool, and head.

            Each block receives the concrete grid it reads; a stride-2 block
            leaves ``ceil(size / 2)`` for the next. The pool and head run once
            per image, and every operation is counted for this invocation.

            Args:
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            image_size = self.image_size
            seq_len = math.prod(image_size)
            costed = _conv2d_cost(
                self.channels_in,
                self.channels_hidden[0],
                kernel_size=3,
                input_grid=image_size,
                batch_size=batch_size,
                stride=1,
                padding=1,
                dtype=dtype,
            )
            grid = image_size
            for stage_blocks in _resnet_blocks(self):
                for block, stride in stage_blocks:
                    propagate_attr(block, "image_size", image_size)
                    propagate_attr(block, "grid", grid)
                    costed += cost(
                        block,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        dtype=dtype,
                        **kwargs,
                    )
                    grid = _grid(grid, kernel_size=3, stride=stride, padding=1)
            c_last = self.channels_hidden[-1]
            at_output = cost(
                BatchNorm2d.Config(c_last, elementwise_affine=True),
                seq_len=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + cost(
                self.activation,
                channels=c_last * batch_size * math.prod(grid),
                dtype=dtype,
            )
            head = (
                matmul_cost(
                    channels_in=c_last,
                    channels_out=self.channels_out,
                    bias=True,
                    rows=batch_size,
                    dtype=dtype,
                )
                if self.proj_out is None
                else cost(
                    self.proj_out,
                    seq_len=1,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
            )
            return (
                costed
                + at_output
                + avg_pool_cost(
                    channels=c_last,
                    positions=math.prod(grid),
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + head
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
        stages: list[nn.Module] = []
        for stage_blocks in _resnet_blocks(config):
            stages.extend(block.make() for block, _ in stage_blocks)
            stages.append(nn.Identity())
        self.stages = nn.Sequential(*stages)
        channels = config.channels_hidden[-1]
        self.norm = nn.BatchNorm2d(channels, momentum=config.norm_momentum)
        self.act = _activation(config.activation)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = (
            config.proj_out.make()
            if config.proj_out is not None
            else nn.Linear(channels, config.channels_out)
        )

    @override
    def forward(self, media: Tensor) -> Tensor:
        """Classify a batch of images.

        Args:
          media: ``(B, channels_in, H, W)`` normalized images.

        Returns:
          logits: ``(B, channels_out)`` class scores.

        """
        x = self.act(self.norm(self.stages(self.stem(media))))
        return self.head(self.pool(x).flatten(1))


class SpeedNet(nn.Module):
    """Wide, shallow network fronted by a frozen PCA-whitening layer.

    The speedrun architecture: whitening the first layer's patches removes the
    strong local correlation of natural images, which is what lets a network
    this shallow reach competitive accuracy in a handful of epochs. The
    whitening weights are data-derived, so :meth:`init_whiten` must run on a
    batch of training images before the first optimizer step.

    References:
      https://github.com/KellerJordan/cifar10-airbench
      Jordan 2024. 94% on CIFAR-10 in 3.29 seconds on a single A100.

    """

    class Config(Fig["SpeedNet"]):
        """Width and block composition of the speedrun network."""

        channels_in: int = -1
        """Input image channels."""

        channels_out: int = -1
        """Output logits, one per class."""

        _: KW_ONLY

        image_size: tuple[int, int] = (32, 32)
        """``(height, width)`` of the input image: the token grid every cost
        is spread over. CIFAR's 32x32 by default."""

        channels_hidden: tuple[int, ...] = (128, 384, 512)
        """Width of each convolution block."""

        block: Makeable[nn.Module] | list[Makeable[nn.Module]] = field(
            default_factory=ConvBlock.Config,
        )
        """Block template broadcast across ``channels_hidden``, or one config
        per block."""

        whiten_kernel: int = 2
        """Patch size the whitening layer decomposes."""

        norm_momentum: float = 0.4
        """BatchNorm momentum; high because runs are only a few epochs long."""

        activation: ActivationFn = silu
        """Activation used throughout the network.

        A ``Makeable`` when the activation carries state (``InlineConfig(nn.PReLU)``
        registers its learnable slope); a bare function otherwise.
        """

        proj_out: Makeable[TensorModule] = field(default_factory=ScaledLinear.Config)
        """Output projection; owns its own scaling."""

        init_conv: InitFn | None = None
        """Re-initializes every convolution after construction; None keeps torch's.

        ``dirac`` makes each block start as an identity map.
        """

        @property
        def whiten_width(self) -> int:
            """Return the whitening layer's width: each eigenvector and its negation.

            PCA yields ``channels_in * kernel**2`` eigenvectors; the layer emits
            both a vector and its negation so a following ReLU-like activation
            can respond to projections of either sign.
            """
            return 2 * self.channels_in * self.whiten_kernel * self.whiten_kernel

        @override
        def finalize(self) -> Self:
            if not self.channels_hidden:
                raise ValueError("channels_hidden must name at least one block.")
            propagate_attr(
                self.proj_out,
                "channels_in",
                self.channels_hidden[-1],
                protocol=ChannelsIn,
            )
            propagate_attr(
                self.proj_out,
                "channels_out",
                self.channels_out,
                protocol=ChannelsOut,
            )
            return super().finalize()

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost whitening, every block, final pool, and head.

            The unpadded whitening convolution shrinks the grid by
            ``whiten_kernel - 1``; each block pools it by two; the final pool
            by three. Every layer is costed at its own concrete grid.

            The whitening weight is frozen but owned: it is in ``params``, and
            its adjoint is the input gradient alone -- one convolution of the
            primal's size, not the two a trainable layer pays.

            Args:
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            image_size = self.image_size
            seq_len = math.prod(image_size)
            grid = _grid(
                image_size,
                kernel_size=self.whiten_kernel,
                stride=1,
                padding=0,
            )
            output_rows = batch_size * math.prod(grid)
            whitening = _conv2d_cost(
                self.channels_in,
                self.whiten_width,
                kernel_size=self.whiten_kernel,
                input_grid=image_size,
                batch_size=batch_size,
                stride=1,
                padding=0,
                dtype=dtype,
                weight_grad=False,
            )
            costed = whitening + cost(
                self.activation,
                channels=self.whiten_width * output_rows,
                dtype=dtype,
            )
            for block in _speednet_blocks(self):
                propagate_attr(block, "image_size", image_size)
                propagate_attr(block, "grid", grid)
                costed += cost(
                    block,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                grid = _grid(grid, kernel_size=2, stride=2, padding=0)
            grid = _grid(grid, kernel_size=3, stride=3, padding=0)
            tail = max_pool_cost(
                channels=self.channels_hidden[-1],
                kernel_size=3,
                rows=batch_size * math.prod(grid),
                dtype=dtype,
            ) + cost(
                self.proj_out,
                seq_len=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            return costed + tail

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.whiten = PCAWhiteningConv2d(
            config.channels_in,
            config.whiten_width,
            config.whiten_kernel,
            padding=0,
            bias=False,
        )
        self.act = _activation(config.activation)
        self.blocks = nn.Sequential(
            *(block.make() for block in _speednet_blocks(config)),
        )
        self.pool = nn.MaxPool2d(3)
        self.head = config.proj_out.make()
        if config.init_conv is not None:
            for module in self.modules():
                if isinstance(module, nn.Conv2d) and module is not self.whiten:
                    # An init that writes an identity kernel needs a square
                    # slice; a wider conv keeps its remaining channels random.
                    call_init(
                        config.init_conv,
                        module.weight.data[: module.weight.size(1)],
                    )
            head_weight = cast(nn.Linear, self.head).weight.data
            head_weight.div_(head_weight.std())

    def init_whiten(
        self,
        media: Tensor,
        *,
        decompose: PcaDecompose = pca_eigh,
    ) -> None:
        """Fit the whitening layer to a batch of training images.

        Args:
          media: ``(N, channels_in, H, W)`` images to decompose.
          decompose: Eigendecomposition backing the PCA fit. The default
            reaches ``linalg.eigh``, which MPS lacks; pass ``pca_power``
            there.

        """
        self.whiten.init_whiten(media, decompose=decompose)

    @override
    def forward(self, media: Tensor) -> Tensor:
        """Classify a batch of images.

        Args:
          media: ``(B, channels_in, H, W)`` normalized images.

        Returns:
          logits: ``(B, channels_out)`` class scores.

        """
        pooled = self.pool(self.blocks(self.act(self.whiten(media))))
        assert isinstance(pooled, Tensor)
        return self.head(pooled.flatten(1))


def _speednet_blocks(config: SpeedNet.Config) -> list[Makeable[nn.Module]]:
    """Wire one block per hidden width, the first reading the whitened channels."""
    blocks: list[Makeable[nn.Module]] = []
    channels = config.whiten_width
    grid = _block_grid(config.block, len(config.channels_hidden), 1)
    for stage_channels, (block,) in zip(config.channels_hidden, grid, strict=True):
        propagate_attr(block, "channels_in", channels, protocol=ChannelsIn)
        channels = stage_channels
        propagate_attr(block, "channels_out", channels, protocol=ChannelsOut)
        propagate_attr(block, "activation", config.activation)
        propagate_attr(block, "norm_momentum", config.norm_momentum)
        blocks.append(block)
    return blocks


# A shared config would be mutated once per stage and every block would end up carrying
# the last stage's width. A caller's list is copied too, so wiring it -- from ``cost``
# as much as from ``__init__`` -- leaves the caller's configs untouched.
def _block_grid(
    block: Makeable[nn.Module] | list[Makeable[nn.Module]],
    num_stages: int,
    blocks_per_stage: int,
) -> list[list[Makeable[nn.Module]]]:
    """Group block configs by stage, copying each so wiring mutates no caller's config."""
    flat: list[Makeable[nn.Module]]
    if isinstance(block, list):
        expected = num_stages * blocks_per_stage
        if len(block) != expected:
            raise ValueError(
                f"block list must hold {expected} configs "
                f"({num_stages} stages x {blocks_per_stage}); got {len(block)}.",
            )
        flat = [copy_tree(member) for member in block]
    else:
        flat = [
            copy_tree(block) for _ in range(num_stages) for _ in range(blocks_per_stage)
        ]
    return [
        flat[index * blocks_per_stage : (index + 1) * blocks_per_stage]
        for index in range(num_stages)
    ]


def _resnet_blocks(
    config: ResNet.Config,
) -> list[list[tuple[Makeable[nn.Module], int]]]:
    """Wire each stage's blocks and pair each with the stride it was given."""
    stages: list[list[tuple[Makeable[nn.Module], int]]] = []
    channels = config.channels_hidden[0]
    grid = _block_grid(
        config.block,
        len(config.channels_hidden),
        config.blocks_per_stage,
    )
    for stage_index, stage_blocks in enumerate(grid):
        stage: list[tuple[Makeable[nn.Module], int]] = []
        for block_index, block in enumerate(stage_blocks):
            propagate_attr(block, "channels_in", channels, protocol=ChannelsIn)
            channels = config.channels_hidden[stage_index]
            propagate_attr(block, "channels_out", channels, protocol=ChannelsOut)
            # Downsample once per stage, on its first block. The first stage
            # keeps full resolution: 32x32 is already small, and halving it
            # here costs accuracy outright.
            stride = 2 if stage_index and not block_index else 1
            propagate_attr(block, "stride", stride)
            propagate_attr(block, "activation", config.activation)
            propagate_attr(block, "norm_momentum", config.norm_momentum)
            stage.append((block, stride))
        stages.append(stage)
    return stages
