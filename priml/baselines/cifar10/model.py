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

from dataclasses import field
from typing import Self, cast, override

from configgle import Fig, Makeable
from configgle.walk import copy_tree
from torch import Tensor, nn

import torch

from priml.math.custom_types import TensorFn
from priml.math.stats import PcaDecompose, pca_eigh
from priml.model.custom_types import (
    ActivationFn,
    ChannelsIn,
    ChannelsOut,
    TensorModule,
    propagate_attr,
)
from priml.model.init import InitFn, call_init
from priml.model.whitening import PCAWhiteningConv2d


class ResidualBlock(nn.Module):
    """Pre-activation residual block, projecting the skip when shape changes."""

    class Config(Fig["ResidualBlock"]):
        """Width, stride, and normalization of one residual block."""

        channels_in: int = -1
        """Input channels (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Output channels (-1 to infer from channels_in)."""

        stride: int = 1
        """Spatial stride of the first convolution; 2 halves resolution."""

        activation: ActivationFn = torch.relu
        """Activation applied after each normalization."""

        norm_momentum: float = 0.1
        """BatchNorm running-statistic momentum."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            return super().finalize()

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


class ConvBlock(nn.Module):
    """Convolution block for :class:`SpeedNet`: conv, pool, norm, activate."""

    class Config(Fig["ConvBlock"]):
        """Width and convolution count of one block."""

        channels_in: int = -1
        """Input channels (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Output channels (-1 to infer from channels_in)."""

        num_convs: int = 3
        """Convolutions in this block; 3 adds a residual connection."""

        activation: ActivationFn = nn.functional.silu
        """Activation applied after each normalization."""

        norm_momentum: float = 0.4
        """BatchNorm momentum; high because runs are only a few epochs long."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            if self.num_convs not in (1, 2, 3):
                raise ValueError(f"num_convs must be 1, 2, or 3; got {self.num_convs}.")
            return super().finalize()

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

        scale: float = 0.0
        """Output multiplier; 0 means ``1 / channels_in``."""

        bias: bool = False
        """Whether the projection learns an additive bias."""

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


class ResNet(nn.Module):
    """Pre-activation residual network for 32x32 images.

    Pre-activation ordering (norm, activation, convolution) leaves an
    unbroken identity path from input to loss, which is what lets the
    network train at this depth without warmup tricks.
    """

    class Config(Fig["ResNet"]):
        """Width and block composition of the residual stack."""

        channels_in: int = 3
        """Input image channels."""

        channels_hidden: tuple[int, ...] = (64, 128, 256)
        """Width of each stage; the first also sizes the stem."""

        channels_out: int = 10
        """Output logits, one per class."""

        block: Makeable[nn.Module] | list[Makeable[nn.Module]] = field(
            default_factory=ResidualBlock.Config,
        )
        """Block template, repeated ``blocks_per_stage`` times per stage, or an
        explicit per-stage list of lists flattened in order."""

        blocks_per_stage: int = 2
        """Repeats of the template within each stage. Ignored for a list."""

        activation: ActivationFn = torch.relu
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
        channels = config.channels_hidden[0]
        blocks = _block_grid(
            config.block,
            len(config.channels_hidden),
            config.blocks_per_stage,
        )
        for stage_index, stage_blocks in enumerate(blocks):
            for block_index, block in enumerate(stage_blocks):
                propagate_attr(block, "channels_in", channels, protocol=ChannelsIn)
                channels = config.channels_hidden[stage_index]
                propagate_attr(block, "channels_out", channels, protocol=ChannelsOut)
                # Downsample once per stage, on its first block. The first stage
                # keeps full resolution: 32x32 is already small, and halving it
                # here costs accuracy outright.
                propagate_attr(
                    block,
                    "stride",
                    2 if stage_index and not block_index else 1,
                )
                propagate_attr(block, "activation", config.activation)
                propagate_attr(block, "norm_momentum", config.norm_momentum)
                stages.append(block.make())
            stages.append(nn.Identity())
        self.stages = nn.Sequential(*stages)
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

        channels_in: int = 3
        """Input image channels."""

        channels_hidden: tuple[int, ...] = (128, 384, 512)
        """Width of each convolution block."""

        channels_out: int = 10
        """Output logits, one per class."""

        block: Makeable[nn.Module] | list[Makeable[nn.Module]] = field(
            default_factory=ConvBlock.Config,
        )
        """Block template broadcast across ``channels_hidden``, or one config
        per block."""

        whiten_kernel: int = 2
        """Patch size the whitening layer decomposes."""

        norm_momentum: float = 0.4
        """BatchNorm momentum; high because runs are only a few epochs long."""

        activation: ActivationFn = nn.functional.silu
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

    def __init__(self, config: Config) -> None:
        super().__init__()
        # Rank doubling: PCA yields C*k*k eigenvectors, and the layer emits both
        # a vector and its negation so a following ReLU-like activation can
        # respond to projections of either sign.
        whiten_width = 2 * config.channels_in * config.whiten_kernel**2
        self.whiten = PCAWhiteningConv2d(
            config.channels_in,
            whiten_width,
            config.whiten_kernel,
            padding=0,
            bias=False,
        )
        self.act = _activation(config.activation)
        blocks: list[nn.Module] = []
        channels = whiten_width
        grid = _block_grid(config.block, len(config.channels_hidden), 1)
        for stage_channels, stage_blocks in zip(
            config.channels_hidden,
            grid,
            strict=True,
        ):
            for block in stage_blocks:
                propagate_attr(block, "channels_in", channels, protocol=ChannelsIn)
                channels = stage_channels
                propagate_attr(block, "channels_out", channels, protocol=ChannelsOut)
                propagate_attr(block, "activation", config.activation)
                propagate_attr(block, "norm_momentum", config.norm_momentum)
                blocks.append(block.make())
        self.blocks = nn.Sequential(*blocks)
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
            head_weight = cast("nn.Linear", self.head).weight.data
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


def _activation(activation: ActivationFn) -> TensorFn:
    """Build an activation from a config, or pass a callable through.

    An ``nn.Module`` activation satisfies ``TensorFn``; naming the callable type
    rather than the union keeps ``self.act(x)`` inferring ``Tensor``.
    """
    if isinstance(activation, Makeable):
        # ``Makeable`` is runtime-checkable, so isinstance erases its type
        # parameter: ``make`` reads as returning ``object`` without the cast.
        return cast("TensorFn", activation.make())
    return activation


def _block_grid(
    block: Makeable[nn.Module] | list[Makeable[nn.Module]],
    num_stages: int,
    blocks_per_stage: int,
) -> list[list[Makeable[nn.Module]]]:
    """Group block configs by stage, copying a template so each is distinct.

    A shared config would be mutated once per stage and every block would end up
    carrying the last stage's width.

    Args:
      block: One template to broadcast, or an explicit flat list.
      num_stages: Number of stages to fill.
      blocks_per_stage: Blocks within each stage.

    Returns:
      grid: One list of configs per stage, in stage order.

    Raises:
      ValueError: If an explicit list does not hold exactly one config per block.

    """
    flat: list[Makeable[nn.Module]]
    if isinstance(block, list):
        expected = num_stages * blocks_per_stage
        if len(block) != expected:
            raise ValueError(
                f"block list must hold {expected} configs "
                f"({num_stages} stages x {blocks_per_stage}); got {len(block)}.",
            )
        # ``Makeable`` is a runtime-checkable Protocol, so ty cannot rule out a
        # config that is also a list; the cast states which arm won.
        flat = cast("list[Makeable[nn.Module]]", block)  # pyright: ignore[reportUnnecessaryCast] -- ty cannot narrow the Protocol arm; pyright can
    else:
        flat = [
            copy_tree(block) for _ in range(num_stages) for _ in range(blocks_per_stage)
        ]
    return [
        flat[index * blocks_per_stage : (index + 1) * blocks_per_stage]
        for index in range(num_stages)
    ]
