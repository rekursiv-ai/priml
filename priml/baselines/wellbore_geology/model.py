"""Networks of the wellbore baseline: spatial blocks, backbone and depth posterior."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Self, override

import math

from configgle import Fig, Makeable
from configgle.walk import traverse
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    nbytes,
    reduction_cost,
    resolve_dtype,
    traffic,
)
from priml.math.custom_types import TensorFn
from priml.model.conv import Conv2d
from priml.model.convnext import (
    ConvNeXt,
    ConvNeXtBlock,
    ConvNeXtStage,
)
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    HasOutputGrid,
    PreservesGrid,
    TensorModule,
    infer_same_width,
    propagate_attr,
    validate_same_width,
)
from priml.model.init import InitFn
from priml.model.linear import Linear
from priml.model.norm import BatchNorm, BatchNorm2d, LayerNorm, LayerNorm2d
from priml.model.pool import AvgPool2d
from priml.model.sequential import Sequential
from priml.model.special import Activation, Dropout, Dropout2d, Identity
from priml.model.swiglu import SwiGLU, gelu, silu


if TYPE_CHECKING:
    from safetensors.torch import load_file

else:
    from wrapt import lazy_import

    load_file = lazy_import("safetensors.torch", "load_file")


class ImageNorm(nn.Module):
    """Normalize narrow inputs in the floating state dtype, float32 by default."""

    class Config(PreservesGrid, Fig["ImageNorm"]):
        channels_in: int = -1
        """Width of the normalized image axis."""

        channels_out: int = -1
        """Width preserved by normalization."""

        _: KW_ONLY

        channels_last: bool = False
        """Whether the incoming image is NHWC."""

        bn: BatchNorm2d.Config = field(
            default_factory=lambda: BatchNorm2d.Config(
                eps=1e-4,
                elementwise_affine=True,
            ),
        )
        """Image statistics and learned affine map."""

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            self.bn.channels_in = self.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count state-precision normalization, casts, and running buffers.

            Args:
                seq_len: Spatial positions per example in either layout.
                batch_size: Examples per invocation.
                dtype: Activation dtype.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete normalization work and carried state.

            """
            del kwargs
            narrow = dtype in (torch.float16, torch.bfloat16)
            normalization_dtype = resolve_dtype(self.bn.dtype) if narrow else dtype
            total = cost(
                self.bn,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=normalization_dtype,
            )
            if narrow and normalization_dtype != dtype:
                elements = self.channels_in * seq_len * batch_size
                for phase in ("primal", "adjoint"):
                    total += traffic(
                        phase,
                        "elementwise",
                        elements=2 * elements,
                        dtype=dtype,
                    )
                    total += traffic(
                        phase,
                        "elementwise",
                        elements=2 * elements,
                        dtype=normalization_dtype,
                    )
            return total

    def __init__(self, config: Config) -> None:
        validate_same_width(config)
        super().__init__()
        self.bn = config.bn.make()
        self.channels_last = config.channels_last

    def reset_parameters(self) -> None:
        """Reset both affine parameters and running statistics."""
        self.bn.reset_parameters()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        dtype = x.dtype
        if self.channels_last:
            x = x.permute(0, 3, 1, 2)
        if dtype in (torch.float16, torch.bfloat16):
            if self.bn.running_mean is None:
                raise ValueError("Expected self.bn.running_mean is not None.")
            x = self.bn(x.to(self.bn.running_mean.dtype)).to(dtype)
        else:
            x = self.bn(x)
        return x.permute(0, 2, 3, 1) if self.channels_last else x


class ConvNormAct(nn.Module):
    """Convolve, normalize, activate, then drop complete feature maps."""

    class Config(Fig["ConvNormAct"]):
        channels_in: int = -1
        """Width entering the convolution."""

        channels_out: int = -1
        """Width leaving the convolution."""

        _: KW_ONLY

        conv: Conv2d.Config = field(default_factory=Conv2d.Config)
        """Spatial projection and initialization."""

        norm: Makeable[TensorModule] = field(
            default_factory=lambda: BatchNorm2d.Config(elementwise_affine=True),
        )
        """Normalization after projection."""

        activation: TensorFn = silu
        """Nonlinearity before feature dropout."""

        drop: Dropout2d.Config = field(default_factory=lambda: Dropout2d.Config(p=0.0))
        """Feature dropout."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            self.conv.channels_in = self.channels_in
            self.conv.channels_out = self.channels_out
            propagate_attr(
                self.norm,
                "channels_in",
                self.channels_out,
                protocol=ChannelsIn,
            )
            return super().finalize()

        def output_grid(self, input_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Return the configured convolution's spatial extents.

            Args:
                input_grid: Spatial extents entering the block.

            Returns:
                output: Spatial extents preserved by normalization and activation.

            """
            return self.conv.output_grid(input_grid)

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count convolution followed by normalization, activation and dropout.

            Args:
                input_grid: Spatial extents entering the convolution.
                batch_size: Examples per invocation.
                dtype: Activation dtype.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete composition cost at the projected grid.

            """
            del kwargs
            grid = self.output_grid(input_grid)
            rows = batch_size * math.prod(grid)
            return (
                cost(
                    self.conv,
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    self.norm,
                    seq_len=math.prod(grid),
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(self.activation, channels=self.channels_out * rows, dtype=dtype)
                + cost(
                    self.drop,
                    seq_len=math.prod(grid),
                    batch_size=batch_size,
                    channels=self.channels_out,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.conv = config.conv.make()
        self.bn = config.norm.make()
        self.act = config.activation
        self.drop: Dropout2d = config.drop.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.act(self.bn(self.conv(x))))


class ResidualBlock(nn.Module):
    """Preserve an independently projected residual across two convolutions."""

    class Config(Fig["ResidualBlock"]):
        channels_in: int = -1
        """Width entering both branches."""

        channels_out: int = -1
        """Width of the merged branches."""

        _: KW_ONLY

        conv1: ConvNormAct.Config = field(default_factory=ConvNormAct.Config)
        """First projection, normalization, activation and dropout."""

        conv2: Conv2d.Config = field(default_factory=Conv2d.Config)
        """Second spatial projection."""

        norm: Makeable[TensorModule] = field(
            default_factory=lambda: BatchNorm2d.Config(elementwise_affine=True),
        )
        """Normalization before the residual merge."""

        drop: Dropout2d.Config = field(default_factory=lambda: Dropout2d.Config(p=0.0))
        """Feature dropout on the residual update."""

        activation: TensorFn = silu
        """Nonlinearity after merging branches."""

        skip: Makeable[TensorModule] = field(default_factory=Identity.Config)
        """Residual projection; a pointwise convolution when the widths differ."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            self.conv1.channels_in = self.channels_in
            self.conv1.channels_out = self.channels_out
            self.conv2.channels_in = self.conv2.channels_out = self.channels_out
            propagate_attr(
                self.norm,
                "channels_in",
                self.channels_out,
                protocol=ChannelsIn,
            )
            propagate_attr(
                self.skip,
                name="channels_in",
                value=self.channels_in,
                protocol=ChannelsIn,
            )
            propagate_attr(
                self.skip,
                name="channels_out",
                value=self.channels_out,
                protocol=ChannelsOut,
            )
            return super().finalize()

        def output_grid(self, input_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Propagate matching residual branches' spatial geometry.

            Args:
                input_grid: Spatial extents entering the block.

            Returns:
                output: Spatial extents shared by the merged branches.

            Raises:
                ValueError: Update and shortcut spatial extents disagree.

            """
            grid = self.conv2.output_grid(self.conv1.conv.output_grid(input_grid))
            skip_grid = (
                self.skip.output_grid(input_grid)
                if isinstance(self.skip, HasOutputGrid)
                else input_grid
            )
            if grid != skip_grid:
                raise ValueError("Decoder residual branches must share their grid.")
            return grid

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count both residual branches and the complete merge.

            Args:
                input_grid: Spatial extents entering both branches.
                batch_size: Examples per invocation.
                dtype: Activation dtype.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete branch, merge and activation cost.

            """
            del kwargs
            first_grid = self.conv1.conv.output_grid(input_grid)
            grid = self.output_grid(input_grid)
            positions = math.prod(grid)
            rows = batch_size * positions
            return (
                cost(
                    self.conv1,
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    self.conv2,
                    input_grid=first_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    self.norm,
                    seq_len=positions,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    self.skip,
                    input_grid=input_grid,
                    seq_len=math.prod(input_grid),
                    channels=self.channels_in,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    self.drop,
                    seq_len=positions,
                    batch_size=batch_size,
                    channels=self.channels_out,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=rows * self.channels_out,
                    adjoint=rows * self.channels_out,
                    channels=self.channels_out,
                    rows=rows,
                    inputs=2,
                    dtype=dtype,
                )
                + cost(self.activation, channels=self.channels_out * rows, dtype=dtype)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.conv1 = config.conv1.make()
        self.conv2 = config.conv2.make()
        self.bn2 = config.norm.make()
        self.drop = config.drop.make()
        self.act = config.activation
        self.skip = config.skip.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        update = self.drop(self.bn2(self.conv2(self.conv1(x))))
        if update.shape != residual.shape:
            raise ValueError("Decoder residual branches must share their grid.")
        return self.act(update + residual)


class ConcatUp(nn.Module):
    """Upsample and join a skip map before projecting the joined channels."""

    class Config(Fig["ConcatUp"]):
        channels_in: int = -1
        """Width arriving from the deeper resolution."""

        channels_out: int = -1
        """Width of the decoder result."""

        _: KW_ONLY

        channels_skip: int = -1
        """Width of the encoder skip map."""

        proj: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1),
        )
        """Projection after concatenation."""

        block: ResidualBlock.Config = field(default_factory=ResidualBlock.Config)
        """Residual refinement at the skip resolution."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            self.proj.channels_in = self.channels_in + self.channels_skip
            self.proj.channels_out = self.channels_out
            self.block.channels_in = self.block.channels_out = self.channels_out
            return super().finalize()

        def output_grid(self, skip_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Propagate the joined skip through projection and refinement.

            Args:
                skip_grid: Encoder skip-map spatial extents.

            Returns:
                output: Complete decoder block output extents.

            """
            return self.block.output_grid(self.proj.output_grid(skip_grid))

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            output_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count bilinear resize, channel concatenation and both projections.

            Args:
                input_grid: Spatial extents before resizing.
                output_grid: Actual skip-map spatial extents.
                batch_size: Examples per invocation.
                dtype: Activation dtype.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete decoder invocation cost.

            """
            del kwargs
            elements = (
                batch_size
                * math.prod(output_grid)
                * (self.channels_in + self.channels_skip)
            )
            return (
                bilinear_cost(
                    input_grid=input_grid,
                    output_grid=output_grid,
                    channels=self.channels_in,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + traffic("primal", "selection", elements=2 * elements, dtype=dtype)
                + cost(
                    self.proj,
                    input_grid=output_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    self.block,
                    input_grid=self.proj.output_grid(output_grid),
                    batch_size=batch_size,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.proj = config.proj.make()
        self.block = config.block.make()

    @override
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        upsampled = functional.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.block(self.proj(torch.cat((upsampled, skip), dim=1)))


class AddUp(nn.Module):
    """Merge full-resolution features by addition before residual refinement."""

    class Config(Fig["AddUp"]):
        channels_in: int = -1
        """Width of both input branches and the result."""

        block: ResidualBlock.Config = field(default_factory=ResidualBlock.Config)
        """Residual refinement of the merged features."""

        @override
        def finalize(self) -> Self:
            self.block.channels_in = self.block.channels_out = self.channels_in
            return super().finalize()

        def output_grid(self, skip_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Propagate the resized skip through residual refinement.

            Args:
                skip_grid: Encoder skip-map spatial extents.

            Returns:
                output: Complete decoder block output extents.

            """
            return self.block.output_grid(skip_grid)

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            output_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count bilinear resize, skip addition and residual refinement.

            Args:
                input_grid: Spatial extents before resizing.
                output_grid: Actual skip-map spatial extents.
                batch_size: Examples per invocation.
                dtype: Activation dtype.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete decoder invocation cost.

            """
            del kwargs
            rows = batch_size * math.prod(output_grid)
            return (
                bilinear_cost(
                    input_grid=input_grid,
                    output_grid=output_grid,
                    channels=self.channels_in,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=rows * self.channels_in,
                    adjoint=rows * self.channels_in,
                    channels=self.channels_in,
                    rows=rows,
                    inputs=2,
                    dtype=dtype,
                )
                + cost(
                    self.block,
                    input_grid=output_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.block = config.block.make()

    @override
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        upsampled = functional.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if upsampled.shape != skip.shape:
            raise ValueError("Additive skip branches must have equal shapes.")
        return self.block(upsampled + skip)


def bilinear_cost(
    *,
    input_grid: tuple[int, ...],
    output_grid: tuple[int, ...],
    channels: int,
    batch_size: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Count four-tap bilinear interpolation and scatter-add gradients.

    Args:
        input_grid: Spatial extents before resizing.
        output_grid: Actual target spatial extents.
        channels: Feature width.
        batch_size: Examples per invocation.
        dtype: Activation dtype.

    Returns:
        execution: Weighted interpolation and input-gradient accumulation.

    """
    outputs = batch_size * channels * math.prod(output_grid)
    inputs = batch_size * channels * math.prod(input_grid)
    return traffic(
        "primal",
        "selection",
        elements=5 * outputs,
        flops=7 * outputs,
        dtype=dtype,
    ) + traffic(
        "adjoint",
        "selection",
        elements=9 * outputs + inputs,
        flops=8 * outputs,
        dtype=dtype,
    )


def backbone_mlp(*, channels_in: int = -1, channels_out: int = -1) -> SwiGLU.Config:
    """Declare the source ungated GELU block and both affine draw policies.

    Args:
      channels_in: Input width, or -1 inherited from the enclosing block.
      channels_out: Output width, or -1 inherited from the enclosing block.

    Returns:
      config: Shared ungated feed-forward with the source affine initialization.

    """
    return SwiGLU.Config(
        channels_in=channels_in,
        channels_out=channels_out,
        gate=False,
        expansion=4.0,
        round_to=1,
        act=gelu,
        up_proj=Linear.Config(bias=True, init_bias=None),
        down_proj=Linear.Config(bias=True, init_bias=None),
    )


def backbone_network() -> ConvNeXt.Config:
    """Declare source affine constructors before callers customize the tree.

    Returns:
        network: Complete independently configurable backbone defaults.

    """
    network = ConvNeXt.Config(blocks_per_stage=(3, 3, 27, 3))
    network.stem.conv = Conv2d.Config(
        kernel_size=4,
        stride=4,
        padding=0,
        bias=True,
        init_bias=None,
    )
    assert isinstance(network.stage, ConvNeXtStage.Config)
    assert isinstance(network.stage.downsample, Sequential.Config)
    assert isinstance(network.stage.downsample.elements, list)
    network.stage.downsample.elements[1] = Conv2d.Config(
        kernel_size=2,
        stride=2,
        padding=0,
        bias=True,
        init_bias=None,
    )
    assert isinstance(network.stage.block, ConvNeXtBlock.Config)
    network.stage.block.conv_dw = Conv2d.Config(
        kernel_size=7,
        padding=3,
        bias=True,
        init_bias=None,
    )
    network.stage.block.mlp = backbone_mlp()
    return network


class Backbone:
    """Allocate the full classifier-free backbone, then retain converted stages."""

    class Config(Fig["Backbone"]):
        network: ConvNeXt.Config = field(default_factory=backbone_network)
        """Complete patch stem, spatial stages and postpool normalization."""

        checkpoint: Path | str | None = None
        """Pinned safetensors checkpoint; None initializes from scratch.

        A string may use the `{dataset_directory}` template, which the owning family
        resolves against its dataset directory before construction.
        """

        init_weight: InitFn = partial(nn.init.trunc_normal_, std=0.02)
        """Final backbone initializer after all default affine draws."""

    def __init__(self, config: Config) -> None:
        network = config.network.make()
        for module in network.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                config.init_weight(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if config.checkpoint is not None:
            state = load_file(str(config.checkpoint), device="cpu")
            # The classifier is absent in the published num_classes=0 constructor.
            state.pop("head.fc.weight", None)
            state.pop("head.fc.bias", None)
            network.load_timm_state_dict(state)
        self.stages = network.stages
        for stage in self.stages:
            replacements: list[tuple[str, ImageNorm]] = []
            for name, module in stage.named_modules():
                if isinstance(module, LayerNorm):
                    if module.bias is None:
                        raise ValueError("Backbone normalization must be affine.")
                    norm = ImageNorm.Config()
                    norm.channels_in = module.normalized_shape[0]
                    norm.channels_last = not isinstance(module, LayerNorm2d)
                    norm.bn.dtype = module.weight.dtype
                    norm.bn.device = module.weight.device
                    replacement = norm.make()
                    with torch.no_grad():
                        replacement.bn.weight.copy_(module.weight)
                        replacement.bn.bias.copy_(module.bias)
                    replacements.append((name, replacement))
            for name, replacement in replacements:
                stage.set_submodule(name, replacement)


def retained_stage_configs(config: Backbone.Config) -> list[ConvNeXtStage.Config]:
    """Copy the executed stages with the same LayerNorm-to-ImageNorm conversion.

    The complete temporary ConvNeXt stem and head are discarded by Backbone;
    neither their parameters nor their work belong to the retained U-Net.

    Args:
        config: Finalized complete pretrained allocation configuration.

    Returns:
        stages: Independent cost-only configurations of the retained modules.

    """
    assert isinstance(config.network.stage, list)
    stages = [stage.copy_tree() for stage in config.network.stage]
    for stage in stages:
        assert isinstance(stage.block, list)
        for child in [stage.downsample, *stage.block]:
            for match in traverse(child, LayerNorm.Config):
                norm = ImageNorm.Config(
                    channels_in=match.config.channels_in,
                    channels_last=not isinstance(match.config, LayerNorm2d.Config),
                )
                norm.bn.dtype = match.config.dtype
                norm.bn.device = match.config.device
                norm.finalize()
                match.replace(norm)
    return stages


class PosteriorUNet(nn.Module):
    """Decode all four ConvNeXt stages into the complete input grid."""

    class Config(Fig["PosteriorUNet"]):
        channels_in: int = -1
        """Input feature-map width."""

        channels_out: int = -1
        """Full-resolution decoder width."""

        _: KW_ONLY

        channels_hidden: tuple[int, ...] = (96, 192, 384, 768)
        """Width at each encoder resolution."""

        stem: ResidualBlock.Config = field(
            default_factory=lambda: ResidualBlock.Config(
                skip=Conv2d.Config(kernel_size=1),
            ),
        )
        """Full-resolution encoder skip, projecting its input width pointwise."""

        stem_down: AvgPool2d.Config = field(
            default_factory=lambda: AvgPool2d.Config(
                kernel_size=(2, 4),
                stride=(2, 4),
                ceil_mode=True,
            ),
        )
        """Reduction before the first backbone stage."""

        to_backbone: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(
                kernel_size=1,
                bias=True,
                init_bias=None,
            ),
        )
        """Projection from the residual stem to the backbone width."""

        backbone: Backbone.Config = field(default_factory=Backbone.Config)
        """Complete pretrained allocation and retained native encoder stages."""

        up_blocks: list[ConcatUp.Config] = field(
            default_factory=lambda: [ConcatUp.Config() for _ in range(3)],
        )
        """Ordered decoder blocks, deepest first; one per encoder resolution after
        the first."""

        to_full_resolution: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1),
        )
        """Projection to the full-resolution feature width."""

        full_up_block: AddUp.Config = field(default_factory=AddUp.Config)
        """Addition of the full-resolution stem skip."""

        out: ResidualBlock.Config = field(default_factory=ResidualBlock.Config)
        """Final spatial refinement."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            self.stem.channels_in = self.channels_in
            self.stem.channels_out = self.channels_out
            self.to_backbone.channels_in = self.channels_out
            self.to_backbone.channels_out = self.channels_hidden[0]
            self.backbone.network.channels_hidden = self.channels_hidden
            for up, incoming, skip in zip(
                self.up_blocks,
                reversed(self.channels_hidden[1:]),
                reversed(self.channels_hidden[:-1]),
                strict=False,
            ):
                up.channels_in = incoming
                up.channels_skip = up.channels_out = skip
            self.to_full_resolution.channels_in = self.channels_hidden[0]
            self.to_full_resolution.channels_out = self.channels_out
            self.full_up_block.channels_in = self.channels_out
            self.out.channels_in = self.out.channels_out = self.channels_out
            return super().finalize()

        def output_grid(self, input_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Propagate the actual full-resolution skip to the output projection.

            Args:
                input_grid: Spatial extents entering the U-Net.

            Returns:
                output: Spatial extents produced by the output projection.

            """
            stem_grid = self.stem.output_grid(input_grid)
            return self.out.output_grid(self.full_up_block.output_grid(stem_grid))

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count retained encoder stages, complete skips and decoder work.

            Args:
                input_grid: Full input spatial extents.
                batch_size: Examples per invocation.
                dtype: Activation dtype.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Whole U-Net cost without the discarded backbone stem/head.

            """
            del kwargs
            total = cost(
                self.stem,
                input_grid=input_grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            stem_grid = self.stem.output_grid(input_grid)
            grid = self.stem_down.output_grid(stem_grid)
            total += cost(
                self.stem_down,
                input_grid=stem_grid,
                channels=self.channels_out,
                batch_size=batch_size,
                dtype=dtype,
            )
            total += cost(
                self.to_backbone,
                input_grid=grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            grid = self.to_backbone.output_grid(grid)
            stages = retained_stage_configs(self.backbone)
            skips: list[tuple[int, ...]] = []
            for index, stage in enumerate(stages):
                total += cost(
                    stage,
                    input_grid=grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                grid = stage.output_grid(grid)
                if index < len(stages) - 1:
                    skips.append(grid)
            for block, skip in zip(self.up_blocks, reversed(skips), strict=True):
                total += cost(
                    block,
                    input_grid=grid,
                    output_grid=skip,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                grid = block.output_grid(skip)
            total += cost(
                self.to_full_resolution,
                input_grid=grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            grid = self.to_full_resolution.output_grid(grid)
            total += cost(
                self.full_up_block,
                input_grid=grid,
                output_grid=stem_grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            return total + cost(
                self.out,
                input_grid=self.full_up_block.output_grid(stem_grid),
                batch_size=batch_size,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        if len(config.up_blocks) != len(config.channels_hidden) - 1:
            raise ValueError(
                "The U-Net requires one decoder block per skip resolution.",
            )
        super().__init__()
        backbone = config.backbone.make()
        self.stem = config.stem.make()
        self.stem_down = config.stem_down.make()
        self.to_backbone = config.to_backbone.make()
        self.backbone_stages = backbone.stages
        self.up_blocks = nn.ModuleList(block.make() for block in config.up_blocks)
        self.to_full_resolution = config.to_full_resolution.make()
        self.full_up_block = config.full_up_block.make()
        self.out = config.out.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        stem = self.stem(x)
        x = self.to_backbone(self.stem_down(stem))
        skips: list[Tensor] = []
        for index, stage in enumerate(self.backbone_stages):
            x = stage(x)
            if index < len(self.backbone_stages) - 1:
                skips.append(x)
        for block, skip in zip(self.up_blocks, reversed(skips), strict=True):
            x = block(x, skip)
        return self.out(self.full_up_block(self.to_full_resolution(x), stem))


class SequenceModel(nn.Module):
    """Infer depth from the complete horizontal-by-typewell feature grid."""

    class Config(Fig["SequenceModel"]):
        channels_in: int = -1
        """Number of input feature maps; -1 inherits the family's feature planes."""

        channels_hidden: int = 32
        """Width at full resolution and at the posterior heads."""

        kernel_size: int = -1
        """Decoder residual kernel extent; -1 keeps each block's own."""

        dropout: float = -1.0
        """Feature dropout probability at every decoder site; negative keeps each
        site's own."""

        reference_columns: int = -1
        """Number of candidate depth bins; -1 inherits the family's features."""

        reference_window: float = -1.0
        """Half-width of the candidate depth interval; negative inherits the
        family's features."""

        target_stats: tuple[float, float] | None = None
        """Regression normalization center and scale; None inherits the family's
        features."""

        channels_last: bool = True
        """Store the CUDA image backbone with contiguous channel vectors."""

        unet: PosteriorUNet.Config = field(default_factory=PosteriorUNet.Config)
        """Spatial feature encoder and decoder."""

        proj_out: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(
                kernel_size=1,
                bias=True,
                init_bias=None,
            ),
        )
        """Shared alignment and regression logits."""

        unet_gr_rmse_head: Sequential.Config = field(
            default_factory=lambda: Sequential.Config(
                elements=[
                    Linear.Config(bias=True, init_bias=None),
                    BatchNorm.Config(elementwise_affine=True),
                    Activation.Config(),
                    Dropout.Config(p=0.0),
                    Linear.Config(bias=True, init_bias=None),
                ],
            ),
        )
        """Auxiliary gamma-ray mismatch prediction head."""

        @override
        def finalize(self) -> Self:
            if self.unet.channels_in == -1:
                self.unet.channels_in = self.channels_in
            if self.unet.channels_out == -1:
                self.unet.channels_out = self.channels_hidden
            for block in (
                self.unet.stem,
                self.unet.full_up_block.block,
                self.unet.out,
                *(up.block for up in self.unet.up_blocks),
            ):
                if self.kernel_size > 0:
                    block.conv1.conv.kernel_size = block.conv2.kernel_size = (
                        self.kernel_size
                    )
                if self.dropout >= 0:
                    block.conv1.drop.p = block.drop.p = self.dropout
            if self.proj_out.channels_in == -1:
                self.proj_out.channels_in = self.channels_hidden
            if self.proj_out.channels_out == -1:
                self.proj_out.channels_out = 1
            head = self.unet_gr_rmse_head
            if head.channels_in == -1:
                head.channels_in = self.channels_hidden
            if head.channels_out == -1:
                head.channels_out = 1
            elements = (
                head.elements if isinstance(head.elements, list) else [head.elements]
            )
            first = elements[0] if elements else None
            if isinstance(first, Linear.Config) and first.channels_out == -1:
                first.channels_out = 2 * self.channels_hidden
            if self.dropout >= 0:
                for element in elements:
                    if isinstance(element, Dropout.Config):
                        element.p = self.dropout
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count the spatial model, shared logits and both posterior heads.

            Args:
                input_grid: Full horizontal-by-typewell spatial extents.
                batch_size: Examples per invocation.
                dtype: Activation dtype.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete model cost, including nonpersistent axis buffers.

            """
            del kwargs
            unet_grid = self.unet.output_grid(input_grid)
            height, width = self.proj_out.output_grid(unet_grid)
            rows = batch_size * height
            pixels = rows * width
            total = cost(
                self.unet,
                input_grid=input_grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            total += cost(
                self.proj_out,
                input_grid=unet_grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            # Stable softmax: row max, subtraction, exp, row sum and division.
            total += elementwise_cost(
                primal=3 * pixels,
                adjoint=3 * pixels,
                channels=width,
                rows=rows,
                dtype=dtype,
            )
            total += reduction_cost(
                input_elements=pixels,
                output_groups=rows,
                dtype=dtype,
            ).tile(2)
            total += reduction_cost(
                input_elements=pixels,
                output_groups=rows,
                dtype=dtype,
                phase="adjoint",
            )
            # The frozen depth axis has no gradient; the two logit names alias.
            total += elementwise_cost(
                primal=pixels,
                adjoint=pixels,
                channels=width,
                rows=rows,
                inputs=2,
                dtype=dtype,
            )
            total += reduction_cost(
                input_elements=pixels,
                output_groups=rows,
                dtype=dtype,
            )
            # Average only the typewell axis before the auxiliary affine head.
            feature_rows = batch_size * unet_grid[0]
            feature_pixels = batch_size * math.prod(unet_grid)
            total += reduction_cost(
                input_elements=feature_pixels * self.channels_hidden,
                output_groups=feature_rows * self.channels_hidden,
                dtype=dtype,
            )
            total += traffic(
                "primal",
                "elementwise",
                elements=2 * feature_rows * self.channels_hidden,
                flops=feature_rows * self.channels_hidden,
                dtype=dtype,
            )
            total += traffic(
                "adjoint",
                "elementwise",
                elements=(feature_pixels + feature_rows) * self.channels_hidden,
                flops=feature_pixels * self.channels_hidden,
                dtype=dtype,
            )
            total += cost(
                self.unet_gr_rmse_head,
                channels=self.channels_hidden,
                seq_len=unet_grid[0],
                batch_size=batch_size,
                dtype=dtype,
            )
            total += elementwise_cost(
                primal=4 * feature_rows,
                adjoint=4 * feature_rows,
                channels=1,
                rows=feature_rows,
                dtype=dtype,
            )
            return total + Cost(
                bytes_state=nbytes(2 * self.reference_columns, torch.float32),
            )

    def __init__(self, config: Config) -> None:
        if min(config.reference_columns, config.reference_window) < 0:
            raise ValueError("The model requires its reference window geometry.")
        if config.target_stats is None:
            raise ValueError("The model requires its target statistics.")
        super().__init__()
        self.config = config
        self.channels_last = config.channels_last
        self.unet = config.unet.make()
        self.proj_out = config.proj_out.make()
        self.typewell_tvt_axis: Tensor
        window, columns = config.reference_window, config.reference_columns
        target_mean, target_std = config.target_stats
        raw_axis = torch.linspace(
            -window + window / columns,
            window - window / columns,
            columns,
            dtype=torch.float32,
            device="cpu",
        )
        self.register_buffer(
            "typewell_tvt_axis_raw",
            raw_axis,
            persistent=False,
        )
        self.register_buffer(
            "typewell_tvt_axis",
            (raw_axis - target_mean) / target_std,
            persistent=False,
        )
        self.unet_gr_rmse_head = config.unet_gr_rmse_head.make()
        self.register_load_state_dict_pre_hook(_rename_legacy_head)

    def format_backbone(self, device: torch.device) -> None:
        """Store the CUDA image backbone with contiguous channel vectors.

        Args:
          device: Placement of the model, which decides whether the layout applies.

        """
        if self.channels_last and device.type == "cuda":
            self.unet.to(memory_format=torch.channels_last)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Return depth expectation and complete posterior diagnostics."""
        features = self.unet(x)
        logits = self.proj_out(features).squeeze(1)
        probabilities = functional.softmax(logits, dim=-1)
        axis = self.typewell_tvt_axis.to(
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        prediction = (probabilities * axis.reshape(1, 1, -1)).sum(dim=-1)
        pooled = features.mean(dim=-1).transpose(1, 2)
        batch, rows, channels = pooled.shape
        mismatch = functional.softplus(
            self.unet_gr_rmse_head(pooled.reshape(batch * rows, channels)),
        ).reshape(batch, rows)
        return prediction, {
            "alignment_logits": logits,
            "regression_logits": logits,
            "unet_GR_RMSE_pred": mismatch,
        }


def _rename_legacy_head(
    module: nn.Module,
    state_dict: dict[str, Tensor],
    prefix: str,
    *args: object,
) -> None:
    """Accept checkpoints that name the output projection ``alignment_head``."""
    del module, args
    for name in ("weight", "bias"):
        legacy = f"{prefix}alignment_head.{name}"
        if legacy in state_dict:
            state_dict[f"{prefix}proj_out.{name}"] = state_dict.pop(legacy)
