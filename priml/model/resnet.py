"""Post-activation ResNet: basic residual blocks, stages and a pooled linear head."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from functools import partial
from typing import TYPE_CHECKING, Self, override

import math
import re

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, cost, elementwise_cost, map_cost, set_cost
from priml.math.custom_types import TensorFn
from priml.model.conv import Conv2d
from priml.model.custom_types import HasResetParameters
from priml.model.init import InitFn, call_init
from priml.model.linear import Linear
from priml.model.norm import BatchNorm2d
from priml.model.pool import avg_pool_cost, max_pool_cost
from priml.model.state_dict import remap_state_dict, validate_state_dict


if TYPE_CHECKING:
    from collections.abc import Mapping


@set_cost(map_cost(primal=1, adjoint=1))
def relu_inplace(x: Tensor) -> Tensor:
    """Rectify in place, with its elementwise cost."""
    return functional.relu(x, inplace=True)


class ResNetBlock(nn.Module):
    """Two convolutional residual branches with postaddition activation."""

    class Config(Fig["ResNetBlock"]):
        channels_in: int = -1
        """Width entering the block."""

        channels_out: int = -1
        """Width leaving the block."""

        _: KW_ONLY

        stride: int = 1
        """Spatial stride of the first convolution and of the shortcut projection."""

        conv1: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(
                padding=1,
                init_weight=partial(
                    nn.init.kaiming_normal_,
                    mode="fan_out",
                    nonlinearity="relu",
                ),
            ),
        )
        """First three-by-three convolution."""

        norm1: BatchNorm2d.Config = field(
            default_factory=lambda: BatchNorm2d.Config(elementwise_affine=True),
        )
        """Normalization before the first activation."""

        conv2: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(
                padding=1,
                init_weight=partial(
                    nn.init.kaiming_normal_,
                    mode="fan_out",
                    nonlinearity="relu",
                ),
            ),
        )
        """Second three-by-three convolution."""

        norm2: BatchNorm2d.Config = field(
            default_factory=lambda: BatchNorm2d.Config(elementwise_affine=True),
        )
        """Normalization before the residual addition."""

        projection: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(
                kernel_size=1,
                padding=0,
                init_weight=partial(
                    nn.init.kaiming_normal_,
                    mode="fan_out",
                    nonlinearity="relu",
                ),
            ),
        )
        """Shortcut projection used when width or stride changes."""

        norm_projection: BatchNorm2d.Config = field(
            default_factory=lambda: BatchNorm2d.Config(elementwise_affine=True),
        )
        """Normalization of the projected shortcut."""

        activation: TensorFn = relu_inplace
        """Inplace rectification before and after the residual addition."""

        init_scale_residual: InitFn | None = None
        """Initializer of the second normalization's scale, set last on every reset.

        Zeroing it makes each block start as the identity; None keeps the
        normalization's own initialization.
        """

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            self.conv1.channels_in = self.channels_in
            self.conv1.channels_out = self.conv2.channels_in = (
                self.conv2.channels_out
            ) = self.channels_out
            self.conv1.stride = self.stride
            self.norm1.channels_in = self.norm2.channels_in = (
                self.norm_projection.channels_in
            ) = self.channels_out
            self.projection.channels_in = self.channels_in
            self.projection.channels_out = self.channels_out
            self.projection.stride = self.stride
            return super().finalize()

        def projects(self) -> bool:
            """Whether the shortcut needs a projection to match the residual branch."""
            return self.stride != 1 or self.channels_in != self.channels_out

        def output_grid(self, input_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Return the spatial extents after both residual convolutions."""
            return self.conv2.output_grid(self.conv1.output_grid(input_grid))

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count both normalized projections and the actual residual shortcut.

            Args:
                input_grid: Spatial extents entering the residual block.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Whole training work and independently owned parameters.

            """
            del kwargs
            grid = input_grid
            total = Cost()
            for projection, norm in (
                (self.conv1, self.norm1),
                (self.conv2, self.norm2),
            ):
                total += cost(
                    projection,
                    input_grid=grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                grid = projection.output_grid(grid)
                total += cost(
                    norm,
                    seq_len=math.prod(grid),
                    batch_size=batch_size,
                    dtype=dtype,
                )
                total += cost(
                    self.activation,
                    channels=batch_size * math.prod(grid) * self.channels_out,
                    dtype=dtype,
                )
            if self.projects():
                total += cost(
                    self.projection,
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                total += cost(
                    self.norm_projection,
                    seq_len=math.prod(grid),
                    batch_size=batch_size,
                    dtype=dtype,
                )
            elements = batch_size * math.prod(grid) * self.channels_out
            return total + elementwise_cost(
                primal=elements,
                adjoint=elements,
                channels=elements,
                inputs=2,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.conv1 = config.conv1.make()
        self.norm1 = config.norm1.make()
        self.conv2 = config.conv2.make()
        self.norm2 = config.norm2.make()
        self.downsample = (
            nn.Sequential(config.projection.make(), config.norm_projection.make())
            if config.projects()
            else None
        )
        self.activation = config.activation
        self.init_scale_residual = config.init_scale_residual
        self._init_scale_residual()

    def reset_parameters(self) -> None:
        """Reset learned state and running statistics, then the residual scale."""
        for module in (self.conv1, self.norm1, self.conv2, self.norm2):
            module.reset_parameters()
        for child in self.downsample or ():
            child.reset_parameters()
        self._init_scale_residual()

    def _init_scale_residual(self) -> None:
        if self.init_scale_residual is not None and self.norm2.affine:
            call_init(self.init_scale_residual, self.norm2.weight)

    @override
    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        hidden = self.activation(self.norm1(self.conv1(x)))
        hidden = self.norm2(self.conv2(hidden))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        hidden += shortcut
        return self.activation(hidden)


class ResNetStage(nn.Sequential):
    """Ordered residual blocks of one stage, the first of which may stride."""

    class Config(Fig["ResNetStage"]):
        channels_in: int = -1
        """Width entering the stage."""

        channels_out: int = -1
        """Width of every residual block's output."""

        _: KW_ONLY

        stride: int = 1
        """Stride of the first block when `block` is a template."""

        blocks_per_stage: int = 2
        """Number of residual blocks when `block` is a template."""

        block: ResNetBlock.Config | list[ResNetBlock.Config] = field(
            default_factory=ResNetBlock.Config,
        )
        """Residual block template or explicit ordered block configurations."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            if isinstance(self.block, list):
                blocks = self.block
            else:
                blocks = [self.block.copy_tree() for _ in range(self.blocks_per_stage)]
                for block in blocks:
                    block.stride = 1
                blocks[0].stride = self.stride
            for index, block in enumerate(blocks):
                block.channels_in = (
                    self.channels_in if index == 0 else self.channels_out
                )
                block.channels_out = self.channels_out
            self.block = blocks
            return super().finalize()

        def output_grid(self, input_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Return the spatial extents after every residual block."""
            assert isinstance(self.block, list)
            grid = input_grid
            for block in self.block:
                grid = block.output_grid(grid)
            return grid

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count every residual block at its own grid.

            Args:
                input_grid: Spatial extents entering the stage.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Whole training work and independently owned parameters.

            """
            del kwargs
            assert isinstance(self.block, list)
            total = Cost()
            grid = input_grid
            for block in self.block:
                total += cost(
                    block,
                    input_grid=grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                grid = block.output_grid(grid)
            return total

    def __init__(self, config: Config) -> None:
        assert isinstance(config.block, list)
        super().__init__(*(block.make() for block in config.block))

    def reset_parameters(self) -> None:
        """Reset every residual block."""
        for block in self:
            if isinstance(block, HasResetParameters):
                block.reset_parameters()


class ResNet(nn.Module):
    """Stem, four residual stages, global average pooling and a linear head."""

    class Config(Fig["ResNet"]):
        channels_in: int = -1
        """Image channels entering the stem."""

        channels_out: int = -1
        """Logits leaving the head."""

        _: KW_ONLY

        stem: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(
                channels_out=64,
                kernel_size=7,
                stride=2,
                padding=3,
                init_weight=partial(
                    nn.init.kaiming_normal_,
                    mode="fan_out",
                    nonlinearity="relu",
                ),
            ),
        )
        """Seven-by-seven stem convolution."""

        norm: BatchNorm2d.Config = field(
            default_factory=lambda: BatchNorm2d.Config(elementwise_affine=True),
        )
        """Stem normalization before rectification and max pooling."""

        channels_hidden: tuple[int, ...] = (64, 128, 256, 512)
        """Output widths of the four residual stages."""

        blocks_per_stage: tuple[int, ...] = (3, 4, 6, 3)
        """Basic blocks of each stage when `stage` is a template."""

        stage: ResNetStage.Config | list[ResNetStage.Config] = field(
            default_factory=ResNetStage.Config,
        )
        """Stage template, or explicit stages that own their blocks and strides.

        A template's first copy keeps stride one; later copies halve the grid.
        """

        proj_out: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True),
        )
        """Linear logits after adaptive average pooling."""

        activation: TensorFn = relu_inplace
        """Inplace stem rectification."""

        @override
        def finalize(self) -> Self:
            self.stem.channels_in = self.channels_in
            self.norm.channels_in = self.stem.channels_out
            self.proj_out.channels_in = self.channels_hidden[-1]
            self.proj_out.channels_out = self.channels_out
            templated = not isinstance(self.stage, list)
            stages = (
                self.stage
                if isinstance(self.stage, list)
                else [self.stage.copy_tree() for _ in self.channels_hidden]
            )
            if len(stages) != len(self.channels_hidden):
                raise ValueError("ResNet width and stage counts must match.")
            width = self.stem.channels_out
            for index, (stage, channels) in enumerate(
                zip(stages, self.channels_hidden, strict=True),
            ):
                stage.channels_in, stage.channels_out = width, channels
                if templated:
                    stage.blocks_per_stage = self.blocks_per_stage[index]
                    stage.stride = 2 if index else 1
                width = channels
            self.stage = stages
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count the stem, every residual grid and the pooled head.

            Args:
                input_grid: Spatial extents entering the image stem.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Whole training work and independently owned parameters.

            """
            del kwargs
            assert isinstance(self.stage, list)
            grid = self.stem.output_grid(input_grid)
            total = cost(
                self.stem,
                input_grid=input_grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            total += cost(
                self.norm,
                seq_len=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
            )
            total += cost(
                self.activation,
                channels=batch_size * math.prod(grid) * self.stem.channels_out,
                dtype=dtype,
            )
            grid = tuple((extent + 1) // 2 for extent in grid)
            total += max_pool_cost(
                channels=self.stem.channels_out,
                kernel_size=3,
                rows=batch_size * math.prod(grid),
                dtype=dtype,
            )
            for stage in self.stage:
                total += cost(
                    stage,
                    input_grid=grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                grid = stage.output_grid(grid)
            return (
                total
                + avg_pool_cost(
                    channels=self.channels_hidden[-1],
                    positions=math.prod(grid),
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(self.proj_out, seq_len=1, batch_size=batch_size, dtype=dtype)
            )

    def __init__(self, config: Config) -> None:
        assert isinstance(config.stage, list)
        super().__init__()
        self.conv1 = config.stem.make()
        self.bn1 = config.norm.make()
        self.activation = config.activation
        self.stages = nn.ModuleList([stage.make() for stage in config.stage])
        self.fc = config.proj_out.make()

    def reset_parameters(self) -> None:
        """Reset the complete model state in place."""
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        for stage in self.stages:
            stage.reset_parameters()
        self.fc.reset_parameters()

    @override
    def forward(self, x: Tensor) -> Tensor:
        hidden = functional.max_pool2d(
            self.activation(self.bn1(self.conv1(x))),
            3,
            stride=2,
            padding=1,
        )
        for stage in self.stages:
            hidden = stage(hidden)
        return self.fc(functional.adaptive_avg_pool2d(hidden, 1).flatten(1))

    def load_torchvision_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Load a complete torchvision-layout checkpoint, or this module's own names.

        The renaming is bijective and every name, shape and dtype is validated
        before any tensor is loaded.

        Args:
            state: Complete parameters and running statistics.

        Raises:
            ValueError: Two names collide, or names, shapes or dtypes differ.

        """
        mapped = remap_state_dict(state, _native_name)
        validate_state_dict(mapped, expected=self.state_dict())
        self.load_state_dict(mapped)


def _native_name(name: str) -> str:
    target = re.sub(
        r"^layer([1-4])\.",
        lambda match: f"stages.{int(match[1]) - 1}.",
        name,
    )
    if target.startswith("stages."):
        target = target.replace(".bn1.", ".norm1.").replace(".bn2.", ".norm2.")
    return target
