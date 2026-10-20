"""Two-dimensional VNet: a residual encoder and skip decoder with channel dropout."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, override

import math

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    traffic,
)
from priml.math.custom_types import TensorFn
from priml.model.conv import Conv2d, ConvTranspose2d
from priml.model.custom_types import ChannelsIn, TensorModule, propagate_attr
from priml.model.norm import BatchNorm2d
from priml.model.special import Dropout2d
from priml.model.swiglu import elu


class ConvNorm(nn.Module):
    """Compose a spatial convolution and an injectable channel normalization."""

    class Config(Fig["ConvNorm"]):
        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count the spatial projection and its output normalization.

            Args:
                input_grid: Input spatial extents.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete work and parameter ownership.

            """
            del kwargs
            return cost(
                self.conv,
                input_grid=input_grid,
                batch_size=batch_size,
                dtype=dtype,
            ) + cost(
                self.norm,
                seq_len=math.prod(self.conv.output_grid(input_grid)),
                batch_size=batch_size,
                dtype=dtype,
            )

        channels_in: int = -1
        """Width entering the spatial convolution."""

        channels_out: int = -1
        """Width of the convolution output and normalization."""

        _: KW_ONLY

        conv: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(
                kernel_size=5,
                padding=2,
                bias=True,
            ),
        )
        """Spatial projection and its initializer."""

        norm: Makeable[TensorModule] = field(
            default_factory=lambda: BatchNorm2d.Config(
                elementwise_affine=True,
            ),
        )
        """Normalization after the spatial projection."""

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
                name="channels_in",
                value=self.channels_out,
                protocol=ChannelsIn,
            )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.conv = config.conv.make()
        self.norm = config.norm.make()

    def reset_parameters(self) -> None:
        """Reset the convolution and normalization."""
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        return self.norm(self.conv(x), **kwargs)


class VNetInput(nn.Module):
    """Add repeated input channels to a normalized convolution before activation."""

    class Config(Fig["VNetInput"]):
        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count normalized projection, repeated input, residual and activation.

            Args:
                input_grid: Input spatial extents.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete work and parameter ownership.

            """
            del kwargs
            rows = batch_size * math.prod(input_grid)
            return (
                cost(
                    self.layer,
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(self.activation, channels=rows * self.channels_out, dtype=dtype)
                + elementwise_cost(
                    primal=rows * self.channels_out,
                    adjoint=rows * self.channels_out,
                    channels=self.channels_out,
                    rows=rows,
                    inputs=2,
                    dtype=dtype,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=2 * rows * self.channels_out,
                    dtype=dtype,
                )
                + traffic(
                    "adjoint",
                    "reduction",
                    elements=rows * (self.channels_in + self.channels_out),
                    flops=rows * (self.channels_out - self.channels_in),
                    dtype=dtype,
                )
            )

        channels_in: int = -1
        """Image input width."""

        channels_out: int = -1
        """First residual width, divisible by the image input width."""

        _: KW_ONLY

        layer: ConvNorm.Config = field(default_factory=ConvNorm.Config)
        """Convolution and normalization before the input residual."""

        activation: TensorFn = elu
        """Activation after adding the repeated image channels."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            self.layer.channels_in = self.channels_in
            self.layer.channels_out = self.channels_out
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.channels_out % config.channels_in:
            raise ValueError("Input width must divide the first residual width.")
        super().__init__()
        self.layer = config.layer.make()
        self.repeats = config.channels_out // config.channels_in
        self.activation = config.activation

    def reset_parameters(self) -> None:
        """Reset the input convolution and normalization."""
        self.layer.reset_parameters()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.activation(
            self.layer(x) + x.repeat([1, self.repeats, 1, 1]),
        )


class VNetDown(nn.Module):
    """Downsample, refine, and add the downsampled residual before activation."""

    class Config(Fig["VNetDown"]):
        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count downsampling and every independently owned refinement layer.

            Args:
                input_grid: Input spatial extents.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete work and parameter ownership.

            """
            del kwargs
            grid = self.proj.conv.output_grid(input_grid)
            rows = batch_size * math.prod(grid)
            total = cost(
                self.proj,
                input_grid=input_grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            total += cost(
                self.block,
                input_grid=grid,
                batch_size=batch_size,
                dtype=dtype,
            ).tile(
                self.num_layers,
                copies=self.num_layers,
            )
            total += cost(
                self.activation,
                channels=rows * self.channels_out,
                dtype=dtype,
            ).tile(self.num_layers + 2)
            if self.dropout is not None:
                total += cost(
                    self.dropout,
                    seq_len=math.prod(grid),
                    batch_size=batch_size,
                    dtype=dtype,
                )
            return total + elementwise_cost(
                primal=rows * self.channels_out,
                adjoint=rows * self.channels_out,
                channels=self.channels_out,
                rows=rows,
                inputs=2,
                dtype=dtype,
            )

        channels_in: int = -1
        """Incoming spatial-map width."""

        channels_out: int = -1
        """Downsampled map width."""

        _: KW_ONLY

        num_layers: int = 1
        """Number of spatial refinement convolutions."""

        proj: ConvNorm.Config = field(
            default_factory=lambda: ConvNorm.Config(
                conv=Conv2d.Config(kernel_size=2, stride=2, padding=0, bias=True),
            ),
        )
        """Normalized stride-two projection."""

        block: ConvNorm.Config = field(default_factory=ConvNorm.Config)
        """Repeated spatial refinement template."""

        dropout: Dropout2d.Config | None = None
        """Channel dropout after downsampling; None omits the operation."""

        activation: TensorFn = elu
        """Activation after projections and residual addition."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            self.proj.channels_in = self.channels_in
            self.proj.channels_out = self.channels_out
            self.block.channels_in = self.block.channels_out = self.channels_out
            if self.dropout is not None:
                self.dropout.channels_in = self.channels_out
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.proj = config.proj.make()
        self.blocks = nn.ModuleList(
            config.block.make() for _ in range(config.num_layers)
        )
        self.dropout: Dropout2d | None = (
            None if config.dropout is None else config.dropout.make()
        )
        self.activation = config.activation

    def reset_parameters(self) -> None:
        """Reset the downsampling projection and refinement blocks."""
        self.proj.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()

    @override
    def forward(self, x: Tensor) -> Tensor:
        residual = self.activation(self.proj(x))
        x = residual if self.dropout is None else self.dropout(residual)
        for block in self.blocks:
            x = self.activation(block(x))
        return self.activation(x + residual)


class VNetUp(nn.Module):
    """Upsample, concatenate a skip map, refine, and add the concatenated residual."""

    class Config(Fig["VNetUp"]):
        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count the complete decoder transition including its skip branch.

            Args:
                input_grid: Decoder input spatial extents, before upsampling.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete work and parameter ownership.

            """
            del kwargs
            grid = self.proj.output_grid(input_grid)
            rows = batch_size * math.prod(grid)
            total = cost(
                self.proj,
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
                channels=rows * self.channels_skip,
                dtype=dtype,
            )
            total += cost(
                self.block,
                input_grid=grid,
                batch_size=batch_size,
                dtype=dtype,
            ).tile(
                self.num_layers,
                copies=self.num_layers,
            )
            total += cost(
                self.activation,
                channels=rows * 2 * self.channels_skip,
                dtype=dtype,
            ).tile(self.num_layers + 1)
            if self.dropout is not None:
                total += cost(
                    self.dropout,
                    seq_len=math.prod(input_grid),
                    batch_size=batch_size,
                    dtype=dtype,
                )
            total += cost(
                self.dropout_skip,
                seq_len=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
            )
            return (
                total
                + elementwise_cost(
                    primal=2 * rows * self.channels_skip,
                    adjoint=2 * rows * self.channels_skip,
                    channels=2 * self.channels_skip,
                    rows=rows,
                    inputs=2,
                    dtype=dtype,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=4 * rows * self.channels_skip,
                    dtype=dtype,
                )
            )

        channels_in: int = -1
        """Incoming decoder map width."""

        channels_skip: int = -1
        """Width of the corresponding encoder skip map."""

        num_layers: int = 2
        """Number of spatial refinement convolutions after concatenation."""

        proj: ConvTranspose2d.Config = field(
            default_factory=lambda: ConvTranspose2d.Config(
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
        )
        """Transposed convolution for learned spatial upsampling."""

        norm: Makeable[TensorModule] = field(
            default_factory=lambda: BatchNorm2d.Config(
                elementwise_affine=True,
            ),
        )
        """Normalization after transposed convolution."""

        block: ConvNorm.Config = field(default_factory=ConvNorm.Config)
        """Repeated spatial refinement template."""

        dropout: Dropout2d.Config | None = None
        """Channel dropout before upsampling; None omits the operation."""

        dropout_skip: Dropout2d.Config = field(
            default_factory=lambda: Dropout2d.Config(p=0.5),
        )
        """Channel dropout applied to the skip map."""

        activation: TensorFn = elu
        """Activation after upsampling, refinement, and residual addition."""

        @override
        def finalize(self) -> Self:
            self.proj.channels_in = self.channels_in
            self.proj.channels_out = self.channels_skip
            propagate_attr(
                self.norm,
                name="channels_in",
                value=self.channels_skip,
                protocol=ChannelsIn,
            )
            self.block.channels_in = self.block.channels_out = 2 * self.channels_skip
            if self.dropout is not None:
                self.dropout.channels_in = self.channels_in
            self.dropout_skip.channels_in = self.channels_skip
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.proj = config.proj.make()
        self.norm = config.norm.make()
        self.blocks = nn.ModuleList(
            config.block.make() for _ in range(config.num_layers)
        )
        self.dropout: Dropout2d | None = (
            None if config.dropout is None else config.dropout.make()
        )
        self.dropout_skip: Dropout2d = config.dropout_skip.make()
        self.activation = config.activation

    def reset_parameters(self) -> None:
        """Reset the transposed convolution, normalization, and refinement blocks."""
        self.proj.reset_parameters()
        self.norm.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()

    @override
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        skip = self.dropout_skip(skip)
        x = self.activation(self.norm(self.proj(x)))
        residual = torch.cat((x, skip), dim=1)
        x = residual
        for block in self.blocks:
            x = self.activation(block(x))
        return self.activation(x + residual)


class VNetOutput(nn.Module):
    """Normalize a convolution at the output width, activate, then project."""

    class Config(Fig["VNetOutput"]):
        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count the output-width convolution, activation and final projection.

            Args:
                input_grid: Input spatial extents.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete work and parameter ownership.

            """
            del kwargs
            grid = self.conv.conv.output_grid(input_grid)
            return (
                cost(
                    self.conv,
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    self.activation,
                    channels=batch_size * math.prod(grid) * self.channels_out,
                    dtype=dtype,
                )
                + cost(self.proj, input_grid=grid, batch_size=batch_size, dtype=dtype)
            )

        channels_in: int = -1
        """Incoming decoder width."""

        channels_out: int = -1
        """Number of output logits."""

        _: KW_ONLY

        conv: ConvNorm.Config = field(default_factory=ConvNorm.Config)
        """Normalized output-width spatial convolution."""

        proj: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(
                kernel_size=1,
                padding=0,
                bias=True,
            ),
        )
        """Final pointwise projection."""

        activation: TensorFn = elu
        """Activation before the final projection."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            self.conv.channels_in = self.channels_in
            self.conv.channels_out = self.channels_out
            self.proj.channels_in = self.proj.channels_out = self.channels_out
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.conv = config.conv.make()
        self.proj = config.proj.make()
        self.activation = config.activation

    def reset_parameters(self) -> None:
        """Reset the output convolution, normalization, and projection."""
        self.conv.reset_parameters()
        self.proj.reset_parameters()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.activation(self.conv(x)))


class VNet(nn.Module):
    """Compose the full two-dimensional encoder and skip decoder."""

    class Config(Fig["VNet"]):
        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count all encoder and decoder stages without allocating model state.

            Args:
                input_grid: Input spatial extents.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete work and parameter ownership.

            """
            del kwargs
            grid = self.input.layer.conv.output_grid(input_grid)
            total = cost(
                self.input,
                input_grid=input_grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            for down in self.down:
                total += cost(down, input_grid=grid, batch_size=batch_size, dtype=dtype)
                grid = down.proj.conv.output_grid(grid)
            for up in self.up:
                total += cost(up, input_grid=grid, batch_size=batch_size, dtype=dtype)
                grid = up.proj.output_grid(grid)
            return total + cost(
                self.output,
                input_grid=grid,
                batch_size=batch_size,
                dtype=dtype,
            )

        channels_in: int = -1
        """Image channel count."""

        channels_out: int = -1
        """Output logit count."""

        _: KW_ONLY

        channels_hidden: tuple[int, ...] = (16, 32, 64, 128, 256)
        """Input residual width followed by four encoder widths."""

        input: VNetInput.Config = field(default_factory=VNetInput.Config)
        """Input projection and repeated-channel residual."""

        down: list[VNetDown.Config] = field(
            default_factory=lambda: [
                VNetDown.Config(num_layers=1),
                VNetDown.Config(num_layers=2),
                VNetDown.Config(num_layers=3, dropout=Dropout2d.Config(p=0.5)),
                VNetDown.Config(num_layers=2, dropout=Dropout2d.Config(p=0.5)),
            ],
        )
        """Four ordered downsampling stages and their refinement layers."""

        up: list[VNetUp.Config] = field(
            default_factory=lambda: [
                VNetUp.Config(num_layers=2, dropout=Dropout2d.Config(p=0.5)),
                VNetUp.Config(num_layers=2, dropout=Dropout2d.Config(p=0.5)),
                VNetUp.Config(num_layers=1),
                VNetUp.Config(num_layers=1),
            ],
        )
        """Four ordered skip-decoder stages and their refinement layers."""

        output: VNetOutput.Config = field(default_factory=VNetOutput.Config)
        """Output-width convolution and projection."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            self.input.channels_in = self.channels_in
            self.input.channels_out = self.channels_hidden[0]
            for stage, channels_in, channels_out in zip(
                self.down,
                self.channels_hidden,
                self.channels_hidden[1:],
                strict=False,
            ):
                stage.channels_in = channels_in
                stage.channels_out = channels_out
            channels = self.channels_hidden[-1]
            for stage, skip in zip(self.up, self.channels_hidden[-2::-1], strict=False):
                stage.channels_in = channels
                stage.channels_skip = skip
                channels = 2 * skip
            self.output.channels_in = channels
            self.output.channels_out = self.channels_out
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if (
            len(config.channels_hidden) != 5
            or len(config.down) != 4
            or len(config.up) != 4
        ):
            raise ValueError("VNet requires five widths and four down/up stages.")
        super().__init__()
        self.input = config.input.make()
        self.down = nn.ModuleList(stage.make() for stage in config.down)
        self.up = nn.ModuleList(stage.make() for stage in config.up)
        self.output = config.output.make()

    def reset_parameters(self) -> None:
        """Reset every convolution and normalization in traversal order."""
        self.input.reset_parameters()
        for stage in (*self.down, *self.up):
            stage.reset_parameters()
        self.output.reset_parameters()

    @override
    def forward(self, x: Tensor) -> Tensor:
        skips = [self.input(x)]
        for stage in self.down:
            skips.append(stage(skips[-1]))
        x = skips.pop()
        for stage, skip in zip(self.up, reversed(skips), strict=True):
            x = stage(x, skip=skip)
        return self.output(x)
