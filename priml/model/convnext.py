"""Composable ConvNeXt with channel-last MLPs and raw spatial features."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import TYPE_CHECKING, Self, override

import math

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, cost, elementwise_cost, map_cost, set_cost
from priml.model.conv import Conv2d, conv_cost
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    HasOutputGrid,
    TensorModule,
    propagate_attr,
    validate_same_width,
)
from priml.model.init import kaiming_uniform
from priml.model.norm import LayerNorm
from priml.model.pool import avg_pool_cost
from priml.model.sequential import Sequential
from priml.model.special import Identity
from priml.model.swiglu import SwiGLU


if TYPE_CHECKING:
    from collections.abc import Mapping


@set_cost(map_cost(primal=5, adjoint=11))
def gelu(x: Tensor) -> Tensor:
    """Apply exact GELU with its complete elementwise cost.

    Args:
      x: Pre-activation values, any shape.

    Returns:
      activated: ``x * (1 + erf(x / sqrt(2))) / 2``, elementwise.

    """
    return functional.gelu(x)


class LayerNorm2d(LayerNorm):
    """Normalize the channel axis of an NCHW tensor."""

    class Config(Makes["LayerNorm2d"], LayerNorm.Config):
        eps: float = 1e-6
        """Epsilon added to the channel variance."""

        elementwise_affine: bool = True
        """Learn a channel scale and bias."""

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        return super().forward(input.permute(0, 2, 3, 1), **kwargs).permute(0, 3, 1, 2)


class ConvNeXtStem(nn.Module):
    """Patch convolution followed by an injectable channel normalization."""

    class Config(Fig["ConvNeXtStem"]):
        channels_in: int = -1
        """Image channel count."""

        channels_out: int = -1
        """Patch feature width."""

        _: KW_ONLY

        conv: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
            ),
        )
        """Patch projection and its initialization."""

        norm: Makeable[TensorModule] = field(default_factory=LayerNorm2d.Config)
        """Normalization after patch projection."""

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

        def output_grid(self, input_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Propagate the patch convolution's spatial extents.

            Args:
                input_grid: Spatial extents entering the stem.

            Returns:
                output: Spatial extents of the normalized patch map.

            """
            return self.conv.output_grid(input_grid)

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            input_grad: bool = True,
            **kwargs: object,
        ) -> Cost:
            """Count patch projection and normalization at its actual output grid.

            Args:
                input_grid: Spatial extents before patch projection.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                input_grad: Include a gradient for the source image.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            return conv_cost(
                channels_in=self.conv.channels_in,
                channels_out=self.conv.channels_out,
                kernel_size=self.conv.kernel_size,
                ndim=2,
                groups=self.conv.groups,
                bias=self.conv.bias,
                input_grid=input_grid,
                batch_size=batch_size,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                dtype=self.conv.dtype if self.conv.dtype is not None else dtype,
                input_grad=input_grad,
            ) + cost(
                self.norm,
                seq_len=math.prod(self.output_grid(input_grid)),
                batch_size=batch_size,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.conv = config.conv.make()
        self.norm = config.norm.make()

    def reset_parameters(self) -> None:
        """Reset the patch projection and channel normalization."""
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        return self.norm(self.conv(x), **kwargs)


class ConvNeXtBlock(nn.Module):
    """Depthwise convolution, channel-last feedforward, and scaled residual."""

    class Config(Fig["ConvNeXtBlock"]):
        channels_in: int = -1
        """Residual channel width."""

        channels_out: int = -1
        """Residual output width, equal to the input width."""

        _: KW_ONLY

        conv_dw: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(
                kernel_size=7,
                padding=3,
                bias=True,
            ),
        )
        """Depthwise spatial convolution and its initialization."""

        norm: Makeable[TensorModule] = field(
            default_factory=lambda: LayerNorm.Config(
                eps=1e-6,
                elementwise_affine=True,
            ),
        )
        """Normalization of the channel-last convolution output."""

        mlp: Makeable[TensorModule] = field(
            default_factory=lambda: SwiGLU.Config(
                gate=False,
                bias=True,
                round_to=1,
                act=gelu,
                init_weight=kaiming_uniform,
                init_weight_out=kaiming_uniform,
            ),
        )
        """Channel-last feedforward projection."""

        layer_scale: float | None = 1e-6
        """Initial per-channel residual scale; None removes the parameter."""

        drop_path: float = 0.0
        """Probability of dropping a complete per-example residual branch."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            self.conv_dw.channels_in = self.channels_in
            self.conv_dw.channels_out = self.channels_in
            self.conv_dw.groups = self.channels_in
            for child in (self.norm, self.mlp):
                propagate_attr(
                    child,
                    name="channels_in",
                    value=self.channels_in,
                    protocol=ChannelsIn,
                )
                propagate_attr(
                    child,
                    name="channels_out",
                    value=self.channels_in,
                    protocol=ChannelsOut,
                )
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count both residual paths, including scale and stochastic depth.

            Args:
                input_grid: Spatial extents of the residual feature map.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            if self.conv_dw.output_grid(input_grid) != input_grid:
                raise ValueError(
                    "ConvNeXt residual convolution must preserve its grid.",
                )
            positions = math.prod(input_grid)
            rows = positions * batch_size
            width = self.channels_in
            total = cost(
                self.conv_dw,
                input_grid=input_grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            for child in (self.norm, self.mlp):
                total += cost(
                    child,
                    seq_len=positions,
                    batch_size=batch_size,
                    dtype=dtype,
                )
            if self.layer_scale is not None:
                total += elementwise_cost(
                    primal=rows * width,
                    adjoint=2 * rows * width,
                    channels=width,
                    params=width,
                    rows=rows,
                    dtype=dtype,
                )
            if self.drop_path:
                total += elementwise_cost(
                    primal=rows * width + batch_size,
                    adjoint=rows * width,
                    channels=width,
                    inputs=2,
                    rows=rows,
                    dtype=dtype,
                )
            return total + elementwise_cost(
                primal=rows * width,
                adjoint=rows * width,
                channels=width,
                inputs=2,
                rows=rows,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        validate_same_width(config)
        if math.isnan(config.drop_path) or config.drop_path < 0 or config.drop_path > 1:
            raise ValueError("drop_path must be between zero and one.")
        super().__init__()
        self.conv_dw = config.conv_dw.make()
        self.norm = config.norm.make()
        self.mlp = config.mlp.make()
        self.gamma = (
            None
            if config.layer_scale is None
            else nn.Parameter(
                config.layer_scale * torch.ones(config.channels_in),
            )
        )
        self.drop_path = config.drop_path
        self._layer_scale = 0.0 if config.layer_scale is None else config.layer_scale

    def reset_parameters(self) -> None:
        """Reset the convolution, normalization, and feedforward modules."""
        self.conv_dw.reset_parameters()
        self.norm.reset_parameters()
        self.mlp.reset_parameters()
        if self.gamma is not None:
            nn.init.constant_(self.gamma, self._layer_scale)

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        shortcut = x
        x = self.conv_dw(x)
        if x.shape != shortcut.shape:
            raise ValueError("ConvNeXt residual convolution must preserve its grid.")
        x = self.norm(x.permute(0, 2, 3, 1), **kwargs)
        x = self.mlp(x, **kwargs).permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        if self.training and self.drop_path:
            keep = 1 - self.drop_path
            mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
            if keep > 0:
                mask.div_(keep)
            x = x * mask
        return x + shortcut


class ConvNeXtStage(nn.Module):
    """Optional norm-and-convolution downsampling followed by residual blocks."""

    class Config(Fig["ConvNeXtStage"]):
        channels_in: int = -1
        """Width entering the stage."""

        channels_out: int = -1
        """Width of every residual block in the stage."""

        _: KW_ONLY

        blocks_per_stage: int = 2
        """Number of residual blocks when block is a template."""

        stride: int = 2
        """Stride of the optional downsampling projection."""

        norm: Makeable[TensorModule] = field(default_factory=LayerNorm2d.Config)
        """Channel normalization before downsampling."""

        proj_in: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
        )
        """Downsampling convolution and its initialization."""

        downsample: Makeable[TensorModule] | None = None
        """Explicit downsampling composition; None derives it from widths and stride."""

        block: ConvNeXtBlock.Config | list[ConvNeXtBlock.Config] = field(
            default_factory=ConvNeXtBlock.Config,
        )
        """Residual block template or explicit ordered block configurations."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            if self.downsample is None:
                if self.channels_in == self.channels_out and self.stride == 1:
                    self.downsample = Identity.Config()
                else:
                    propagate_attr(
                        self.norm,
                        name="channels_in",
                        value=self.channels_in,
                        protocol=ChannelsIn,
                    )
                    self.proj_in.channels_in = self.channels_in
                    self.proj_in.channels_out = self.channels_out
                    self.proj_in.stride = self.stride
                    self.downsample = Sequential.Config(
                        elements=[self.norm, self.proj_in],
                    )
            propagate_attr(
                self.downsample,
                name="channels_in",
                value=self.channels_in,
                protocol=ChannelsIn,
            )
            propagate_attr(
                self.downsample,
                name="channels_out",
                value=self.channels_out,
                protocol=ChannelsOut,
            )
            blocks = (
                self.block
                if isinstance(self.block, list)
                else [self.block.copy_tree() for _ in range(self.blocks_per_stage)]
            )
            for block in blocks:
                block.channels_in = block.channels_out = self.channels_out
            self.block = blocks
            return super().finalize()

        def output_grid(self, input_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Return the spatial extents after the configured downsampler.

            Args:
                input_grid: Spatial extents entering the stage.

            Returns:
                output: Spatial extents preserved by the residual blocks.

            Raises:
                TypeError: The downsampler does not declare spatial geometry.
                ValueError: The downsampler removes spatial axes.

            """
            if not isinstance(self.downsample, HasOutputGrid):
                raise TypeError("ConvNeXt requires a spatially declared downsampler.")
            grid = self.downsample.output_grid(input_grid)
            if len(grid) != 2:
                raise ValueError("ConvNeXt downsamplers must retain two spatial axes.")
            return grid

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count the configured downsampler and every independently owned block.

            Args:
                input_grid: Spatial extents entering the stage.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            if self.downsample is None:
                raise ValueError("Expected self.downsample is not None.")
            assert isinstance(self.block, list)
            total = cost(
                self.downsample,
                input_grid=input_grid,
                seq_len=math.prod(input_grid),
                batch_size=batch_size,
                dtype=dtype,
            )
            grid = self.output_grid(input_grid)
            return total + sum(
                (
                    cost(block, input_grid=grid, batch_size=batch_size, dtype=dtype)
                    for block in self.block
                ),
                Cost(),
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.downsample is None:
            raise ValueError("Expected config.downsample is not None.")
        assert isinstance(config.block, list)
        self.downsample = config.downsample.make()
        self.blocks = nn.Sequential(*(block.make() for block in config.block))

    def reset_parameters(self) -> None:
        """Reset the downsampling composition and all residual blocks."""
        self.downsample.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        x = self.downsample(x, **kwargs)
        for block in self.blocks:
            x = block(x, **kwargs)
        return x


class ConvNeXtHead(nn.Module):
    """Average spatial positions before normalizing the pooled features."""

    class Config(Fig["ConvNeXtHead"]):
        channels_in: int = -1
        """Final spatial-map channel width."""

        norm: Makeable[TensorModule] = field(
            default_factory=lambda: LayerNorm.Config(
                eps=1e-6,
                elementwise_affine=True,
            ),
        )
        """Normalization applied after global average pooling."""

        proj_out: Makeable[TensorModule] = field(default_factory=Identity.Config)
        """Projection of the normalized feature vector."""

        @override
        def finalize(self) -> Self:
            for child in (self.norm, self.proj_out):
                propagate_attr(
                    child,
                    name="channels_in",
                    value=self.channels_in,
                    protocol=ChannelsIn,
                )
            return super().finalize()

        def output_grid(self, input_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Remove both spatial axes by global average pooling.

            Args:
                input_grid: Spatial extents entering the pooled head.

            Returns:
                output: An empty grid: one feature vector per example.

            """
            del input_grid
            return ()

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count spatial average pooling, pooled normalization and projection.

            Args:
                input_grid: Spatial extents before global pooling.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            return avg_pool_cost(
                channels=self.channels_in,
                positions=math.prod(input_grid),
                batch_size=batch_size,
                dtype=dtype,
            ) + sum(
                (
                    cost(child, seq_len=1, batch_size=batch_size, dtype=dtype)
                    for child in (self.norm, self.proj_out)
                ),
                Cost(),
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm = config.norm.make()
        self.proj_out = config.proj_out.make()

    def reset_parameters(self) -> None:
        """Reset the normalization and output projection."""
        self.norm.reset_parameters()
        self.proj_out.reset_parameters()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        return self.proj_out(self.norm(x.mean((-2, -1)), **kwargs), **kwargs)


class ConvNeXt(nn.Module):
    """Compose a patch stem, ordered spatial stages, and a pooled feature head."""

    class Config(Fig["ConvNeXt"]):
        channels_in: int = 3
        """Image channel count."""

        channels_hidden: tuple[int, ...] = (96, 192, 384, 768)
        """Spatial feature width in each stage."""

        blocks_per_stage: tuple[int, ...] = (3, 3, 27, 3)
        """Residual block count in each stage."""

        stem: ConvNeXtStem.Config = field(default_factory=ConvNeXtStem.Config)
        """Patch projection and normalization."""

        stage: ConvNeXtStage.Config | list[ConvNeXtStage.Config] = field(
            default_factory=ConvNeXtStage.Config,
        )
        """Stage template or explicit stage configurations."""

        head: Makeable[TensorModule] = field(default_factory=ConvNeXtHead.Config)
        """Postpool normalization and output projection."""

        drop_path: float = 0.0
        """Maximum residual dropout, linearly increasing across all blocks."""

        def block_counts(self) -> tuple[int, ...]:
            """Return realized residual-block counts from the configured stages.

            Returns:
                counts: Each explicit block-list length or stage-template count.

            """
            stages = (
                self.stage
                if isinstance(self.stage, list)
                else [self.stage] * len(self.channels_hidden)
            )
            return tuple(
                len(stage.block) if isinstance(stage.block, list) else count
                for stage, count in zip(stages, self.blocks_per_stage, strict=False)
            )

        def drop_path_rates(self) -> tuple[float, ...]:
            """Return the selected per-block stochastic-depth schedule.

            Returns:
                rates: CPU float32 linspace values in flat block order.

            """
            return tuple(
                float(rate)
                for rate in torch.linspace(
                    0,
                    self.drop_path,
                    sum(self.block_counts()),
                    device="cpu",
                    dtype=torch.float32,
                )
            )

        @override
        def finalize(self) -> Self:
            self.stem.channels_in = self.channels_in
            self.stem.channels_out = self.channels_hidden[0]
            stages = (
                self.stage
                if isinstance(self.stage, list)
                else [self.stage.copy_tree() for _ in self.channels_hidden]
            )
            expanded_blocks: list[list[ConvNeXtBlock.Config]] = []
            for index, (stage, channels, count) in enumerate(
                zip(
                    stages,
                    self.channels_hidden,
                    self.blocks_per_stage,
                    strict=False,
                ),
            ):
                stage.channels_in = self.channels_hidden[max(0, index - 1)]
                stage.channels_out = channels
                stage.stride = 1 if index == 0 else 2
                if not isinstance(stage.block, list):
                    stage.block = [stage.block.copy_tree() for _ in range(count)]
                expanded_blocks.append(stage.block)
            self.stage = stages
            rates = iter(self.drop_path_rates())
            for blocks in expanded_blocks:
                for block in blocks:
                    block.drop_path = next(rates)
            propagate_attr(
                self.head,
                name="channels_in",
                value=self.channels_hidden[-1],
                protocol=ChannelsIn,
            )
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            input_grad: bool = True,
            **kwargs: object,
        ) -> Cost:
            """Count the complete finalized encoder and its injected pooled head.

            Args:
                input_grid: Spatial extents of the input images.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                input_grad: Include a gradient for the source image.
                **kwargs: Unused outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            assert isinstance(self.stage, list)
            total = cost(
                self.stem,
                input_grid=input_grid,
                batch_size=batch_size,
                dtype=dtype,
                input_grad=input_grad,
            )
            grid = self.stem.output_grid(input_grid)
            for stage in self.stage:
                total += cost(
                    stage,
                    input_grid=grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                if stage.downsample is None:
                    raise ValueError("Expected stage.downsample is not None.")
                grid = stage.output_grid(grid)
            return total + cost(
                self.head,
                input_grid=grid,
                seq_len=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
            )

        def feature_grid(self, input_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Derive the raw feature-map grid from finalized spatial projections.

            Args:
                input_grid: Spatial extents entering the image stem.

            Returns:
                grid: Spatial extents before the pooled head.

            """
            assert isinstance(self.stage, list)
            grid = self.stem.output_grid(input_grid)
            for stage in self.stage:
                if stage.downsample is None:
                    raise ValueError("Expected stage.downsample is not None.")
                grid = stage.output_grid(grid)
            return grid

        def output_grid(self, input_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Return the actual spatial extents after the pooled head.

            Args:
                input_grid: Spatial extents entering the image stem.

            Returns:
                output: An empty grid for the resulting feature vectors.

            Raises:
                TypeError: The head does not declare spatial geometry.

            """
            if not isinstance(self.head, HasOutputGrid):
                raise TypeError("ConvNeXt requires a spatially declared head.")
            return self.head.output_grid(self.feature_grid(input_grid))

    def __init__(self, config: Config) -> None:
        assert isinstance(config.stage, list)
        stages = len(config.channels_hidden)
        if len(config.stage) != stages or len(config.blocks_per_stage) != stages:
            raise ValueError("ConvNeXt width and block stage counts must match.")
        super().__init__()
        self.stem = config.stem.make()
        self.stages = nn.ModuleList(stage.make() for stage in config.stage)
        self.head = config.head.make()
        self.num_features = config.channels_hidden[-1]

    def reset_parameters(self) -> None:
        """Reset the stem, ordered spatial stages, and pooled head."""
        self.stem.reset_parameters()
        for stage in self.stages:
            stage.reset_parameters()
        self.head.reset_parameters()

    def forward_features(self, x: Tensor, **kwargs: object) -> Tensor:
        """Return the raw NCHW spatial map before pooling and head normalization."""
        x = self.stem(x, **kwargs)
        for stage in self.stages:
            x = stage(x, **kwargs)
        return x

    def load_timm_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Load every tensor of a standard patch-stem ConvNeXt checkpoint.

        The mapping is bijective and validates names, shapes, and dtypes before
        loading any tensor. Classifier parameters require a matching head slot.

        Args:
          state: Complete checkpoint parameters and buffers.

        Raises:
          ValueError: A source key collides, or names, shapes, or dtypes differ.

        """
        mapped: dict[str, Tensor] = {}
        for name, value in state.items():
            target = (
                name.replace("stem.0.", "stem.conv.")
                .replace(
                    "stem.1.",
                    "stem.norm.",
                )
                .replace(".mlp.fc1.", ".mlp.up_proj.")
                .replace(
                    ".mlp.fc2.",
                    ".mlp.down_proj.",
                )
                .replace("head.fc.", "head.proj_out.")
            )
            if target in mapped:
                raise ValueError(f"Checkpoint keys collide at {target}.")
            mapped[target] = value
        expected = self.state_dict()
        if mapped.keys() != expected.keys():
            raise ValueError(
                "Checkpoint tensor names do not match ConvNeXt: "
                f"missing={sorted(expected.keys() - mapped.keys())}; "
                f"unexpected={sorted(mapped.keys() - expected.keys())}.",
            )
        for name, value in mapped.items():
            if (
                value.shape != expected[name].shape
                or value.dtype != expected[name].dtype
            ):
                raise ValueError(f"Checkpoint tensor shape or dtype differs: {name}.")
        self.load_state_dict(mapped, strict=True)

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        return self.head(self.forward_features(x, **kwargs), **kwargs)
