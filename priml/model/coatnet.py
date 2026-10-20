"""CoAtNet: convolutional and relative-bias attention stages with a pooled output."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import TYPE_CHECKING, Self, override

import math

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    map_cost,
    nbytes,
    reduction_cost,
    set_cost,
    traffic,
)
from priml.math.custom_types import TensorFn
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.conv import Conv2d
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    HasOutputGrid,
    HasResetParameters,
    TensorModule,
    propagate_attr,
    validate_same_width,
)
from priml.model.linear import Linear
from priml.model.norm import BatchNorm2d, LayerNorm2d
from priml.model.pool import AvgPool2d, avg_pool_cost
from priml.model.sequential import Sequential
from priml.model.special import Dropout, Identity
from priml.model.squeeze_excitation import SqueezeExcitation
from priml.model.state_dict import validate_state_dict
from priml.model.swiglu import gelu


if TYPE_CHECKING:
    from collections.abc import Mapping


@set_cost(map_cost(primal=5, adjoint=5))
def silu_inplace(x: Tensor) -> Tensor:
    """Apply SiLU in place, with its elementwise cost."""
    return functional.silu(x, inplace=True)


class CoAtModule(nn.Module):
    """Reset composed Priml modules through their declared reset contracts."""

    def reset_parameters(self) -> None:
        """Reset child parameters and running statistics in place."""
        for child in self.children():
            if isinstance(child, HasResetParameters):
                child.reset_parameters()


class CoAtNorm(BatchNorm2d):
    """Batch normalization followed by the selected inplace SiLU activation."""

    class Config(Makes["CoAtNorm"], BatchNorm2d.Config):
        elementwise_affine: bool = True
        """Retain learned batch-normalization scale and offset."""

        activation: TensorFn = silu_inplace
        """Activation applied after normalization."""

        @override
        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count batch normalization followed by its injected activation.

            Args:
                seq_len: Positions entering each projection.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Remaining outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            return BatchNorm2d.Config.cost(
                self,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + cost(
                self.activation,
                channels=seq_len * batch_size * self.channels_in,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.activation = config.activation

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        return self.activation(super().forward(input, **kwargs))


class CoAtPool(CoAtModule):
    """Two-by-two average pooling followed by an optional channel projection."""

    class Config(Fig["CoAtPool"]):
        channels_in: int = -1
        """Width before spatial pooling."""

        channels_out: int = -1
        """Width after the optional projection."""

        _: KW_ONLY

        pool: AvgPool2d.Config = field(
            default_factory=lambda: AvgPool2d.Config(kernel_size=2, stride=2),
        )
        """Injected native average pooling before the shortcut projection."""

        projection: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1, padding=0, bias=True),
        )
        """One-by-one projection used when the width changes."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            self.pool.channels_in = self.pool.channels_out = self.channels_in
            self.projection.channels_in = self.channels_in
            self.projection.channels_out = self.channels_out
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count each two-by-two average and the optional width projection.

            Args:
                input_grid: Spatial extents entering this component.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Remaining outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            grid = self.pool.output_grid(input_grid)
            total = cost(
                self.pool,
                input_grid=input_grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            if self.channels_in != self.channels_out:
                total += cost(
                    self.projection,
                    input_grid=grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
            return total

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.pool = config.pool.make()
        self.expand = (
            config.projection.make()
            if config.channels_in != config.channels_out
            else nn.Identity()
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.expand(self.pool(x))


class CoAtMBConv(CoAtModule):
    """Prenormalized expansion, depthwise convolution, SE, and residual projection."""

    class Config(Fig["CoAtMBConv"]):
        channels_in: int = -1
        """Input feature width."""

        channels_out: int = -1
        """Output feature width."""

        _: KW_ONLY

        stride: int = 1
        """Spatial stride in the depthwise convolution."""

        expansion: float = 4.0
        """Expansion factor applied to the input width."""

        pre_norm: CoAtNorm.Config = field(default_factory=CoAtNorm.Config)
        """Input normalization and activation."""

        conv1_1x1: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1, padding=0),
        )
        """Expansion convolution."""

        norm1: CoAtNorm.Config = field(default_factory=CoAtNorm.Config)
        """Expanded-feature normalization and activation."""

        conv2_kxk: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(padding=1),
        )
        """Depthwise three-by-three convolution."""

        norm2: CoAtNorm.Config = field(default_factory=CoAtNorm.Config)
        """Depthwise-feature normalization and activation."""

        se: SqueezeExcitation.Config = field(
            default_factory=lambda: SqueezeExcitation.Config(
                act=silu_inplace,
                fc1=Conv2d.Config(kernel_size=1, padding=0, bias=True),
                fc2=Conv2d.Config(kernel_size=1, padding=0, bias=True),
            ),
        )
        """Squeeze-and-excitation gate after the second normalization."""

        conv3_1x1: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1, padding=0),
        )
        """Final linear channel projection."""

        shortcut: CoAtPool.Config = field(
            default_factory=lambda: CoAtPool.Config(
                projection=Conv2d.Config(kernel_size=1, padding=0),
            ),
        )
        """Average-pool shortcut with an unbiased width projection."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            expanded = self.channels_in * self.expansion
            middle = max(8, int(expanded + 4) // 8 * 8)
            if middle < 0.9 * expanded:
                middle += 8
            self.pre_norm.channels_in = self.channels_in
            self.conv1_1x1.channels_in = self.channels_in
            self.conv1_1x1.channels_out = middle
            self.norm1.channels_in = self.norm2.channels_in = middle
            self.conv2_kxk.channels_in = self.conv2_kxk.channels_out = middle
            self.conv2_kxk.groups = middle
            self.conv2_kxk.stride = self.stride
            self.se.channels_in = self.se.channels_out = middle
            if self.se.channels_hidden == -1:
                self.se.channels_hidden = int(middle * 0.25)
            self.conv3_1x1.channels_in = middle
            self.conv3_1x1.channels_out = self.channels_out
            self.shortcut.channels_in = self.channels_in
            self.shortcut.channels_out = self.channels_out
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count expansion, depthwise processing, squeeze-excitation and shortcut.

            Args:
                input_grid: Spatial extents entering this component.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Remaining outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            total = cost(
                self.pre_norm,
                seq_len=math.prod(input_grid),
                batch_size=batch_size,
                dtype=dtype,
            )
            grid = input_grid
            for projection, norm in (
                (self.conv1_1x1, self.norm1),
                (self.conv2_kxk, self.norm2),
            ):
                total += cost(
                    projection,
                    input_grid=grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                grid = _convolution_grid(grid, config=projection)
                total += cost(
                    norm,
                    seq_len=math.prod(grid),
                    batch_size=batch_size,
                    dtype=dtype,
                )
            total += cost(self.se, input_grid=grid, batch_size=batch_size, dtype=dtype)
            total += cost(
                self.conv3_1x1,
                input_grid=grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            if self.stride == 2:
                total += cost(
                    self.shortcut,
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
            rows = batch_size * math.prod(
                _convolution_grid(grid, config=self.conv3_1x1),
            )
            return total + elementwise_cost(
                primal=rows * self.channels_out,
                adjoint=rows * self.channels_out,
                channels=self.channels_out,
                rows=rows,
                inputs=2,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        if config.stride != 2:
            validate_same_width(config)
        super().__init__()
        self.shortcut = config.shortcut.make() if config.stride == 2 else nn.Identity()
        self.pre_norm = config.pre_norm.make()
        self.conv1_1x1 = config.conv1_1x1.make()
        self.norm1 = config.norm1.make()
        self.conv2_kxk = config.conv2_kxk.make()
        self.norm2 = config.norm2.make()
        self.se = config.se.make()
        self.conv3_1x1 = config.conv3_1x1.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut(x)
        hidden = self.conv1_1x1(self.pre_norm(x))
        hidden = self.norm1(hidden)
        hidden = self.norm2(self.conv2_kxk(hidden))
        return self.conv3_1x1(self.se(hidden)) + shortcut


class CoAtLinearMLP(CoAtModule):
    """The two linear projections used to generate log-coordinate relative bias."""

    class Config(Fig["CoAtLinearMLP"]):
        channels_hidden: int = 512
        """Hidden relative-position feature width."""

        channels_out: int = -1
        """Number of attention heads receiving relative bias."""

        fc1: Linear.Config = field(
            default_factory=lambda: Linear.Config(channels_in=2, bias=True),
        )
        """Projection from the two signed log coordinates."""

        fc2: Linear.Config = field(default_factory=lambda: Linear.Config(bias=True))
        """Projection from rectified hidden features to attention heads."""

        dropout: Dropout.Config = field(default_factory=lambda: Dropout.Config(p=0.125))
        """Dropout applied after rectification during training."""

        @override
        def finalize(self) -> Self:
            self.fc1.channels_out = self.fc2.channels_in = self.channels_hidden
            self.dropout.channels_in = self.channels_hidden
            self.fc2.channels_out = self.channels_out
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            input_grad: bool = True,
            **kwargs: object,
        ) -> Cost:
            """Count both projections and omit only an absent coordinate input gradient.

            Args:
                seq_len: Positions entering each projection.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                input_grad: Include the first projection's input gradient.
                **kwargs: Remaining outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            first = cost(
                self.fc1,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                input_grad=input_grad,
            )
            elements = batch_size * seq_len * self.channels_hidden
            total = first + cost(
                self.fc2,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
            )
            total += elementwise_cost(
                primal=elements,
                adjoint=elements,
                channels=elements,
                dtype=dtype,
            )
            total += cost(
                self.dropout,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
            )
            return total

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc1 = config.fc1.make()
        self.fc2 = config.fc2.make()
        self.drop1 = config.dropout.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.drop1(functional.relu(self.fc1(x))))


class CoAtRelativeBias(CoAtModule):
    """Signed log-coordinate MLP bias with ordered two-dimensional indexing."""

    class Config(Fig["CoAtRelativeBias"]):
        feature_size: tuple[int, int] = (20, 20)
        """Spatial attention grid at this stage."""

        num_heads: int = -1
        """Number of attention heads."""

        mlp: CoAtLinearMLP.Config = field(default_factory=CoAtLinearMLP.Config)
        """Injectable relative-position projection composition."""

        @override
        def finalize(self) -> Self:
            self.mlp.channels_out = self.num_heads
            return super().finalize()

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count one shared coordinate MLP table and its complete ordered gather.

            Args:
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Remaining outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs, batch_size
            height, width = self.feature_size
            positions = (2 * height - 1) * (2 * width - 1)
            area = height * width
            total = cost(
                self.mlp,
                seq_len=positions,
                batch_size=1,
                dtype=dtype,
                input_grad=False,
            )
            total += traffic(
                "primal",
                "selection",
                elements=(positions + area**2) * self.num_heads,
                dtype=dtype,
            )
            total += traffic(
                "adjoint",
                "selection",
                elements=(positions + area**2) * self.num_heads,
                flops=area**2 * self.num_heads,
                dtype=dtype,
            )
            return total + Cost(
                bytes_state=nbytes(area**2, torch.int64)
                + nbytes(2 * positions, torch.float32),
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        height, width = config.feature_size
        self.feature_size = config.feature_size
        self.num_heads = config.num_heads
        self.mlp = config.mlp.make()
        yy, xx = torch.meshgrid(
            torch.arange(height),
            torch.arange(width),
            indexing="ij",
        )
        coords = torch.stack([yy, xx]).flatten(1)
        relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0)
        relative[:, :, 0] += height - 1
        relative[:, :, 1] += width - 1
        relative[:, :, 0] *= 2 * width - 1
        self.relative_position_index = nn.Buffer(
            relative.sum(-1).contiguous().view(-1),
            persistent=False,
        )
        yy, xx = torch.meshgrid(
            torch.arange(-(height - 1), height).to(torch.float32),
            torch.arange(-(width - 1), width).to(torch.float32),
            indexing="ij",
        )
        coordinates = torch.stack([yy, xx]).permute(1, 2, 0).contiguous()
        self.rel_coords_log = nn.Buffer(
            torch.sign(coordinates) * torch.log(1.0 + coordinates.abs()),
            persistent=False,
        )

    def get_bias(self) -> Tensor:
        """Return a contiguous batch-broadcast attention bias.

        Returns:
            bias: Head-by-query-by-key bias with a broadcast batch dimension.

        """
        area = self.feature_size[0] * self.feature_size[1]
        table = self.mlp(self.rel_coords_log)
        selected = table.view(-1, self.num_heads)[self.relative_position_index]
        return (
            selected.view(area, area, self.num_heads)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
        )


class CoAtAttention(CoAtModule):
    """Head-first fused QKV attention on an NCHW feature map."""

    class Config(Fig["CoAtAttention"]):
        channels_in: int = -1
        """Query, key, and value projection input width."""

        channels_out: int = -1
        """Output projection width."""

        _: KW_ONLY

        channels_head: int = 32
        """Channels per attention head before the output expansion."""

        qkv: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1, padding=0, bias=True),
        )
        """Fused head-first query, key, and value projection."""

        relative: CoAtRelativeBias.Config = field(
            default_factory=CoAtRelativeBias.Config,
        )
        """Relative-position bias generator."""

        projection: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1, padding=0, bias=True),
        )
        """Projection that expands the attention width to the output width."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            self.qkv.channels_in = self.channels_in
            self.qkv.channels_out = 3 * self.channels_in
            self.relative.num_heads = self.channels_in // self.channels_head
            self.projection.channels_in = self.channels_in
            self.projection.channels_out = self.channels_out
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count fused projections, relative bias, and full dense two-product attention.

            Args:
                input_grid: Spatial extents entering this component.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Remaining outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            total = cost(
                self.qkv,
                input_grid=input_grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            grid = _convolution_grid(input_grid, config=self.qkv)
            total += cost(self.relative, batch_size=batch_size, dtype=dtype)
            total += attention_kernel_cost(
                seq_len=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
                num_heads=self.channels_in // self.channels_head,
                channels_head=self.channels_head,
            )
            bias_elements = self.relative.num_heads * math.prod(grid) ** 2
            total += traffic(
                "primal",
                kernel="elementwise",
                elements=(2 * batch_size + 1) * bias_elements,
                flops=batch_size * bias_elements,
                dtype=dtype,
            )
            total += reduction_cost(
                input_elements=batch_size * bias_elements,
                output_groups=bias_elements,
                phase="adjoint",
                dtype=dtype,
            )
            return total + cost(
                self.projection,
                input_grid=grid,
                batch_size=batch_size,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.channels_in % config.channels_head:
            raise ValueError("Attention width must divide into complete heads.")
        self.channels_head = config.channels_head
        self.num_heads = config.channels_in // config.channels_head
        self.qkv = config.qkv.make()
        self.rel_pos = config.relative.make()
        self.proj = config.projection.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        query, key, value = (
            self.qkv(x)
            .view(batch, self.num_heads, self.channels_head * 3, -1)
            .chunk(3, dim=2)
        )
        attended = functional.scaled_dot_product_attention(
            query.transpose(-1, -2).contiguous(),
            key.transpose(-1, -2).contiguous(),
            value.transpose(-1, -2).contiguous(),
            attn_mask=self.rel_pos.get_bias(),
            dropout_p=0.0,
        )
        attended = attended.transpose(-1, -2).reshape(batch, -1, height, width)
        return self.proj(attended)


class CoAtScale(nn.Module):
    """Per-channel LayerScale in the NCHW layout."""

    class Config(Fig["CoAtScale"]):
        channels_in: int = -1
        """Number of scaled channels."""

        initial: float = 1e-6
        """Initial channel multiplier."""

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count learned channel multiplication and its reduced scale gradient.

            Args:
                input_grid: Spatial extents entering this component.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Remaining outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            rows = batch_size * math.prod(input_grid)
            return elementwise_cost(
                primal=rows * self.channels_in,
                adjoint=2 * rows * self.channels_in,
                channels=self.channels_in,
                params=self.channels_in,
                rows=rows,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.initial = config.initial
        self.gamma = nn.Parameter(torch.full((config.channels_in,), config.initial))

    def reset_parameters(self) -> None:
        """Restore all learned channel multipliers."""
        nn.init.constant_(self.gamma, self.initial)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma.view(1, -1, 1, 1)


class CoAtConvMLP(CoAtModule):
    """Spatially shared one-by-one convolutional GELU feedforward."""

    class Config(Fig["CoAtConvMLP"]):
        channels_in: int = -1
        """Input and output feature width."""

        expansion: float = 4.0
        """Hidden width multiplier."""

        fc1: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1, padding=0, bias=True),
        )
        """Hidden projection."""

        fc2: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1, padding=0, bias=True),
        )
        """Residual-width output projection."""

        @override
        def finalize(self) -> Self:
            self.fc1.channels_in = self.fc2.channels_out = self.channels_in
            self.fc1.channels_out = self.fc2.channels_in = int(
                self.channels_in * self.expansion,
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
            """Count both spatial projections and their exact GELU activation.

            Args:
                input_grid: Spatial extents entering this component.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Remaining outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            grid = _convolution_grid(input_grid, config=self.fc1)
            elements = batch_size * math.prod(grid) * self.fc1.channels_out
            return (
                cost(
                    self.fc1,
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=5 * elements,
                    adjoint=11 * elements,
                    channels=elements,
                    dtype=dtype,
                )
                + cost(self.fc2, input_grid=grid, batch_size=batch_size, dtype=dtype)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc1 = config.fc1.make()
        self.fc2 = config.fc2.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(gelu(self.fc1(x)))


class CoAtDownNorm(CoAtModule):
    """Normalize channels before average-pooling the transition attention branch."""

    class Config(Fig["CoAtDownNorm"]):
        channels_in: int = -1
        """Width entering the normalization; -1 keeps the children's own widths."""

        norm: LayerNorm2d.Config = field(default_factory=LayerNorm2d.Config)
        """Channel normalization before downsampling."""

        down: CoAtPool.Config = field(default_factory=CoAtPool.Config)
        """Width-preserving two-by-two pooling."""

        @override
        def finalize(self) -> Self:
            if self.channels_in != -1:
                self.norm.channels_in = self.channels_in
                self.down.channels_in = self.down.channels_out = self.channels_in
            return super().finalize()

        def output_grid(self, input_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Return the pooled spatial extents."""
            return self.down.pool.output_grid(input_grid)

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count channel normalization before the configured transition pooling.

            Args:
                input_grid: Spatial extents entering this component.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Remaining outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            return cost(
                self.norm,
                seq_len=math.prod(input_grid),
                batch_size=batch_size,
                dtype=dtype,
            ) + cost(
                self.down,
                input_grid=input_grid,
                batch_size=batch_size,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm = config.norm.make()
        self.down = config.down.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.norm(x))


class CoAtTransformer(CoAtModule):
    """Relative-bias attention and convolutional MLP with two scaled residuals."""

    class Config(Fig["CoAtTransformer"]):
        channels_in: int = -1
        """Input feature width."""

        channels_out: int = -1
        """Output feature width."""

        _: KW_ONLY

        norm1: Makeable[TensorModule] = field(default_factory=LayerNorm2d.Config)
        """Normalization before attention, with the downsampling of a transition."""

        shortcut: Makeable[TensorModule] = field(default_factory=Identity.Config)
        """Residual path; a transition pools and projects it to the output width."""

        attention: CoAtAttention.Config = field(default_factory=CoAtAttention.Config)
        """Relative-position attention branch."""

        ls1: CoAtScale.Config = field(default_factory=CoAtScale.Config)
        """Attention residual scale."""

        norm2: LayerNorm2d.Config = field(default_factory=LayerNorm2d.Config)
        """Normalization before the feedforward branch."""

        mlp: CoAtConvMLP.Config = field(default_factory=CoAtConvMLP.Config)
        """Convolutional feedforward branch."""

        ls2: CoAtScale.Config = field(default_factory=CoAtScale.Config)
        """Feedforward residual scale."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            propagate_attr(
                self.norm1,
                name="channels_in",
                value=self.channels_in,
                protocol=ChannelsIn,
            )
            self.norm2.channels_in = self.channels_out
            if not isinstance(self.shortcut, Identity.Config):
                propagate_attr(
                    self.shortcut,
                    name="channels_in",
                    value=self.channels_in,
                    protocol=ChannelsIn,
                )
                propagate_attr(
                    self.shortcut,
                    name="channels_out",
                    value=self.channels_out,
                    protocol=ChannelsOut,
                )
            self.attention.channels_in = self.channels_in
            self.attention.channels_out = self.channels_out
            self.mlp.channels_in = self.ls1.channels_in = self.ls2.channels_in = (
                self.channels_out
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
            """Count both scaled residual branches and optional transition pooling.

            Args:
                input_grid: Spatial extents entering this component.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Remaining outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            total = cost(
                self.norm1,
                input_grid=input_grid,
                seq_len=math.prod(input_grid),
                batch_size=batch_size,
                dtype=dtype,
            )
            total += cost(
                self.shortcut,
                input_grid=input_grid,
                seq_len=math.prod(input_grid),
                batch_size=batch_size,
                dtype=dtype,
            )
            grid = (
                self.norm1.output_grid(input_grid)
                if isinstance(self.norm1, HasOutputGrid)
                else input_grid
            )
            total += cost(
                self.attention,
                input_grid=grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            grid = _convolution_grid(grid, config=self.attention.projection)
            for child in (self.ls1, self.mlp, self.ls2):
                total += cost(
                    child,
                    input_grid=grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
            total += cost(
                self.norm2,
                seq_len=math.prod(grid),
                batch_size=batch_size,
                dtype=dtype,
            )
            rows = batch_size * math.prod(grid)
            return total + elementwise_cost(
                primal=2 * rows * self.channels_out,
                adjoint=2 * rows * self.channels_out,
                channels=self.channels_out,
                rows=rows,
                inputs=2,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        if isinstance(config.shortcut, Identity.Config):
            validate_same_width(config)
        super().__init__()
        self.shortcut = config.shortcut.make()
        self.norm1 = config.norm1.make()
        self.attn = config.attention.make()
        self.ls1 = config.ls1.make()
        self.norm2 = config.norm2.make()
        self.mlp = config.mlp.make()
        self.ls2 = config.ls2.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        hidden = self.shortcut(x) + self.ls1(self.attn(self.norm1(x)))
        return hidden + self.ls2(self.mlp(self.norm2(hidden)))


class CoAtStem(CoAtModule):
    """Two convolutions with a single intermediate normalization and SiLU."""

    class Config(Fig["CoAtStem"]):
        channels_in: int = 3
        """Number of image channels."""

        channels_hidden: tuple[int, int] = (64, 128)
        """Intermediate and output stem widths."""

        conv1: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(stride=2, padding=1),
        )
        """Stride-two input convolution."""

        norm1: CoAtNorm.Config = field(default_factory=CoAtNorm.Config)
        """Intermediate normalization and SiLU."""

        conv2: Conv2d.Config = field(default_factory=lambda: Conv2d.Config(padding=1))
        """Stride-one output convolution."""

        @override
        def finalize(self) -> Self:
            self.conv1.channels_in = self.channels_in
            self.conv1.channels_out = self.conv2.channels_in = (
                self.norm1.channels_in
            ) = self.channels_hidden[0]
            self.conv2.channels_out = self.channels_hidden[1]
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count the two spatial stem projections and intermediate activated norm.

            Args:
                input_grid: Spatial extents entering this component.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Remaining outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            grid = _convolution_grid(input_grid, config=self.conv1)
            return (
                cost(
                    self.conv1,
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    self.norm1,
                    seq_len=math.prod(grid),
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(self.conv2, input_grid=grid, batch_size=batch_size, dtype=dtype)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.conv1 = config.conv1.make()
        self.norm1 = config.norm1.make()
        self.conv2 = config.conv2.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.norm1(self.conv1(x)))


class CoAtStage(CoAtModule):
    """A transition block followed by independently owned blocks of one stage."""

    class Config(Fig["CoAtStage"]):
        channels_in: int = -1
        """Width entering the stage."""

        channels_out: int = -1
        """Width of every block's output."""

        _: KW_ONLY

        feature_size: tuple[int, int] = (20, 20)
        """Spatial output grid of the stage."""

        blocks_per_stage: int = 2
        """Blocks of the stage, the transition included, when `block` is a template."""

        transition: Makeable[TensorModule] = field(
            default_factory=lambda: CoAtMBConv.Config(stride=2),
        )
        """First block, which halves the grid and sets the stage width."""

        block: Makeable[TensorModule] | list[Makeable[TensorModule]] = field(
            default_factory=CoAtMBConv.Config,
        )
        """Block template repeated after the transition, or explicit ordered blocks."""

        @override
        def finalize(self) -> Self:
            if self.blocks_per_stage < 1 and not isinstance(self.block, list):
                raise ValueError("A CoAtNet stage requires its transition block.")
            blocks = (
                self.block
                if isinstance(self.block, list)
                else [self.block.copy_tree() for _ in range(self.blocks_per_stage - 1)]
            )
            for index, block in enumerate([self.transition, *blocks]):
                propagate_attr(
                    block,
                    name="channels_in",
                    value=self.channels_in if index == 0 else self.channels_out,
                    protocol=ChannelsIn,
                )
                propagate_attr(
                    block,
                    name="channels_out",
                    value=self.channels_out,
                    protocol=ChannelsOut,
                )
                if isinstance(block, CoAtTransformer.Config):
                    block.attention.relative.feature_size = self.feature_size
            self.block = blocks
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count the transition and every block with independent ownership.

            Args:
                input_grid: Spatial extents entering this component.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Remaining outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            assert isinstance(self.block, list)
            total = cost(
                self.transition,
                input_grid=input_grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            for block in self.block:
                total += cost(
                    block,
                    input_grid=self.feature_size,
                    batch_size=batch_size,
                    dtype=dtype,
                )
            return total

    def __init__(self, config: Config) -> None:
        assert isinstance(config.block, list)
        super().__init__()
        self.blocks = Sequential.Config(
            elements=[config.transition, *config.block],
        ).make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class CoAtNet(CoAtModule):
    """C/C/T/T CoAtNet with log-coordinate MLP relative bias, ending in global pooling."""

    class Config(Fig["CoAtNet"]):
        image_size: tuple[int, int] = (224, 224)
        """Input field the relative-bias grids are sized for."""

        stem: CoAtStem.Config = field(default_factory=CoAtStem.Config)
        """Two-convolution image stem."""

        channels_hidden: tuple[int, ...] = (128, 256, 512, 1024)
        """Feature widths of the C/C/T/T stages."""

        blocks_per_stage: tuple[int, ...] = (2, 6, 14, 2)
        """Block counts of every stage whose `block` is a template."""

        stage: CoAtStage.Config | list[CoAtStage.Config] = field(
            default_factory=lambda: [
                CoAtStage.Config(),
                CoAtStage.Config(),
                CoAtStage.Config(
                    transition=CoAtTransformer.Config(
                        norm1=CoAtDownNorm.Config(),
                        shortcut=CoAtPool.Config(),
                    ),
                    block=CoAtTransformer.Config(),
                ),
                CoAtStage.Config(
                    transition=CoAtTransformer.Config(
                        norm1=CoAtDownNorm.Config(),
                        shortcut=CoAtPool.Config(),
                    ),
                    block=CoAtTransformer.Config(),
                ),
            ],
        )
        """Stage template, or explicit stages that own their blocks.

        The default is two convolutional stages followed by two attention stages.
        """

        norm: LayerNorm2d.Config = field(default_factory=LayerNorm2d.Config)
        """Final channel normalization before global average pooling."""

        @override
        def finalize(self) -> Self:
            self.norm.channels_in = self.channels_hidden[-1]
            stages = (
                self.stage
                if isinstance(self.stage, list)
                else [self.stage.copy_tree() for _ in self.channels_hidden]
            )
            width = self.stem.channels_hidden[-1]
            height, columns = _convolution_grid(
                _convolution_grid(self.image_size, config=self.stem.conv1),
                config=self.stem.conv2,
            )
            for index, (stage, channels) in enumerate(
                zip(stages, self.channels_hidden, strict=True),
            ):
                height, columns = (height - 1) // 2 + 1, (columns - 1) // 2 + 1
                stage.channels_in, stage.channels_out = width, channels
                stage.feature_size = (height, columns)
                if not isinstance(stage.block, list):
                    stage.blocks_per_stage = self.blocks_per_stage[index]
                width = channels
            self.stage = stages
            return super().finalize()

        def feature_width(self) -> int:
            """Return the width of each pooled feature vector."""
            return self.channels_hidden[-1]

        def cost(
            self,
            *,
            input_grid: tuple[int, ...],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count the complete finalized backbone, final norm and global pooling.

            Args:
                input_grid: Spatial extents entering this component.
                batch_size: Images in this invocation.
                dtype: Operand dtype; None uses Torch's default.
                **kwargs: Remaining outer cost arguments.

            Returns:
                execution: Complete forward/backward work and parameter ownership.

            """
            del kwargs
            total = cost(
                self.stem,
                input_grid=input_grid,
                batch_size=batch_size,
                dtype=dtype,
            )
            grid = _convolution_grid(
                _convolution_grid(input_grid, config=self.stem.conv1),
                config=self.stem.conv2,
            )
            assert isinstance(self.stage, list)
            for stage in self.stage:
                total += cost(
                    stage,
                    input_grid=grid,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                grid = stage.feature_size
            return (
                total
                + cost(
                    self.norm,
                    seq_len=math.prod(grid),
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + avg_pool_cost(
                    channels=self.channels_hidden[-1],
                    positions=math.prod(grid),
                    batch_size=batch_size,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        assert isinstance(config.stage, list)
        _validate_geometry(config)
        super().__init__()
        self.stem = config.stem.make()
        elements: list[Makeable[TensorModule]] = list(config.stage)
        self.stages = Sequential.Config(elements=elements).make()
        self.norm = config.norm.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        features = self.norm(self.stages(self.stem(x)))
        return functional.adaptive_avg_pool2d(features, 1).flatten(1)

    def load_checkpoint(self, state: Mapping[str, Tensor]) -> None:
        """Validate every checkpoint tensor before mutating any model state.

        Args:
            state: Every checkpoint parameter and persistent running statistic.

        """
        validate_state_dict(state, expected=self.state_dict())
        self.load_state_dict(state)


def _validate_geometry(config: CoAtNet.Config) -> None:
    assert isinstance(config.stage, list)
    grid = _convolution_grid(config.image_size, config=config.stem.conv1)
    grid = _convolution_grid(grid, config=config.stem.conv2)
    for stage in config.stage:
        assert isinstance(stage.block, list)
        for block in [stage.transition, *stage.block]:
            if (isinstance(block, CoAtMBConv.Config) and block.stride != 2) or (
                isinstance(block, CoAtTransformer.Config)
                and isinstance(block.shortcut, Identity.Config)
            ):
                validate_same_width(block)
            if isinstance(block, CoAtMBConv.Config):
                shortcut = (
                    _pool_grid(grid, config=block.shortcut)
                    if block.stride == 2
                    else grid
                )
                for conv in (block.conv1_1x1, block.conv2_kxk, block.conv3_1x1):
                    grid = _convolution_grid(grid, config=conv)
                if grid != shortcut:
                    raise ValueError("CoAtNet MBConv residual grids differ.")
            elif isinstance(block, CoAtTransformer.Config):
                grid = _transformer_grid(grid, config=block)


def _convolution_grid(
    grid: tuple[int, ...],
    *,
    config: Conv2d.Config,
) -> tuple[int, ...]:
    output = config.output_grid(grid)
    if min(output) <= 0:
        raise ValueError("CoAtNet convolution grid must remain positive.")
    return output


def _pool_grid(grid: tuple[int, ...], *, config: CoAtPool.Config) -> tuple[int, ...]:
    pooled = config.pool.output_grid(grid)
    if min(pooled) <= 0:
        raise ValueError("CoAtNet pooling grid must remain positive.")
    return (
        _convolution_grid(pooled, config=config.projection)
        if config.channels_in != config.channels_out
        else pooled
    )


def _transformer_grid(
    grid: tuple[int, ...],
    *,
    config: CoAtTransformer.Config,
) -> tuple[int, ...]:
    shortcut = (
        _pool_grid(grid, config=config.shortcut)
        if isinstance(config.shortcut, CoAtPool.Config)
        else grid
    )
    attended = (
        config.norm1.output_grid(grid)
        if isinstance(config.norm1, HasOutputGrid)
        else grid
    )
    if (
        attended != config.attention.relative.feature_size
        or _convolution_grid(attended, config=config.attention.qkv) != attended
    ):
        raise ValueError("CoAtNet attention grid differs from its relative bias grid.")
    output = _convolution_grid(attended, config=config.attention.projection)
    hidden = _convolution_grid(output, config=config.mlp.fc1)
    if output != shortcut or _convolution_grid(hidden, config=config.mlp.fc2) != output:
        raise ValueError("CoAtNet attention or MLP residual grids differ.")
    return output
