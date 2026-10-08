"""A convolutional first stage for the MinGRU policy, conditioned on the previous action.

:class:`BoardEncoder` fills :class:`~priml.baselines.craftax.model.MinGRUPolicy`'s
``embedding`` slot. It reads the packed observation with the previous action
id appended (844 floats) and emits the multi-hot embedding's own layout -- the
99 embedded cells, then the 51 status scalars -- so ``proj_in`` is unchanged:

- ``cells``: the multi-hot embedding of the packed row, 99 cells of 16.
- ``status``: :class:`ActionConditionedStatus`, two residual MLPs over the
  status scalars, each reading them beside an embedding of the previous action
  (44 ids: the 43 actions, and 43 itself for "none", at an episode's start).
- ``board``: :class:`BoardCNN` over the 9 x 11 view of embedded cells: two
  depthwise residual blocks, a 128-wide branch modulated by the status
  (:class:`FilmBranch`), then a mix of 3 x 3 and whole-board averages
  (:class:`MultiscaleMix`). The residual blocks are a slot: a
  :class:`ConvNextTrunk` in their place lifts the board to 64 channels, runs
  two ConvNeXt blocks (:class:`ConvNextBlock`) and projects it back.

Every residual branch's output projection starts at zero, so the stage begins
as the multi-hot embedding with the status passed through. Weights are torch's
default draws -- U(+-1/sqrt(fan_in)), a depthwise kernel's fan-in its taps and
a bias's its layer's -- a LayerNorm's ones and zeros, and N(0, 1) for the
action table, drawn in fp32; the policy rounds them to its dtype once.

Each module mirrors its reference's operation order, which fixes the order in
which autograd sums a tensor's gradients (a board feeds four ops in
:class:`MultiscaleMix`), and so the gradients' bits.

References:
    https://arxiv.org/abs/1709.07871
        Perez et al. FiLM: Visual Reasoning with a General Conditioning Layer.
    https://arxiv.org/abs/2201.03545
        Liu et al. A ConvNet for the 2020s.

"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from functools import partial
from typing import (
    Self,
    cast,
    override,
)

import math

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.craftax.lib.costs import (
    activation_cost,
    concat_cost,
    residual_cost,
)
from priml.baselines.craftax.model import packed_observation_embedding
from priml.cost import Cost, cost, elementwise_cost, reduction_cost
from priml.math.custom_types import TensorFn
from priml.model.conv import Conv2d
from priml.model.custom_types import ChannelsIn, propagate_attr
from priml.model.embedding import Embedding, MultiHotEmbedding
from priml.model.init import normal
from priml.model.linear import Linear
from priml.model.norm import LayerNorm


class MLP(nn.Module):
    """Two projections with an activation between: ``proj_out(activation(proj_in(x)))``.

    The policy's residual branches: the caller adds the output to a stream, and
    a zero-initialized ``proj_out`` starts the branch as a no-op.
    """

    class Config(Fig["MLP"], kw_only=False):
        """The widths and the three pieces; the projections' widths derive from them."""

        channels_in: int = -1
        """Input width; a parent may push it."""

        channels_out: int = -1
        """Output width; -1 takes ``channels_in``."""

        _: KW_ONLY

        channels_hidden: int = -1
        """Width between the projections; -1 takes ``channels_in``."""

        proj_in: Linear.Config = field(default_factory=Linear.Config)
        """``channels_in -> channels_hidden``."""

        activation: TensorFn = functional.silu
        """Applied between the projections."""

        proj_out: Linear.Config = field(default_factory=Linear.Config)
        """``channels_hidden -> channels_out``."""

        @override
        def finalize(self) -> Self:
            if self.channels_hidden == -1:
                self.channels_hidden = self.channels_in
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            self.proj_in.channels_in = self.channels_in
            self.proj_in.channels_out = self.channels_hidden
            self.proj_out.channels_in = self.channels_hidden
            self.proj_out.channels_out = self.channels_out
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost both projections and the activation between them.

            Args:
              seq_len: Rows per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            rows = seq_len * batch_size
            return (
                self.proj_in.cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + activation_cost(
                    self.activation,
                    elements=rows * self.channels_hidden,
                    dtype=dtype,
                )
                + self.proj_out.cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.proj_in = config.proj_in.make()
        self.activation = config.activation
        self.proj_out = config.proj_out.make()

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Map ``[..., channels_in]`` to ``[..., channels_out]``."""
        return self.proj_out(self.activation(self.proj_in(input)))


class DepthwiseResidual(nn.Module):
    """A residual block over a board: ``x + pointwise(activation(depthwise(x)))``."""

    class Config(Fig["DepthwiseResidual"]):
        """The board's channels; both convolutions keep them."""

        channels_in: int = -1
        """Channels of the board, in and out."""

        depthwise: Conv2d.Config = field(default_factory=Conv2d.Config)
        """Each channel's own 3 x 3 window, zero-padded; its groups are the channels."""

        activation: TensorFn = functional.silu
        """Applied between the convolutions."""

        pointwise: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(
                kernel_size=1,
                init_weight=nn.init.zeros_,
            ),
        )
        """The 1 x 1 mix back into the board; zero at init, so the block starts
        as the identity."""

        @override
        def finalize(self) -> Self:
            channels = self.channels_in
            self.depthwise.channels_in = self.depthwise.channels_out = channels
            self.depthwise.groups = channels
            self.pointwise.channels_in = self.pointwise.channels_out = channels
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: tuple[int, int],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost both convolutions, the activation, and the residual add.

            Args:
              input_grid: Rows and columns of each board.
              batch_size: Boards in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            rows = batch_size * math.prod(input_grid)
            return (
                self.depthwise.cost(
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + activation_cost(
                    self.activation,
                    elements=rows * self.channels_in,
                    dtype=dtype,
                )
                + self.pointwise.cost(
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + residual_cost(channels=self.channels_in, rows=rows, dtype=dtype)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.depthwise = config.depthwise.make()
        self.activation = config.activation
        self.pointwise = config.pointwise.make()

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Update a board ``[batch, channels, rows, columns]``."""
        return input + self.pointwise(self.activation(self.depthwise(input)))


class ConvNextBlock(nn.Module):
    """A ConvNeXt block over a board: a depthwise window, then a scaled per-cell MLP.

        x + layer_scale * mlp(norm(depthwise(x)))

    The norm and the MLP read each cell's channels; ``layer_scale`` is a
    learned gain per channel that starts near zero, so the block starts near
    the identity.
    """

    class Config(Fig["ConvNextBlock"]):
        """The board's channels and the MLP's expansion; every width derives from them."""

        channels_in: int = -1
        """Channels of the board, in and out."""

        expansion: int = 4
        """The MLP's hidden width, as a multiple of ``channels_in``."""

        depthwise: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(kernel_size=5),
        )
        """Each channel's own 5 x 5 window, zero-padded; its groups are the channels."""

        norm: LayerNorm.Config = field(
            default_factory=lambda: LayerNorm.Config(eps=1e-6, elementwise_affine=True),
        )
        """Each cell's channels, normalized, with a learned scale and shift."""

        mlp: MLP.Config = field(
            default_factory=lambda: MLP.Config(
                proj_in=Linear.Config(bias=True),
                activation=functional.gelu,
                proj_out=Linear.Config(bias=True),
            ),
        )
        """``channels_in -> expansion * channels_in -> channels_in``, per cell.
        Each projection's bias draws torch's U(+-1/sqrt(fan_in)), its fan-in
        set from the widths."""

        layer_scale: float = 1e-6
        """Every channel's gain on the MLP's output, at init."""

        @override
        def finalize(self) -> Self:
            channels = self.channels_in
            hidden = self.expansion * channels
            self.depthwise.channels_in = self.depthwise.channels_out = channels
            self.depthwise.groups = channels
            self.norm.channels_in = channels
            self.mlp.channels_in = channels
            self.mlp.channels_hidden = hidden
            self.mlp.proj_in.init_bias = FanInUniform(fan_in=channels)
            self.mlp.proj_out.init_bias = FanInUniform(fan_in=hidden)
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: tuple[int, int],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the window, the per-cell norm and MLP, the gain, and the residual.

            The gain is one multiply per value forward; back, one for the
            input's gradient and one for its own, summed over every cell.

            Args:
              input_grid: Rows and columns of each board.
              batch_size: Boards in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            cells = math.prod(input_grid)
            rows, channels = batch_size * cells, self.channels_in
            gain = elementwise_cost(
                primal=rows * channels,
                adjoint=2 * rows * channels,
                channels=channels,
                params=channels,
                rows=rows,
                dtype=dtype,
            )
            return (
                self.depthwise.cost(
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + self.norm.cost(
                    seq_len=cells,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + self.mlp.cost(
                    seq_len=cells,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + gain
                + residual_cost(channels=channels, rows=rows, dtype=dtype)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.depthwise = config.depthwise.make()
        self.norm = config.norm.make()
        self.mlp = config.mlp.make()
        self.layer_scale = nn.Parameter(
            torch.full((config.channels_in,), config.layer_scale),
        )

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Update a board ``[batch, channels, rows, columns]``."""
        mixed = self.depthwise(input).permute(0, 2, 3, 1)
        mixed = self.mlp(self.norm(mixed))
        return input + (mixed * self.layer_scale).permute(0, 3, 1, 2)


class ConvNextTrunk(nn.Module):
    """A residual stage at a wider width: lift the board, run blocks, project back.

        x + proj_out(blocks(stem(x)))

    ``proj_out`` starts at zero, so the stage starts as the identity.
    """

    class Config(Fig["ConvNextTrunk"]):
        """The board's channels and the stage's own; every width derives from them."""

        channels_in: int = -1
        """Channels of the board, in and out."""

        channels_hidden: int = 64
        """Channels inside the stage."""

        stem: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1),
        )
        """``channels_in -> channels_hidden``, 1 x 1."""

        block: ConvNextBlock.Config = field(default_factory=ConvNextBlock.Config)
        """Block template, broadcast ``num_layers`` times."""

        num_layers: int = 2
        """Blocks inside the stage."""

        proj_out: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(
                kernel_size=1,
                init_weight=nn.init.zeros_,
            ),
        )
        """``channels_hidden -> channels_in``, 1 x 1; zero at init."""

        @override
        def finalize(self) -> Self:
            hidden = self.channels_hidden
            self.stem.channels_in = self.channels_in
            self.stem.channels_out = hidden
            self.block.channels_in = hidden
            self.proj_out.channels_in = hidden
            self.proj_out.channels_out = self.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: tuple[int, int],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the stem, every block, the projection back, and the residual.

            Args:
              input_grid: Rows and columns of each board.
              batch_size: Boards in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            rows = batch_size * math.prod(input_grid)
            blocks = self.block.cost(
                input_grid=input_grid,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ).tile(self.num_layers, copies=self.num_layers)
            return (
                self.stem.cost(
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + blocks
                + self.proj_out.cost(
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + residual_cost(channels=self.channels_in, rows=rows, dtype=dtype)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.stem = config.stem.make()
        self.blocks = nn.ModuleList(
            config.block.copy_tree().make() for _ in range(config.num_layers)
        )
        self.proj_out = config.proj_out.make()

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Update a board ``[batch, channels, rows, columns]``."""
        board = self.stem(input)
        for block in self.blocks:
            board = block(board)
        return input + self.proj_out(board)


@dataclass(frozen=True, slots=True, kw_only=True)
class FanInUniform:
    """Torch's default draw for a layer's bias, ``U(+-1/sqrt(fan_in))``, in place.

    The bias does not carry its layer's fan-in, so the draw holds it; as a value
    it compares and prints, where a bound ``partial`` would not compare.
    """

    fan_in: int
    """The layer's input width."""

    def __call__(self, bias: Tensor) -> None:
        """Fill ``bias`` with the draw."""
        bound = self.fan_in**-0.5
        nn.init.uniform_(bias, -bound, bound)


class FilmBranch(nn.Module):
    """A wide residual branch over a board, scaled and shifted by a condition (FiLM).

    The board is widened by a 1 x 1 convolution; the condition gives each wide
    channel a scale ``1 + gamma`` and a shift ``beta``, shared by every cell;
    a depthwise 3 x 3 window and a 1 x 1 convolution map it back:

        x + proj_out(act(depthwise(act(proj_in(x)) * (1 + gamma) + beta)))
    """

    class Config(Fig["FilmBranch"]):
        """The widths; every convolution's and the condition's derive from them."""

        channels_in: int = -1
        """Channels of the board, in and out."""

        channels_hidden: int = 128
        """Channels of the widened board."""

        channels_condition: int = -1
        """Width of the condition vector."""

        proj_in: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1),
        )
        """``channels_in -> channels_hidden``, 1 x 1."""

        proj_condition: Linear.Config = field(default_factory=Linear.Config)
        """``channels_condition -> 2 * channels_hidden``: gamma, then beta."""

        depthwise: Conv2d.Config = field(default_factory=Conv2d.Config)
        """Each wide channel's own 3 x 3 window, zero-padded."""

        activation: TensorFn = functional.silu
        """Applied after ``proj_in`` and after ``depthwise``."""

        proj_out: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(
                kernel_size=1,
                init_weight=nn.init.zeros_,
            ),
        )
        """``channels_hidden -> channels_in``, 1 x 1; zero at init."""

        @override
        def finalize(self) -> Self:
            hidden = self.channels_hidden
            self.proj_in.channels_in = self.channels_in
            self.proj_in.channels_out = hidden
            self.proj_condition.channels_in = self.channels_condition
            self.proj_condition.channels_out = 2 * hidden
            self.depthwise.channels_in = self.depthwise.channels_out = hidden
            self.depthwise.groups = hidden
            self.proj_out.channels_in = hidden
            self.proj_out.channels_out = self.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: tuple[int, int],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the widening, the condition's projection and FiLM, the window, the way back.

            FiLM adds 1 to each board's scale, then multiplies and shifts every
            wide value; back, it multiplies the gradient by the scale and by the
            value, and sums both products' and the plain gradient's share over
            each board's cells.

            Args:
              input_grid: Rows and columns of each board.
              batch_size: Boards in this invocation, one condition each.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            rows = batch_size * math.prod(input_grid)
            wide = rows * self.channels_hidden
            film = elementwise_cost(
                primal=2 * wide + batch_size * self.channels_hidden,
                adjoint=2 * wide,
                channels=self.channels_hidden,
                rows=rows,
                dtype=dtype,
            ) + reduction_cost(
                input_elements=2 * wide,
                output_groups=2 * batch_size * self.channels_hidden,
                dtype=dtype,
                phase="adjoint",
            )
            return (
                self.proj_in.cost(
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + activation_cost(self.activation, elements=wide, dtype=dtype)
                + self.proj_condition.cost(
                    seq_len=1,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + film
                + self.depthwise.cost(
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + activation_cost(self.activation, elements=wide, dtype=dtype)
                + self.proj_out.cost(
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + residual_cost(channels=self.channels_in, rows=rows, dtype=dtype)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.proj_in = config.proj_in.make()
        self.proj_condition = config.proj_condition.make()
        self.depthwise = config.depthwise.make()
        self.activation = config.activation
        self.proj_out = config.proj_out.make()

    @override
    def forward(self, input: Tensor, condition: Tensor) -> Tensor:
        """Update a board ``[batch, channels, rows, columns]`` by ``[batch, condition]``."""
        spatial = self.activation(self.proj_in(input))
        scale, shift = self.proj_condition(condition).chunk(2, dim=-1)
        spatial = spatial * (1 + scale[..., None, None]) + shift[..., None, None]
        spatial = self.proj_out(self.activation(self.depthwise(spatial)))
        return input + spatial


class MultiscaleMix(nn.Module):
    """A residual mix of each cell with its neighbourhood's and the board's averages.

    Each cell's channels, their average over its ``kernel_size`` window (zero
    padding counted) and their average over the whole board are concatenated
    and mixed by two 1 x 1 convolutions back into the board.
    """

    class Config(Fig["MultiscaleMix"]):
        """The widths and the local window; the convolutions' widths derive from them."""

        channels_in: int = -1
        """Channels of the board, in and out."""

        channels_hidden: int = 32
        """Channels between the two convolutions."""

        kernel_size: int = 3
        """Side of the local average's window: stride 1, padded to keep the board."""

        proj_in: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(kernel_size=1),
        )
        """``3 * channels_in -> channels_hidden``, 1 x 1."""

        activation: TensorFn = functional.silu
        """Applied between the convolutions."""

        proj_out: Conv2d.Config = field(
            default_factory=lambda: Conv2d.Config(
                kernel_size=1,
                init_weight=nn.init.zeros_,
            ),
        )
        """``channels_hidden -> channels_in``, 1 x 1; zero at init."""

        @override
        def finalize(self) -> Self:
            self.proj_in.channels_in = 3 * self.channels_in
            self.proj_in.channels_out = self.channels_hidden
            self.proj_out.channels_in = self.channels_hidden
            self.proj_out.channels_out = self.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            input_grid: tuple[int, int],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost both averages, the concatenation, both convolutions, and the residual.

            Each local average sums its window and divides once; its gradient
            spreads a share to every cell of the window. The whole-board mean
            sums each board's cells per channel and divides once; its gradient
            is one share per value.

            Args:
              input_grid: Rows and columns of each board.
              batch_size: Boards in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            rows, channels = batch_size * math.prod(input_grid), self.channels_in
            values, window = rows * channels, self.kernel_size**2
            averages = (
                reduction_cost(
                    input_elements=window * values,
                    output_groups=values,
                    dtype=dtype,
                )
                + reduction_cost(
                    input_elements=values,
                    output_groups=batch_size * channels,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=values + batch_size * channels,
                    adjoint=(window + 1) * values,
                    channels=channels,
                    rows=rows,
                    dtype=dtype,
                )
            )
            return (
                averages
                + concat_cost(elements=3 * values, dtype=dtype)
                + self.proj_in.cost(
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + activation_cost(
                    self.activation,
                    elements=rows * self.channels_hidden,
                    dtype=dtype,
                )
                + self.proj_out.cost(
                    input_grid=input_grid,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + residual_cost(channels=channels, rows=rows, dtype=dtype)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.kernel_size = config.kernel_size
        self.proj_in = config.proj_in.make()
        self.activation = config.activation
        self.proj_out = config.proj_out.make()

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Update a board ``[batch, channels, rows, columns]``."""
        # The board feeds the pool, the mean, the concatenation and the residual
        # add, and autograd sums its four gradients in the reverse of that order,
        # in bf16. Taking the mean before the pool moved 230 of the table's 2,464
        # gradients against the reference (measured).
        local = functional.avg_pool2d(
            input,
            self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        whole = input.mean(dim=(-2, -1), keepdim=True).expand_as(input)
        mixed = self.proj_in(torch.cat((input, local, whole), dim=1))
        return input + self.proj_out(self.activation(mixed))


class BoardCNN(nn.Module):
    """The embedded cells as a board: residual blocks, a FiLM branch, a multiscale mix."""

    class Config(Fig["BoardCNN"]):
        """The board's geometry; each stage's widths derive from it."""

        channels_in: int = -1
        """Channels per cell: the cell embedding's width."""

        channels_condition: int = -1
        """Width of the condition the FiLM branch reads."""

        grid: tuple[int, int] = (9, 11)
        """Rows and columns of the view, whose cells arrive row-major."""

        block: Makeable[nn.Module] = field(default_factory=DepthwiseResidual.Config)
        """Residual block template over the board, broadcast ``num_layers``
        times; it maps ``channels_in`` channels to as many (a
        :class:`ConvNextTrunk` is one block)."""

        num_layers: int = 2
        """Residual blocks before the FiLM branch."""

        spatial: FilmBranch.Config = field(default_factory=FilmBranch.Config)
        """The branch the condition modulates."""

        multiscale: MultiscaleMix.Config = field(default_factory=MultiscaleMix.Config)
        """The last stage: local and whole-board averages."""

        @override
        def finalize(self) -> Self:
            propagate_attr(
                self.block,
                "channels_in",
                self.channels_in,
                protocol=ChannelsIn,
            )
            self.spatial.channels_in = self.channels_in
            self.spatial.channels_condition = self.channels_condition
            self.multiscale.channels_in = self.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost every stage over one board per row; the layout changes are views.

            Args:
              seq_len: Rows per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            blocks = cost(
                self.block,
                input_grid=self.grid,
                batch_size=seq_len * batch_size,
                dtype=dtype,
                **kwargs,
            ).tile(self.num_layers, copies=self.num_layers)
            return (
                blocks
                + self.spatial.cost(
                    input_grid=self.grid,
                    batch_size=seq_len * batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + self.multiscale.cost(
                    input_grid=self.grid,
                    batch_size=seq_len * batch_size,
                    dtype=dtype,
                    **kwargs,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.grid = config.grid
        self.channels = config.channels_in
        self.blocks = nn.ModuleList(
            config.block.copy_tree().make() for _ in range(config.num_layers)
        )
        self.spatial = config.spatial.make()
        self.multiscale = config.multiscale.make()

    @override
    def forward(self, cells: Tensor, condition: Tensor) -> Tensor:
        """Encode ``[..., cells * channels]`` cells, conditioned on ``[..., condition]``.

        The board is ``[batch, channels, rows, columns]`` over the cells'
        memory, so each convolution reads it channels-last.
        """
        leading = cells.shape[:-1]
        board = cells.reshape(-1, *self.grid, self.channels).permute(0, 3, 1, 2)
        for block in self.blocks:
            board = cast("Tensor", block(board))
        board = self.spatial(board, condition.reshape(-1, condition.shape[-1]))
        board = self.multiscale(board)
        return board.permute(0, 2, 3, 1).reshape(*leading, -1)


class ActionConditionedStatus(nn.Module):
    """Residual MLPs over the status scalars, each reading the previous action's embedding."""

    class Config(Fig["ActionConditionedStatus"]):
        """The status width, the action table and the block template."""

        channels_in: int = -1
        """Status scalars, in and out."""

        embedding: Embedding.Config = field(
            default_factory=lambda: Embedding.Config(
                44,
                32,
                init_weight=partial(normal, std=1.0),
            ),
        )
        """The previous action's table: a row per action, and one (id 43) for
        none, at an episode's start. N(0, 1), as ``nn.Embedding`` draws."""

        block: MLP.Config = field(
            default_factory=lambda: MLP.Config(
                channels_hidden=192,
                proj_out=Linear.Config(init_weight=nn.init.zeros_),
            ),
        )
        """Residual branch template, ``status + action -> status``; zero output
        at init."""

        num_layers: int = 2
        """Residual blocks."""

        @override
        def finalize(self) -> Self:
            self.block.channels_in = self.channels_in + self.embedding.channels_out
            self.block.channels_out = self.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the action's lookup and each block's concatenation, MLP, and residual.

            Args:
              seq_len: Rows per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            rows, channels = seq_len * batch_size, self.channels_in
            block = (
                concat_cost(elements=rows * self.block.channels_in, dtype=dtype)
                + self.block.cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + residual_cost(channels=channels, rows=rows, dtype=dtype)
            )
            return self.embedding.cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + block.tile(self.num_layers, copies=self.num_layers)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.embedding = config.embedding.make()
        self.blocks = nn.ModuleList(
            config.block.copy_tree().make() for _ in range(config.num_layers)
        )

    @override
    def forward(self, status: Tensor, action: Tensor) -> Tensor:
        """Refine ``[..., channels_in]`` scalars given ``[...]`` action ids as floats."""
        condition = self.embedding(action.long())
        for block in self.blocks:
            status = status + block(torch.cat((status, condition), dim=-1))
        return status


class BoardEncoder(nn.Module):
    """The packed observation and the previous action, encoded as the multi-hot embedding's features."""

    class Config(Fig["BoardEncoder"]):
        """The three stages; their widths derive from the cells' layout."""

        cells: MultiHotEmbedding.Config = field(
            default_factory=packed_observation_embedding,
        )
        """The multi-hot embedding of the packed row."""

        status: ActionConditionedStatus.Config = field(
            default_factory=ActionConditionedStatus.Config,
        )
        """The status scalars, refined given the previous action."""

        board: BoardCNN.Config = field(default_factory=BoardCNN.Config)
        """The embedded cells as a board, conditioned on the refined status."""

        @property
        def observation_size(self) -> int:
            """Floats per observation: the packed row, then the previous action id."""
            return self.cells.observation_size + 1

        @property
        def channels_concat(self) -> int:
            """Width of the features: every cell's channels, then the status."""
            return self.cells.channels_concat

        @override
        def finalize(self) -> Self:
            self.status.channels_in = self.cells.num_scalars
            self.board.channels_in = self.cells.channels_out
            self.board.channels_condition = self.cells.num_scalars
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the three stages and the features' concatenation.

            Args:
              seq_len: Rows per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            rows = seq_len * batch_size
            return (
                self.cells.cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + self.status.cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + self.board.cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + concat_cost(elements=rows * self.channels_concat, dtype=dtype)
            )

    def __init__(self, config: Config) -> None:
        """Build the three stages.

        Args:
          config: The stages.

        Raises:
          ValueError: The board's grid does not hold the embedding's cells.

        """
        if math.prod(config.board.grid) != config.cells.num_cells:
            msg = (
                f"a {config.board.grid} board does not hold "
                f"{config.cells.num_cells} cells"
            )
            raise ValueError(msg)
        super().__init__()
        self.packed_size = config.cells.observation_size
        self.cells_width = config.cells.num_cells * config.cells.channels_out
        self.cells = config.cells.make()
        self.status = config.status.make()
        self.board = config.board.make()

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Map observations ``[..., observation_size]`` to ``[..., channels_concat]``."""
        features = self.cells(input[..., : self.packed_size])
        action = input[..., self.packed_size]
        status = self.status(features[..., self.cells_width :], action)
        board = self.board(features[..., : self.cells_width], status)
        return torch.cat((board, status), dim=-1)
