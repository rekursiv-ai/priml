"""Input-embedding channels for grid-puzzle models.

A puzzle model's input is a token grid plus some number of ADDITIONAL signals
added onto it: learned structural positions, a per-puzzle prefix, the previous
step's own prediction. Which signals apply is a property of the puzzle, not of
the network, so each is a separate module filling a slot rather than an
``if``-gated branch inside one embedding method.

Earlier implementations hardwired exactly these branches -- register tokens
gated on ``register_tokens is not None``, 2D positions on a grid shape being
set, feedback on a stashed tensor -- so adding a puzzle meant editing one
shared embedding method. Here each signal is a :class:`GridChannel`
implementation and a puzzle supplies a list.

The composition order is a NUMERICS contract: every channel is added to the
grid-token embedding in list order, so reordering the list changes the
floating-point sum and therefore the trained result. It is a config field
precisely so the order is visible in ``pprint`` rather than buried in a method.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Protocol, Self, override, runtime_checkable

import math

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.model.cost import (
    Cost,
    cost,
    elementwise_cost,
    reduction_cost,
    traffic,
    with_rows,
)
from priml.model.custom_types import ChannelsIn, ChannelsOut
from priml.model.embedding import Embedding
from priml.model.init import truncated_normal


@runtime_checkable
class GridChannel(Protocol):
    """An additive contribution to the grid-token embeddings.

    Implementations return a tensor broadcastable onto ``[B, grid_len, C]``
    and are summed onto the token embedding in the order the config lists
    them.
    """

    def forward(self, tokens: Tensor, embeddings: Tensor) -> Tensor:
        """Return this channel's contribution for one batch.

        Args:
          tokens: ``[B, grid_len]`` input token ids.
          embeddings: ``[B, grid_len, C]`` token embeddings built so far.

        Returns:
          contribution: Broadcastable to ``embeddings``' shape, or a
            zero-element tensor to contribute nothing this step.

        """
        ...


@runtime_checkable
class HasGridLen(Protocol):
    """A channel config sized by the puzzle's grid-token count."""

    grid_len: int


class FactoredPositions(nn.Module):
    """Learned row + column + box position tables for a 2D puzzle grid.

    Sudoku's constraint structure is row/column/box, so a position is better
    described by which row, column, and box a cell belongs to than by its index
    in a flattened sequence. Three small tables cost
    ``(rows + cols + boxes) * C`` parameters against ``rows * cols * C`` for a
    dense table, and share statistics across cells that share a constraint.

    Set ``box_shape`` to ``(1, 1)`` for a puzzle with no box structure; the box
    table then degenerates to one entry per cell and the sum is row + column
    positions plus a constant.
    """

    class Config(Fig["FactoredPositions"]):
        """Grid factorization and table initialization."""

        grid_shape: tuple[int, int] = (9, 9)
        """``(rows, cols)`` factorization of the flat grid."""

        box_shape: tuple[int, int] = (3, 3)
        """``(rows, cols)`` of one constraint box tiling the grid."""

        channels_out: int = -1
        """Cost width; -1 inherits the model's hidden size."""

        init_std: float = 1.0
        """Realized standard deviation of the table entries."""

        embed_scale: float = -1.0
        """Runtime multiplier applied to the tables; -1 inherits the model's.

        The embedding-rescale trick initializes tables at ``1/sqrt(C)`` and
        multiplies by ``sqrt(C)`` at runtime, so this must match whatever the
        token embedding uses or the channels enter at different magnitudes."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Price three table gathers, two adds, and a scale per cell.

            Each table is a gather and a scatter-add back, as
            :class:`~priml.model.embedding.Embedding` prices one; the adds
            pass gradient through, so only the scale pulls one back. The
            ``[grid_len, C]`` sum runs once and broadcasts over puzzles; its
            adjoint first reduces the broadcast gradient across puzzles.

            A cell is the token and the grid is fixed by ``grid_shape``, so
            ``seq_len`` and its rows are ignored: the batch is
            ``batch_size`` whole grids.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-cell cost of this module.

            """
            puzzles = batch_size
            dt = dtype
            grid_rows, grid_cols = self.grid_shape
            box_rows, box_cols = self.box_shape
            width = self.channels_out
            tables = (
                grid_rows,
                grid_cols,
                (grid_rows // box_rows) * (grid_cols // box_cols),
            )
            del seq_len
            cells = grid_rows * grid_cols
            gathers = sum(
                (
                    cost(
                        Embedding.Config(channels_in=n, channels_out=width),
                        seq_len=cells,
                        batch_size=1,
                        dtype=dt,
                        **with_rows(cells, **kwargs),
                    )
                    for n in tables
                ),
                Cost(),
            )
            shared = gathers + elementwise_cost(
                primal=3 * width,
                adjoint=width,
                channels=width,
                inputs=5,
                outputs=3,
                adjoint_inputs=1,
                dtype=dt,
            )
            broadcast = (
                reduction_cost(
                    input_elements=puzzles * width,
                    output_groups=width,
                    rows=puzzles,
                    dtype=dt,
                    phase="adjoint",
                )
                if puzzles > 1
                else Cost()
            )
            return shared.tile(1 / puzzles) + broadcast

    def __init__(self, config: Config) -> None:
        super().__init__()
        rows, cols = config.grid_shape
        box_rows, box_cols = config.box_shape
        if rows % box_rows or cols % box_cols:
            raise ValueError(
                f"box_shape {config.box_shape} does not tile grid_shape "
                f"{config.grid_shape}.",
            )
        if config.channels_out <= 0:
            raise ValueError(
                f"channels_out must be positive; got {config.channels_out}. It is "
                "normally inherited from the model during finalize.",
            )
        self.config = config
        self.embed_scale: float = (
            config.channels_out**0.5 if config.embed_scale < 0 else config.embed_scale
        )
        cell = torch.arange(rows * cols)
        row = cell // cols
        col = cell % cols
        box = (row // box_rows) * (cols // box_cols) + col // box_cols
        self.row_index = nn.Buffer(row, persistent=False)
        self.col_index = nn.Buffer(col, persistent=False)
        self.box_index = nn.Buffer(box, persistent=False)
        num_boxes = (rows // box_rows) * (cols // box_cols)
        # Init order is a checkpoint-parity contract: row, then column, then
        # box. Each draws from the global RNG, so reordering changes every
        # seeded run's weights.
        width, std = config.channels_out, config.init_std
        self.embed_pos_row = _table(rows, channels_out=width, init_std=std)
        self.embed_pos_col = _table(cols, channels_out=width, init_std=std)
        self.embed_pos_box = _table(num_boxes, channels_out=width, init_std=std)

    @override
    def forward(self, tokens: Tensor, embeddings: Tensor) -> Tensor:
        """Return ``[grid_len, C]`` positions, broadcast over the batch."""
        del tokens
        positions: Tensor = (
            self.embed_pos_row[self.row_index]
            + self.embed_pos_col[self.col_index]
            + self.embed_pos_box[self.box_index]
        )
        scaled: Tensor = self.embed_scale * positions.to(dtype=embeddings.dtype)
        return scaled


class PredictionFeedback(nn.Module):
    """Re-embeds the previous step's own decoded grid onto the input.

    A recurrent solver that re-reads only the original puzzle cannot condition
    on what it currently believes; this channel closes that loop symbolically.
    The caller stashes the previous step's argmax grid (with the puzzle's given
    cells clamped back) via :meth:`set_feedback`, and it is consumed exactly
    once, so a stale grid can never leak into a later forward.

    Zero-initialized by default, which makes the first forward bit-identical to
    a model without the channel -- an A/B against the no-feedback baseline
    therefore starts from the same weights and must LEARN to use it.
    """

    class Config(Fig["PredictionFeedback"], kw_only=False):
        """Feedback-table size and initialization."""

        channels_in: int = -1
        """Token vocabulary; -1 inherits the model's."""

        channels_out: int = -1
        """Cost width; -1 inherits the model's hidden size."""

        _: KW_ONLY

        init_std: float = 0.0
        """Realized standard deviation; 0 zero-initializes the table."""

        embed_scale: float = -1.0
        """Runtime multiplier; -1 inherits the model's (see
        :class:`FactoredPositions.Config.embed_scale`)."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Price one table gather and a scale per cell.

            Priced with a grid stashed, as every step under adaptive
            computation time is; a forward without one contributes nothing.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-cell cost of this module.

            """
            width = self.channels_out
            table = Embedding.Config(channels_in=self.channels_in, channels_out=width)
            return cost(
                table,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + elementwise_cost(
                primal=width,
                adjoint=width,
                channels=width,
                adjoint_inputs=1,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.channels_out <= 0 or config.channels_in <= 0:
            raise ValueError(
                f"channels_in and channels_out must be positive; got "
                f"{config.channels_in} and {config.channels_out}. Both are "
                "normally inherited from the model during finalize.",
            )
        self.config = config
        self.embed_scale: float = (
            config.channels_out**0.5 if config.embed_scale < 0 else config.embed_scale
        )
        self.embed_feedback = _table(
            config.channels_in,
            channels_out=config.channels_out,
            init_std=config.init_std,
        )
        self._feedback_ids: Tensor | None = None

    def set_feedback(self, feedback_ids: Tensor | None) -> None:
        """Stash the decoded grid the NEXT forward consumes.

        Args:
          feedback_ids: ``[B, grid_len]`` token ids of the previous step's
            decoded grid, givens already clamped by the caller, or None to
            contribute nothing.

        """
        self._feedback_ids = feedback_ids

    @override
    def forward(self, tokens: Tensor, embeddings: Tensor) -> Tensor:
        """Return the stashed grid's embedding, consuming the stash."""
        del tokens
        feedback = self._feedback_ids
        self._feedback_ids = None  # Consume-once: stale grids never leak.
        if feedback is None:
            empty: Tensor = embeddings.new_zeros(())
            return empty
        rows: Tensor = self.embed_feedback[feedback].to(dtype=embeddings.dtype)
        scaled: Tensor = self.embed_scale * rows
        return scaled


class GridEmbedding(nn.Module):
    """Token embedding for a puzzle grid, plus a list of additive channels.

    The base embedding uses the rescale trick -- tables initialized at
    ``1/sqrt(C)`` and multiplied by ``sqrt(C)`` at runtime -- so every channel
    must apply the same scale to enter at a comparable magnitude. Channels are
    summed in list order onto the grid tokens ONLY; a prefix (register tokens,
    a per-puzzle embedding) is prepended by the model, after this returns.
    """

    class Config(Fig["GridEmbedding"], kw_only=False):
        """Vocabulary, width, and the additive channel list."""

        channels_in: int = -1
        """Token vocabulary size: the one-hot input width the table is linear over."""

        channels_out: int = -1
        """Embedding width; -1 inherits the model's own."""

        _: KW_ONLY

        grid_shape: tuple[int, ...] = (81,)
        """Token layout per puzzle. A flat ``(81,)`` and a ``(9, 9)`` grid
        describe the same 81 tokens; the shape is what a channel factorizes."""

        channels: list[Makeable[GridChannel]] = field(
            default_factory=list[Makeable[GridChannel]],
        )
        """Additive channels, summed onto the grid tokens IN THIS ORDER.

        The order is a numerics contract: floating-point addition is not
        associative, so reordering changes the trained result. Empty is the
        plain baseline -- token embeddings alone."""

        @property
        def grid_len(self) -> int:
            """Number of grid tokens per puzzle."""
            return math.prod(self.grid_shape)

        @override
        def finalize(self) -> Self:
            for channel in self.channels:
                if isinstance(channel, ChannelsOut) and channel.channels_out == -1:
                    channel.channels_out = self.channels_out
                if isinstance(channel, HasGridLen) and channel.grid_len == -1:
                    channel.grid_len = self.grid_len
                if isinstance(channel, ChannelsIn) and channel.channels_in == -1:
                    channel.channels_in = self.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Price the token table, its scale, and every channel plus one add each.

            The stream before an add feeds a channel only for its dtype, so
            the add's adjoint is a pass-through with no accumulation and only
            its primal is counted.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-cell cost of this module.

            """
            width = self.channels_out
            total = cost(
                _token_table(self),
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + elementwise_cost(
                primal=width,
                adjoint=width,
                channels=width,
                adjoint_inputs=1,
                dtype=dtype,
            )
            add = traffic(
                "primal",
                "elementwise",
                elements=3 * width,
                flops=width,
                dtype=dtype,
            )
            for channel in self.channels:
                total += (
                    cost(
                        channel,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        dtype=dtype,
                        **kwargs,
                    )
                    + add
                )
            return total

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.channels_in <= 0:
            raise ValueError(
                f"channels_in must be positive; got {config.channels_in}. It "
                "is normally inherited from the model during finalize.",
            )
        if config.channels_out <= 0:
            raise ValueError(
                f"channels_out must be positive; got {config.channels_out}. It "
                "is normally inherited from the model during finalize.",
            )
        self.config = config
        self.embed_scale: float = config.channels_out**0.5
        self.embed_tokens = _token_table(config).make()
        truncated_normal(
            self.embed_tokens.weight,
            std=1.0 / self.embed_scale,
            depth_index=(),
            variance_correction=True,
        )
        # The slot is typed by what a channel DOES (``GridChannel``) while
        # ``ModuleList`` holds what it IS, and iterating one yields a bare
        # ``Module`` whose ``__call__`` says nothing. Keep the typed list beside
        # it: the ModuleList owns registration (parameters, device moves) and
        # this owns the call contract, so ``forward`` needs no cast.
        built: list[nn.Module] = []
        self._channels: list[GridChannel] = []
        for channel_config in config.channels:
            channel = channel_config.make()
            assert isinstance(channel, nn.Module), type(channel).__name__
            built.append(channel)
            self._channels.append(channel)
        self.channels = nn.ModuleList(built)

    @override
    def forward(self, tokens: Tensor) -> Tensor:
        """Embed ``[B, grid_len]`` tokens and add every channel in order."""
        tokens_emb: Tensor = self.embed_tokens(tokens)
        embeddings: Tensor = self.embed_scale * tokens_emb
        for channel in self._channels:
            contribution = channel.forward(tokens, embeddings)
            if contribution.numel():
                embeddings = embeddings + contribution
        return embeddings


def _token_table(config: GridEmbedding.Config) -> Embedding.Config:
    """Configure the grid-token table at the model's width and vocabulary."""
    return Embedding.Config(
        channels_in=config.channels_in,
        channels_out=config.channels_out,
    )


# The initializer's std is divided by ``sqrt(channels_out)`` because the caller
# multiplies by that factor at runtime; the two cancel to ``init_std``.
def _table(
    num_embeddings: int,
    *,
    channels_out: int,
    init_std: float,
) -> nn.Parameter:
    """Return a learned table whose REALIZED std is ``init_std`` after rescaling."""
    w = torch.zeros(num_embeddings, channels_out)
    if init_std > 0:
        truncated_normal(
            w,
            std=init_std / channels_out**0.5,
            depth_index=(),
            variance_correction=True,
        )
    return nn.Parameter(w)
