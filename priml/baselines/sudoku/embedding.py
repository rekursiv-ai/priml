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

from dataclasses import field
from typing import Protocol, Self, override, runtime_checkable

import math

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.model.embedding import Embedding
from priml.model.init import truncated_normal


@runtime_checkable
class GridChannel(Protocol):
    """An additive contribution to the grid-token embeddings.

    Implementations return a tensor broadcastable onto ``[B, grid_len, C]``
    and are summed onto the token embedding in the order the config lists
    them.
    """

    def __call__(self, tokens: Tensor, embeddings: Tensor) -> Tensor:
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
class HasHiddenSize(Protocol):
    """A channel config sized by the model's hidden width."""

    hidden_size: int


@runtime_checkable
class HasGridLen(Protocol):
    """A channel config sized by the puzzle's grid-token count."""

    grid_len: int


@runtime_checkable
class HasVocabSize(Protocol):
    """A channel config sized by the token vocabulary."""

    vocab_size: int


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

        hidden_size: int = -1
        """Table width; -1 inherits the model's hidden size."""

        init_std: float = 1.0
        """Realized standard deviation of the table entries."""

        embed_scale: float = -1.0
        """Runtime multiplier applied to the tables; -1 inherits the model's.

        The embedding-rescale trick initializes tables at ``1/sqrt(C)`` and
        multiplies by ``sqrt(C)`` at runtime, so this must match whatever the
        token embedding uses or the channels enter at different magnitudes."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        rows, cols = config.grid_shape
        box_rows, box_cols = config.box_shape
        if rows % box_rows or cols % box_cols:
            raise ValueError(
                f"box_shape {config.box_shape} does not tile grid_shape "
                f"{config.grid_shape}.",
            )
        if config.hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive; got {config.hidden_size}. It is "
                "normally inherited from the model during finalize.",
            )
        self.config = config
        self.embed_scale = (
            config.hidden_size**0.5 if config.embed_scale < 0 else config.embed_scale
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
        width, std = config.hidden_size, config.init_std
        self.embed_pos_row = _table(rows, hidden_size=width, init_std=std)
        self.embed_pos_col = _table(cols, hidden_size=width, init_std=std)
        self.embed_pos_box = _table(num_boxes, hidden_size=width, init_std=std)

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

    class Config(Fig["PredictionFeedback"]):
        """Feedback-table size and initialization."""

        vocab_size: int = -1
        """Token vocabulary; -1 inherits the model's."""

        hidden_size: int = -1
        """Table width; -1 inherits the model's hidden size."""

        init_std: float = 0.0
        """Realized standard deviation; 0 zero-initializes the table."""

        embed_scale: float = -1.0
        """Runtime multiplier; -1 inherits the model's (see
        :class:`FactoredPositions.Config.embed_scale`)."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.hidden_size <= 0 or config.vocab_size <= 0:
            raise ValueError(
                f"vocab_size and hidden_size must be positive; got "
                f"{config.vocab_size} and {config.hidden_size}. Both are "
                "normally inherited from the model during finalize.",
            )
        self.config = config
        self.embed_scale = (
            config.hidden_size**0.5 if config.embed_scale < 0 else config.embed_scale
        )
        self.embed_feedback = _table(
            config.vocab_size,
            hidden_size=config.hidden_size,
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

    class Config(Fig["GridEmbedding"]):
        """Vocabulary, width, and the additive channel list."""

        vocab_size: int = 11
        """Token vocabulary size."""

        hidden_size: int = -1
        """Embedding width; -1 inherits the model's hidden size."""

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
                if isinstance(channel, HasHiddenSize) and channel.hidden_size == -1:
                    channel.hidden_size = self.hidden_size
                if isinstance(channel, HasGridLen) and channel.grid_len == -1:
                    channel.grid_len = self.grid_len
                if isinstance(channel, HasVocabSize) and channel.vocab_size == -1:
                    channel.vocab_size = self.vocab_size
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive; got {config.hidden_size}. It is "
                "normally inherited from the model during finalize.",
            )
        self.config = config
        self.embed_scale = config.hidden_size**0.5
        self.embed_tokens = Embedding.Config(
            channels_out=config.hidden_size,
            num_embeddings=config.vocab_size,
        ).make()
        truncated_normal(
            self.embed_tokens.weight,
            std=1.0 / self.embed_scale,
            depth=-1,
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
            contribution = channel(tokens, embeddings)
            if contribution.numel():
                embeddings = embeddings + contribution
        return embeddings


def _table(
    num_embeddings: int,
    *,
    hidden_size: int,
    init_std: float,
) -> nn.Parameter:
    """A learned table whose REALIZED std is ``init_std`` after rescaling.

    The std passed to the initializer is divided by ``sqrt(hidden_size)``
    because the caller multiplies by that factor at runtime (the embedding
    rescale trick), so the two cancel and the effective std is ``init_std``.
    """
    w = torch.zeros(num_embeddings, hidden_size)
    if init_std > 0:
        truncated_normal(
            w,
            std=init_std / hidden_size**0.5,
            depth=-1,
            variance_correction=True,
        )
    return nn.Parameter(w)
