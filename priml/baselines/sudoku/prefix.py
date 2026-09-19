"""Prefix tokens prepended to a puzzle's grid embedding.

The grid says what THIS puzzle looks like; a prefix says what task it belongs
to, and gives the halt readout a token of its own that no grid cell has to
share. Both are tokens the solver sees before the grid, so they fill one slot.

Two kinds compose:

* :class:`RegisterTokens` -- a few learned vectors shared by every puzzle.
  Scratch space, and a stable position for a readout.
* :class:`SparsePuzzleEmbedding` -- one learned vector PER PUZZLE. A benchmark
  where each task is a distinct rule, seen many times under augmentation,
  can learn a per-task vector; the table is large (ARC-AGI: ~876k rows) so it
  is trained sparsely, only the rows a batch touches.

Sparse training is why this is not a plain ``nn.Embedding``. Backpropagating
into an 876k-row table would allocate a gradient the size of the table every
step; instead the forward copies the batch's rows into a small buffer that
carries the gradient, and the optimizer scatters those few rows back.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import TYPE_CHECKING, Protocol, Self, override

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    traffic,
)
from priml.model.custom_types import ChannelsOut
from priml.model.init import truncated_normal


if TYPE_CHECKING:
    from collections.abc import Callable


class PrefixConfig(Makeable[nn.Module], Protocol):
    """A config that builds a prefix module and declares its token count.

    The solver sizes its latent sequence from this number before anything is
    built, so a prefix that cannot say how many tokens it prepends cannot fill
    the slot -- or be stacked.
    """

    @property
    def num_tokens(self) -> int:
        """Tokens this prefix prepends to every puzzle."""
        ...


class RegisterTokens(nn.Module):
    """Learned vectors prepended to every puzzle, identical across the batch.

    Gives the model scratch positions that carry no grid cell, and a fixed
    place to read a per-puzzle scalar from -- the halt head reads position 0,
    so with a prefix that position is never a cell whose value the head would
    otherwise have to share.
    """

    class Config(Fig["RegisterTokens"]):
        """How many tokens, how wide, and how strongly initialized."""

        num_tokens: int = 1
        """Tokens prepended. 0 disables the module."""

        channels_out: int = -1
        """Token width; -1 inherits the model's hidden size."""

        init_std: float = 1.0
        """Realized standard deviation of the learned tokens."""

        learnable: bool = True
        """Train the tokens. False freezes them as a fixed random basis."""

        embed_scale: float = -1.0
        """Runtime multiplier; -1 derives it from ``channels_out``.

        Must match the grid embedding's, or the prefix enters the sequence at
        a different magnitude than the tokens it precedes."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one scale over every register token in the complete batch.

            The expand is a view. A frozen basis owns nothing, so its gradient
            reduction over puzzles vanishes with its parameters.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-batch cost of this module.

            """
            del seq_len, kwargs
            width = self.num_tokens * self.channels_out
            rows = batch_size
            return elementwise_cost(
                primal=width * rows,
                adjoint=width * rows,
                channels=width,
                params=width if self.learnable else 0,
                rows=rows,
                inputs=0 if self.learnable else 1,
                adjoint_inputs=1,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.channels_out <= 0:
            raise ValueError(
                f"channels_out must be positive; got {config.channels_out}. It is "
                "normally inherited from the model during finalize.",
            )
        self.config = config
        self.embed_scale = (
            config.channels_out**0.5 if config.embed_scale < 0 else config.embed_scale
        )
        tokens = torch.empty(config.num_tokens, config.channels_out)
        truncated_normal(
            tokens,
            std=config.init_std / self.embed_scale,
            depth_index=(),
            variance_correction=True,
        )
        self.register_tokens = (
            nn.Parameter(tokens)
            if config.learnable
            else nn.Buffer(tokens, persistent=True)
        )

    @override
    def forward(self, batch_size: int, **kwargs: object) -> Tensor:
        """Return ``[B, num_tokens, C]``, the same tokens for every row."""
        del kwargs
        tokens: Tensor = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        return self.embed_scale * tokens


class SparsePuzzleEmbedding(nn.Module):
    """One learned vector per puzzle, trained only on the rows a batch uses.

    The table is far too large to backpropagate densely -- ARC-AGI has ~876k
    puzzles, so a dense gradient would be gigabytes per step. Instead the master
    table is a buffer (no autograd), the forward copies the batch's rows into a
    small ``local_weights`` buffer that DOES carry gradient, and a sparse
    optimizer scatters those rows back afterwards. Evaluation reads the master
    table directly, since nothing needs a gradient there.

    Zero-initialized by default, so a fresh model behaves as though the prefix
    were absent and must learn to use it.
    """

    class Config(Fig["SparsePuzzleEmbedding"], kw_only=False):
        """Cost size, prefix width, and the per-batch gradient buffer."""

        channels_in: int = -1
        """Model width; -1 inherits it. Sets the reshaped token width."""

        channels_out: int = -1
        """Per-puzzle vector width; -1 inherits the model's hidden size."""

        _: KW_ONLY

        num_puzzles: int = 1
        """Rows in the table: one per distinct puzzle in the dataset.

        Must match the dataset's identifier count, or an identifier indexes off
        the end of the table -- the dataset verifies this at build time."""

        num_tokens: int = 16
        """Prefix tokens this embedding contributes.

        The per-puzzle vector is reshaped into this many tokens, so a wider
        prefix gives a task more room without a wider model."""

        batch_size: int = 384
        """Rows in the gradient buffer: the training batch size.

        Sized once at construction because the buffer is preallocated; a batch
        larger than this cannot be scattered back."""

        init_std: float = 0.0
        """Realized standard deviation; 0 zero-initializes the table."""

        embed_scale: float = -1.0
        """Runtime multiplier; -1 derives it from ``channels_out``."""

        dtype: torch.dtype | None = None
        """Cast applied to the forward output; ``None`` keeps the table dtype."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one row gathered and the prefix scaled for the batch; nothing owned.

            The table is a buffer, so there are no parameters, and the scatter
            back into it is the sparse optimizer's work, which the counting
            policy excludes: the adjoint holds only the scale. The pad is a
            zero fill and the reshape a view. The puzzle id is one ``int64``
            read, plus two more for the gather's bookkeeping.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-batch cost of this module.

            """
            del seq_len, kwargs
            width = self.num_tokens * self.channels_out
            copied = 4 * self.channels_out
            if width > self.channels_out:
                copied += self.channels_out + width
            rows = batch_size
            return (
                traffic(
                    "primal",
                    "selection",
                    elements=3 * rows,
                    dtype=torch.int64,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=copied * rows,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=width * rows,
                    adjoint=width * rows,
                    channels=width,
                    adjoint_inputs=1,
                    rows=rows,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.channels_out <= 0:
            raise ValueError(
                "channels_out must be positive; it is normally inherited "
                f"from the model during finalize. Got {config.channels_out}.",
            )
        self.config = config
        self.embed_scale = (
            config.channels_out**0.5 if config.embed_scale < 0 else config.embed_scale
        )
        table = torch.zeros(config.num_puzzles, config.channels_out)
        if config.init_std > 0:
            truncated_normal(
                table,
                std=config.init_std,
                depth_index=(),
                variance_correction=True,
            )
        # A buffer, not a Parameter: autograd must not build a dense gradient
        # for a table this size. The sparse optimizer writes it directly.
        self.weights = nn.Buffer(table, persistent=True)
        self.local_weights = nn.Buffer(
            torch.zeros(config.batch_size, config.channels_out, requires_grad=True),
            persistent=False,
        )
        self.local_ids = nn.Buffer(
            torch.zeros(config.batch_size, dtype=torch.int32),
            persistent=False,
        )

    @override
    def forward(self, batch_size: int, **kwargs: object) -> Tensor:
        """Return ``[B, num_tokens, hidden]`` for this batch's puzzle ids.

        Args:
          batch_size: Rows in the batch.
          **kwargs: Must carry ``puzzle_identifiers``, ``[B]`` integer ids.

        Returns:
          prefix: The per-puzzle vectors, reshaped into prefix tokens.

        Raises:
          TypeError: If ``puzzle_identifiers`` is absent from the batch.

        """
        identifiers = kwargs.get("puzzle_identifiers")
        if not isinstance(identifiers, Tensor):
            raise TypeError(
                "SparsePuzzleEmbedding requires puzzle_identifiers in the batch; "
                f"got {type(identifiers).__name__}.",
            )
        vectors = self._lookup(identifiers)
        config = self.config
        width = config.num_tokens * config.channels_out
        if vectors.shape[-1] < width:
            vectors = nn.functional.pad(vectors, (0, width - vectors.shape[-1]))
        prefix: Tensor = vectors.reshape(
            batch_size,
            config.num_tokens,
            config.channels_out,
        )
        return self.embed_scale * prefix

    def _lookup(self, identifiers: Tensor) -> Tensor:
        """Read the batch's rows, via the gradient buffer when training."""
        config = self.config
        if not self.training:
            rows = self.weights[identifiers.to(torch.long)]
            return rows if config.dtype is None else rows.to(config.dtype)
        with torch.no_grad():
            self.local_weights.copy_(self.weights[identifiers.to(torch.long)])
            self.local_ids.copy_(identifiers.to(torch.int32))
        return (
            self.local_weights
            if config.dtype is None
            else self.local_weights.to(config.dtype)
        )

    # ``nn.Module._apply`` replaces buffers with transformed copies, and a copy does not
    # carry ``requires_grad`` -- so without this the local buffer stops receiving
    # gradients the moment the model moves to a GPU, and the table silently never
    # trains.
    @override
    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        """Re-establish the gradient buffer after a device or dtype move."""
        module = super()._apply(fn, recurse=recurse)
        self.local_weights = nn.Buffer(
            self.local_weights.detach().requires_grad_(True),
            persistent=False,
        )
        self.local_ids = nn.Buffer(self.local_ids.detach(), persistent=False)
        return module


class PrefixStack(nn.Module):
    """Concatenates several prefix modules into one ``[B, P, C]`` block.

    The solver holds a single prefix slot, but ARC needs both a per-puzzle
    embedding and register tokens. Order is a numerics contract only in that it
    fixes which position the halt readout lands on: the readout reads position
    0, so whichever module comes first owns it.
    """

    class Config(Fig["PrefixStack"]):
        """The prefix modules, in sequence order."""

        parts: list[PrefixConfig] = field(default_factory=list["PrefixConfig"])
        """Modules whose outputs are concatenated along the token axis."""

        channels_out: int = -1
        """Token width; -1 inherits the model's.

        Declared even though this module owns no weights: the model propagates
        the width to whatever fills its prefix slot, and without the field the
        stack is skipped and its parts are built at the sentinel."""

        @property
        def num_tokens(self) -> int:
            """Tokens the stack prepends: every part's, in order."""
            return sum(part.num_tokens for part in self.parts)

        @override
        def finalize(self) -> Self:
            for part in self.parts:
                if isinstance(part, ChannelsOut) and part.channels_out == -1:
                    part.channels_out = self.channels_out
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Sum every part and the concatenation's input/output copy.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-batch cost of this module.

            """
            children = sum(
                (
                    cost(
                        part,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        dtype=dtype,
                        **kwargs,
                    )
                    for part in self.parts
                ),
                Cost(),
            )
            return children + traffic(
                "primal",
                "selection",
                elements=2 * batch_size * self.num_tokens * self.channels_out,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        built: list[nn.Module] = []
        for part in config.parts:
            module = part.make()
            assert isinstance(module, nn.Module), type(module).__name__
            built.append(module)
        self.parts = nn.ModuleList(built)

    @override
    def forward(self, batch_size: int, **kwargs: object) -> Tensor:
        """Concatenate every part's tokens along the sequence axis."""
        pieces = [part(batch_size, **kwargs) for part in self.parts]
        return torch.cat(pieces, dim=1)
