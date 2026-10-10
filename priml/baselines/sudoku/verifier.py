"""Learned Sudoku solution verifier: the net and the committee lock.

The verifier classifies "is this candidate grid THE solution to this puzzle?"
from the (puzzle, candidate) token rows alone -- no solver internals (no
q-halt, no latents, no generator identity) and no programmatic Sudoku-rule
oracle at inference. Labels come from ground-truth solutions offline:
``label = (candidate == solution)``.

* :class:`SudokuVerifier` -- a tiny transformer (128 channels, 4 layers, 4 heads,
  ~0.8M params) whose attention is masked to the 27-group Sudoku constraint
  graph: each cell attends only to the 20 peers sharing its row, column, or
  box. The structural bias -- not a rule oracle; the consistency predicate
  itself is learned -- is what buys perfect dev separation at threshold 0.
* :class:`VerifierAcceptor` -- the frozen committee lock: min over member
  grid logits must STRICTLY exceed the threshold (default 0, unanimity).

Its training stream is :mod:`~priml.baselines.sudoku.verifier_data`; its
dedicated fit, which writes the member checkpoints this lock reads, is
:mod:`~priml.baselines.sudoku.verifier_fit`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from functools import partial
from pathlib import Path
from typing import Protocol, override

import math

from configgle import Fig, Makes
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.sudoku.puzzle_spec import SudokuSpec, sudoku_group_indices
from priml.baselines.sudoku.runner import checkpoint_payload
from priml.cost import Cost, cost, elementwise_cost, reduction_cost
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.model.norm import LayerNorm
from priml.paths import resolve_working_dir
from priml.runtime import get_device


def group_attention_mask() -> Tensor:
    """Boolean ``[81, 81]`` mask; True where two cells share a group."""
    member = torch.zeros(27, 81, dtype=torch.bool)
    member.scatter_(1, sudoku_group_indices(SudokuSpec()), True)
    return (member.float().T @ member.float()) > 0


def cell_positions() -> Tensor:
    """Return one learned-position index per cell (the recipe's flat table).

    Returns:
      index: ``[81, 1]`` long table indices.

    """
    return torch.arange(81).unsqueeze(-1)


def row_column_positions() -> Tensor:
    """Return each cell's row coordinate and column coordinate.

    Generic 2D grid coordinates (rows ``0..8``, columns ``9..17`` of one
    18-entry table) with deliberately NO box coordinate: boxes are Sudoku
    structure, and the prior-free verifier must discover them.

    Returns:
      index: ``[81, 2]`` long table indices.

    """
    cell = torch.arange(81)
    return torch.stack([cell // 9, 9 + cell % 9], dim=-1)


class SudokuVerifier(nn.Module):
    """Tiny transformer verifier over 81 (puzzle, candidate) cell tokens.

    Output is ``[B, 82]``: column 0 is the grid logit (positive = candidate
    IS the solution), columns 1: are per-cell wrongness logits (positive =
    this cell differs from the solution), used as dense auxiliary
    supervision.
    """

    class Config(Fig["SudokuVerifier"]):
        """Architecture of the verifier."""

        channels_hidden: int = 128
        """Embedding and attention width."""

        num_layers: int = 4
        """Number of transformer blocks."""

        heads: int = 4
        """Attention heads per block."""

        attention_mask: Callable[[], Tensor] | None = group_attention_mask
        """``[81, 81]`` boolean attention mask builder. The recipe restricts
        attention to cells sharing a row, column, or box (the 27-group
        incidence graph) -- a structural bias, not a rule oracle. None lets
        every cell attend everywhere (without a denoising objective, the
        measured control: AUROC > 0.97 but no usable zero-false-accept
        operating point)."""

        position_index: Callable[[], Tensor] = cell_positions
        """``[81, P]`` per-cell indices into one learned position table; a
        cell's position embedding is the sum of its ``P`` entries."""

        vocab_size: int = 11
        """Token vocabulary: 0=pad, 1=blank, 2-10 = digits 1-9."""

        def cost(self, *, batch_size: int, dtype: torch.dtype | None) -> Cost:
            """Cost one scoring forward over 81-cell rows, and its backward.

            Args:
              batch_size: (puzzle, candidate) rows in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            width = self.channels_hidden
            cells = 81
            rows = cells * batch_size
            over_cells = partial(
                cost,
                seq_len=cells,
                batch_size=batch_size,
                dtype=dtype,
            )
            over_rows = partial(cost, seq_len=1, batch_size=batch_size, dtype=dtype)
            norm = LayerNorm.Config(width, elementwise_affine=True)
            linear = partial(Linear.Config, bias=True)
            add = elementwise_cost(
                primal=width * rows,
                adjoint=width * rows,
                channels=width,
                inputs=2,
                rows=rows,
                dtype=dtype,
            )
            block = (
                over_cells(norm).tile(2, copies=2)
                + over_cells(linear(width, 3 * width))
                + attention_kernel_cost(
                    seq_len=cells,
                    batch_size=batch_size,
                    dtype=dtype,
                    num_heads=self.heads,
                    channels_head=width // self.heads,
                )
                + over_cells(linear(width, width))
                + over_cells(linear(width, 4 * width))
                + _gelu_cost(channels=4 * width, rows=rows, dtype=dtype)
                + over_cells(linear(4 * width, width))
                + add.tile(2)
            )
            index = self.position_index()
            lookups = index.shape[-1]
            tables = (
                over_cells(Embedding.Config(self.vocab_size, width)).tile(2, copies=2)
                + over_cells(Embedding.Config(int(index.max()) + 1, width)).tile(
                    lookups,
                )
                + add.tile(1 + lookups)
            )
            heads = (
                over_cells(norm)
                + over_cells(linear(width, 1))
                + reduction_cost(
                    input_elements=rows * width,
                    output_groups=batch_size * width,
                    dtype=dtype,
                )
                + over_rows(linear(width, width))
                + _gelu_cost(channels=width, rows=batch_size, dtype=dtype)
                + over_rows(linear(width, 1))
            )
            return tables + block.tile(self.num_layers, copies=self.num_layers) + heads

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.channels_hidden % config.heads != 0:
            raise ValueError(
                f"channels_hidden {config.channels_hidden} must divide by heads "
                f"{config.heads}.",
            )
        self.config = config
        width = config.channels_hidden
        self.candidate_embedding = nn.Embedding(config.vocab_size, width)
        self.puzzle_embedding = nn.Embedding(config.vocab_size, width)
        index = config.position_index()
        self.position_embedding = nn.Embedding(int(index.max()) + 1, width)
        self.blocks = nn.ModuleList(
            _VerifierBlock(width=width, heads=config.heads)
            for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(width)
        self.cell_head = nn.Linear(width, 1)
        self.grid_head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.position_index = nn.Buffer(index, persistent=False)
        mask = config.attention_mask
        self.register_buffer(
            "attention_mask",
            mask() if mask is not None else None,
            persistent=False,
        )
        self.attention_mask: Tensor | None

    @override
    def forward(self, puzzle: Tensor, candidate: Tensor) -> Tensor:
        """Score a batch of (puzzle, candidate) rows.

        Args:
          puzzle: ``[B, 81]`` original puzzle tokens (1=blank, 2-10 digits).
          candidate: ``[B, 81]`` complete candidate grid tokens.

        Returns:
          logits: ``[B, 82]``; column 0 the grid logit, columns 1: per-cell
            wrongness logits.

        """
        x = self.final_norm(self.trunk(puzzle, candidate))
        cell_logits = self.cell_head(x).squeeze(-1)
        grid_logit = self.grid_head(x.mean(dim=-2)).squeeze(-1)
        return torch.cat([grid_logit.unsqueeze(-1).float(), cell_logits.float()], -1)

    def trunk(self, puzzle: Tensor, candidate: Tensor) -> Tensor:
        """Embed the rows and run every block; the final norm is the caller's.

        Args:
          puzzle: ``[B, 81]`` original puzzle tokens.
          candidate: ``[B, 81]`` candidate grid tokens.

        Returns:
          latents: ``[B, 81, width]`` cell latents.

        """
        x = (
            self.candidate_embedding(candidate.long())
            + self.position_embedding(self.position_index).sum(dim=-2)
            + self.puzzle_embedding(puzzle.long())
        )
        for block in self.blocks:
            x = block(x, self.attention_mask)
        return x


class DenoisingVerifier(SudokuVerifier):
    """A verifier that also learns to fill masked cells of its candidate.

    Training forwards add a masked-digit denoising task: random non-given
    candidate cells are blanked and a 9-way head predicts the digit there.
    The Bayes-optimal predictor of a masked digit must consult exactly the
    cell's constraint peers, so the task teaches the attention routing that
    the verification loss alone rewards only weakly -- with no mask, the
    verifier discovers the constraint graph. Labels are the data itself.

    In training mode the output gains ``81 * 9`` digit logits and ``81``
    targets (the masked cell's digit class, -1 elsewhere) after the ``[B,
    82]`` verification columns, for :class:`DenoisingLoss`. In eval mode the
    output is the plain ``[B, 82]``, so a committee scores it unchanged.
    """

    class Config(Makes["DenoisingVerifier"], SudokuVerifier.Config):
        """The verifier plus the masked-cell fraction of its training task."""

        denoise_fraction: float = 0.25
        """Fraction of non-given candidate cells blanked per training row."""

    config: Config
    """Narrowed from the parent's declaration for the denoising fraction."""

    def __init__(self, config: Config) -> None:
        fraction = config.denoise_fraction
        if math.isnan(fraction) or fraction <= 0 or fraction >= 1:
            raise ValueError(
                f"denoise_fraction must be in (0, 1), got {config.denoise_fraction}.",
            )
        super().__init__(config)
        self.denoise_head = nn.Linear(config.channels_hidden, 9)

    @override
    def forward(self, puzzle: Tensor, candidate: Tensor) -> Tensor:
        """Score rows; in training, append the denoising logits and targets.

        Args:
          puzzle: ``[B, 81]`` original puzzle tokens.
          candidate: ``[B, 81]`` complete candidate grid tokens.

        Returns:
          logits: ``[B, 82]`` in eval mode; ``[B, 82 + 81 * 9 + 81]`` in
            training mode (see the class docstring).

        """
        output = super().forward(puzzle, candidate)
        if not self.training:
            return output
        # Drawn from the ambient stream, as dropout is: the fit seeds it.
        masked = (
            torch.rand(candidate.shape, device=candidate.device)
            < self.config.denoise_fraction
        ) & (puzzle < 2)
        blanked = torch.where(masked, torch.ones_like(candidate), candidate)
        digits = self.denoise_head(self.final_norm(self.trunk(puzzle, blanked)))
        target = torch.where(masked, (candidate.long() - 2).clamp(0, 8), -1)
        return torch.cat(
            [output, digits.reshape(len(output), -1).float(), target.float()],
            dim=-1,
        )


class CommitteeLock(Protocol):
    """What the sieve needs from the committee lock.

    :class:`VerifierAcceptor` inherits this protocol explicitly (nominal
    conformance) so its ``Config`` satisfies ``Makeable[CommitteeLock]``
    under checkers without structural ``type[cls]`` -> ``type[Protocol]``
    support; test stubs conform structurally.
    """

    def __call__(self, preds: Tensor, media: Tensor) -> Tensor:
        """``[N]`` bool acceptance of decoded grids against original boards."""
        ...


class VerifierAcceptor(CommitteeLock):
    """Frozen verifier committee scoring (puzzle, candidate) rows."""

    class Config(Fig["VerifierAcceptor"]):
        """Committee membership and the frozen operating rule."""

        model: SudokuVerifier.Config = field(default_factory=SudokuVerifier.Config)
        """Shared architecture of every committee member."""

        base_dir: Path | str | None = None
        """Scratch root; ``None`` resolves beneath ``/opt/scratch`` at
        construction."""

        checkpoint_paths: tuple[str | Path, ...] = ()
        """Member checkpoints, required: the fits of a chain rung. The
        :class:`VerifierFit` layout
        (``state["step"]["model"]``) or the HF model-only layout (top-level
        ``state["model"]``). A relative logical path resolves beneath
        ``base_dir``. Acceptance is UNANIMOUS: the minimum member grid logit
        must STRICTLY exceed ``threshold``."""

        threshold: float = 0.0
        """Frozen accept threshold on the minimum member grid logit. The
        comparison is strict: a logit exactly at the threshold is rejected."""

        max_rows: int = 8_192
        """Scoring chunk width (memory guard)."""

        device: torch.device | str | None = None
        """Device for the committee."""

    def __init__(self, config: Config) -> None:
        if not config.checkpoint_paths:
            raise ValueError("the verifier committee needs >= 1 checkpoint.")
        if config.max_rows < 1:
            raise ValueError(f"max_rows must be >= 1, got {config.max_rows}.")
        self.config = config
        device = get_device(config.device)
        base = config.base_dir if config.base_dir is not None else Path("/opt/scratch")
        self.members: list[SudokuVerifier] = []
        for path in config.checkpoint_paths:
            member = config.model.make().to(device)
            # A ``Path`` is a literal override; a ``str`` is a logical path
            # resolved beneath the scratch base.
            resolved = (
                path if isinstance(path, Path) else resolve_working_dir(base, path)
            )
            # HF model-only files carry {"model": ...} unnested; the
            # VerifierFit layout nests it under "step".
            member.load_state_dict(checkpoint_payload(resolved, device)["model"])
            member.eval()
            member.requires_grad_(False)
            self.members.append(member)

    @override
    def __call__(self, preds: Tensor, media: Tensor) -> Tensor:
        """Unanimous learned acceptance of decoded grids.

        Args:
          preds: ``[N, G]`` decoded candidate grids.
          media: ``[N, G]`` the ORIGINAL puzzle boards (never pinned
            boards -- pins are hypotheses, exactly like the rule path).

        Returns:
          accepted: ``[N]`` bool mask, True iff EVERY member's grid logit
            exceeds the frozen threshold.

        """
        return self.scores(preds, media) > self.config.threshold

    def scores(self, preds: Tensor, media: Tensor) -> Tensor:
        """Minimum member grid logit per row (the committee score).

        Args:
          preds: ``[N, G]`` decoded candidate grids.
          media: ``[N, G]`` the original puzzle boards.

        Returns:
          scores: ``[N]`` fp32 minimum member grid logits.

        """
        chunks: list[Tensor] = []
        with torch.inference_mode():
            for lo in range(0, preds.shape[0], self.config.max_rows):
                puzzle = media[lo : lo + self.config.max_rows]
                candidate = preds[lo : lo + self.config.max_rows]
                member_scores = torch.stack(
                    [
                        member(puzzle, candidate)[:, 0].float()
                        for member in self.members
                    ],
                )
                chunks.append(member_scores.min(dim=0).values)
        return torch.cat(chunks) if chunks else preds.new_zeros(0, dtype=torch.float32)


def _gelu_cost(*, channels: int, rows: int, dtype: torch.dtype | None) -> Cost:
    """Cost exact GELU over ``rows`` rows of ``channels``: erf forward, pdf back."""
    return elementwise_cost(
        primal=8 * channels * rows,
        adjoint=8 * channels * rows,
        channels=channels,
        rows=rows,
        dtype=dtype,
    )


class _VerifierBlock(nn.Module):
    """Pre-norm attention + MLP block under an optional cell attention mask."""

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    @override
    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        batch, cells, width = x.shape
        qkv = self.qkv(self.norm_attention(x))
        q, k, v = qkv.reshape(batch, cells, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        attended = functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
        )
        attended = attended.transpose(1, 2).reshape(batch, cells, width)
        x = x + self.proj(attended)
        return x + self.mlp(self.norm_mlp(x))
