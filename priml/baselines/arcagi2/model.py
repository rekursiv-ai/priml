"""ARC2 positional attention injected into the shared grid solver."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, override

from configgle.fig import Makes
from torch import Tensor

import torch

from priml.baselines.sudoku.embedding import GridEmbedding
from priml.baselines.sudoku.model import (
    DeepRecurrence,
    GridConfig,
    RecurrenceConfig,
    SudokuNet,
    corrected_fan_in_normal,
)
from priml.baselines.sudoku.prefix import PrefixConfig, SparsePuzzleEmbedding
from priml.model.attention.rope import RoPE
from priml.model.attention.self_attention import SelfAttention
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock


if TYPE_CHECKING:
    from configgle.custom_types import Makeable

    from priml.model.custom_types import TensorModule


class RotaryBlock(TransformerBlock):
    """Apply one-dimensional rotary positions across the prefix and grid."""

    class Config(Makes["RotaryBlock"], TransformerBlock.Config):
        """Transformer block and the rotary position encoding it consumes."""

        attn: Makeable[TensorModule] = field(
            default_factory=lambda: SelfAttention.Config(
                channels_head=64,
                init_weight=corrected_fan_in_normal,
            ),
        )
        """Eight-head attention with variance-corrected fan-in initialization."""

        ffn: Makeable[TensorModule] = field(
            default_factory=lambda: SwiGLU.Config(
                init_weight=corrected_fan_in_normal,
                init_weight_out=corrected_fan_in_normal,
            ),
        )
        """Unnormalized SwiGLU, matching the reference TRM recipe."""

        norm1: Makeable[TensorModule] = field(
            default_factory=lambda: RMSNorm.Config(eps=1e-5),
        )
        """Post-attention normalization."""

        norm2: Makeable[TensorModule] = field(
            default_factory=lambda: RMSNorm.Config(eps=1e-5),
        )
        """Post-MLP normalization."""

        prenorm: bool = False
        """Normalize after each residual addition."""

        rope: RoPE.Config = field(default_factory=RoPE.Config)
        """Rotary encoding, with the same head width as the attention slot."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.rope = config.rope.make()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        """Supply rotary positions to the shared transformer block."""
        kwargs["cos_sin"] = self.rope(torch.arange(x.shape[-2], device=x.device))
        return super().forward(x, **kwargs)


class PuzzleEmbedding(SparsePuzzleEmbedding):
    """Round lookup rows to compute precision, then scale in master precision."""

    class Config(Makes["PuzzleEmbedding"], SparsePuzzleEmbedding.Config):
        """ARC2 sparse lookup precision."""

        dtype: torch.dtype | None = torch.bfloat16
        """Quantization of the gathered row before restoring float32."""

    @override
    def _lookup(self, identifiers: Tensor) -> Tensor:
        return super()._lookup(identifiers).float()


class ArcModelConfig(Makes["SudokuNet"], SudokuNet.Config):
    """Reference TRM assembled from the shared puzzle solver's slots."""

    vocab_size: int = 12
    """Pad, blank, and ten colors."""

    embedding: GridConfig = field(
        default_factory=lambda: GridEmbedding.Config(grid_shape=(900,)),
    )
    """Token embeddings without additional position or feedback channels."""

    block: Makeable[TensorModule] = field(default_factory=RotaryBlock.Config)
    """Post-normalized rotary transformer."""

    recurrence: RecurrenceConfig | None = field(
        default_factory=lambda: DeepRecurrence.Config(slow_cycles=3, fast_cycles=4),
    )
    """Three slow cycles, each refining the fast state four times."""

    prefix: PrefixConfig | None = field(
        default_factory=lambda: PuzzleEmbedding.Config(
            num_puzzles=1_191_727,
            batch_size=256,
        ),
    )
    """Per-task sparse prefix; no register tokens."""
