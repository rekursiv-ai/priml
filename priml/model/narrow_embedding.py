"""A lookup table held at a narrower width than it was drawn at.

A lookup is a gather, not an arithmetic reduction, so half precision costs a
table nothing it was using -- while halving the largest tensors in a language
model and the traffic to read them.

The two steps are one decision, which is why they are one module: the draws
happen at full precision and the table is cast AFTER, because sampling straight
into bfloat16 quantizes every value to its ~3 significant digits and changes the
table's spread (measured: std 0.0189 against 0.0203 at ``std=0.02``). A caller
setting ``Embedding.Config.dtype`` gets the quantized table instead, so the
ordering has to belong to something that owns both steps.

A narrowed table makes the model runnable ONLY under autocast, since the
half-precision stream it emits meets a float32 projection one layer later. That
is a property of the training recipe, so the experiment declares it by filling
this slot rather than the architecture assuming it.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Any, Self, override

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.model.custom_types import (
    ChannelsOut,
    LookupTable,
    propagate_attr,
)
from priml.model.embedding import Embedding


class NarrowEmbedding(nn.Module):
    """A table drawn at full precision, then held at ``dtype``."""

    class Config(Fig["NarrowEmbedding"], kw_only=False):
        dtype: torch.dtype | None = None
        """Width the table is held at; None keeps whatever ``inner`` drew."""

        _: KW_ONLY

        inner: Makeable[LookupTable] = field(default_factory=Embedding.Config)
        """The table being narrowed."""

        channels_out: int = -1
        """Embedding width; forwarded to ``inner``."""

        num_embeddings: int = -1
        """Vocabulary size; forwarded to ``inner``."""

        @override
        def finalize(self) -> Self:
            propagate_attr(
                self.inner,
                "channels_out",
                self.channels_out,
                protocol=ChannelsOut,
            )
            propagate_attr(self.inner, "num_embeddings", self.num_embeddings)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.dtype = config.dtype
        self.inner = config.inner.make()
        self._narrow()

    def reset_parameters(self) -> None:
        """Redraw at full precision, then narrow again.

        Both halves, in order: a reset that left the table narrow would draw
        into bfloat16 -- the very quantization this module exists to avoid --
        and meta-device materialization drives init through here alone.
        """
        if self.dtype is not None:
            self.inner.to(dtype=torch.float32)
        self.inner.reset_parameters()
        self._narrow()

    def _narrow(self) -> None:
        """Cast the built table, once its draws are done."""
        if self.dtype is not None:
            self.inner.to(dtype=self.dtype)

    @override
    def forward(self, tokens: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        return self.inner(tokens)
