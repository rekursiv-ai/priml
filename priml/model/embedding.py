"""Embedding layers."""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import Literal, override

from configgle import Fig
from torch import nn

import torch


class Embedding(nn.Embedding):
    """Embedding with truncated normal init."""

    class Config(Fig["Embedding"], kw_only=False):
        channels_out: int = -1
        """Dimensionality of each embedding vector."""

        _: KW_ONLY

        num_embeddings: int = -1
        """Size of the embedding vocabulary."""

        padding_idx: int | None = None
        """Index whose embedding is zeroed out (e.g. for padding tokens)."""

        device: torch.device | str | None = None
        """Device for parameter allocation."""

        dtype: torch.dtype | None = None
        """Data type for parameters."""

        shard: Literal["none", "colwise", "rowwise", "vocab"] = "none"
        """Tensor-parallel shard style over the mesh tp dim; none = replicated."""

    def __init__(self, config: Config) -> None:
        self.shard = config.shard
        super().__init__(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.channels_out,
            padding_idx=config.padding_idx,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
