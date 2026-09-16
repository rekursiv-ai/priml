"""Embedding layers."""

from __future__ import annotations

from dataclasses import KW_ONLY
from functools import partial
from typing import override

from configgle import Fig
from torch import nn

import torch

from priml.model.cost import Bytes, Compute, Cost, Flops
from priml.model.custom_types import DepthIndex, ShardStyle
from priml.model.init import InitFn, call_init, truncated_normal


class Embedding(nn.Embedding):
    """Embedding with truncated normal init."""

    class Config(Fig["Embedding"], kw_only=False):
        channels_in: int = -1
        """Vocabulary size: the input token range the table indexes."""

        channels_out: int = -1
        """Dimensionality of each embedding vector."""

        _: KW_ONLY

        padding_idx: int | None = None
        """Index whose embedding is zeroed out (e.g. for padding tokens)."""

        device: torch.device | str | None = None
        """Device for parameter allocation."""

        dtype: torch.dtype | None = None
        """Data type for parameters."""

        shard: ShardStyle | None = None
        """Tensor-parallel shard style over the mesh tp dim; ``None`` replicates."""

        depth_index: DepthIndex = ()
        """Block depth index for depth-scaled init (-1 = no scaling).

        Present, and forwarded, for the same reason ``Linear`` and ``Conv``
        carry one: every initializer in :mod:`priml.model.init` divides by
        ``sqrt(depth + 1)`` and DEFAULTS that depth to 1, so a table that never
        states one is drawn at 0.707 of the spread it asked for. A lookup table
        has no residual branch to scale down, hence -1 rather than a depth."""

        init_weight: InitFn = partial(truncated_normal, std=0.02)
        """Draws the table.

        A slot rather than a fixed rule, because the right spread is a property
        of what READS the table: one feeding an RMS norm has its scale divided
        out and wants unit variance, while one summed into a residual stream
        does not."""

        def cost(self, **kwargs: object) -> Cost:
            """Price a gather and its scatter-add adjoint.

            Every gradient element is added to a zero-initialized table, so
            repeated indices do not change the count. Padding rows skip it, so
            this is an upper bound when padding is present.

            Args:
              **kwargs: The open message bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del kwargs
            row = self.channels_out
            return Cost(
                primal=Compute(bytes=Bytes(selection=row)),
                adjoint=Compute(flops=Flops(selection=row), bytes=Bytes(selection=row)),
                params=self.channels_in * row,
                params_active=row,
            )

    def __init__(self, config: Config) -> None:
        self.shard = config.shard
        self.depth_index = config.depth_index
        self._init_weight = config.init_weight
        super().__init__(
            num_embeddings=config.channels_in,
            embedding_dim=config.channels_out,
            padding_idx=config.padding_idx,
            device=config.device,
            dtype=config.dtype,
        )

    @override
    def reset_parameters(self) -> None:
        # Depth is PASSED, as every other parameterized module here passes it.
        # Omitting it does not mean "no scaling": it takes the initializer's own
        # default of 1, which divides by sqrt(2) -- a table 0.707 as wide as the
        # one requested, invisible to every shape, name, and dtype check.
        call_init(self._init_weight, self.weight, depth_index=self.depth_index)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
