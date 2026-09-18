"""Transformer stack with optional input and output projections.

Attention configs and masks determine causality; the wrapper does not impose it.
Without projections, inputs and outputs are continuous hidden states. For a
language model, inject an ``Embedding`` as ``proj_in`` and a head as
``proj_out``. The head owns its final normalization; the stack applies none of
its own.

``block`` accepts a template broadcast over ``num_layers`` or an explicit
per-layer list. Each layer receives its own depth index.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, cast, override

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    cost,
)
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsInOutConfig,
    ChannelsOut,
    HasDepthIndex,
    HasResetParameters,
    TensorModule,
    propagate_attr,
)
from priml.model.legacy_keys import absorb_legacy_keys
from priml.model.sequential import Sequential
from priml.model.special import TiedLinear
from priml.model.transformer.block import TransformerBlock


class Transformer(nn.Module):
    """Optional projections around a configurable transformer stack."""

    class Config(Fig["Transformer"], kw_only=False):
        channels_in: int = -1
        """Input feature width, or embedding width for integer-token inputs."""

        channels_out: int = -1
        """Output width after ``proj_out``.

        Inferred from an ``proj_out`` that states one, else the hidden width.
        A head that cannot state one (a borrowed weight) takes it from here,
        so a language model with a tied head sets its vocabulary here."""

        _: KW_ONLY

        num_layers: int = -1
        """Number of stacked transformer blocks."""

        proj_in: Makeable[TensorModule] | None = None
        """Input projection or embedding; None accepts hidden states directly."""

        block: ChannelsInOutConfig | list[ChannelsInOutConfig] = field(
            default_factory=TransformerBlock.Config,
        )
        """Block template or explicit per-layer list of length num_layers."""

        proj_out: Makeable[TensorModule] | None = None
        """Head applied after the last block; None returns hidden states.

        Owns its normalization: a bare norm, a projection, or a norm composed
        in front of a projection."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                if isinstance(self.proj_in, ChannelsIn):
                    self.channels_in = self.proj_in.channels_in
                elif isinstance(self.proj_in, ChannelsOut):
                    self.channels_in = self.proj_in.channels_out
                if self.channels_in == -1:
                    self.channels_in = self.channels_out
            if self.num_layers == -1 and isinstance(self.block, list):
                self.num_layers = len(self.block)
            if isinstance(self.proj_in, ChannelsIn) and self.proj_in.channels_in == -1:
                self.proj_in.channels_in = self.channels_in
            if (
                isinstance(self.proj_in, ChannelsOut)
                and self.proj_in.channels_out == -1
            ):
                self.proj_in.channels_out = self.channels_in
            channels_hidden = (
                self.proj_in.channels_out
                if isinstance(self.proj_in, ChannelsOut)
                else self.channels_in
            )
            templates = (
                self.block
                if isinstance(self.block, list)
                else [self.block] * max(self.num_layers, 0)
            )
            block_configs: list[ChannelsInOutConfig] = []
            for index, template in enumerate(templates):
                config = template.copy_tree()
                if isinstance(config, HasDepthIndex):
                    config.depth_index = ((index, self.num_layers),)
                block_configs.append(config)
            self.block = block_configs
            for config in block_configs:
                if config.channels_in == -1:
                    propagate_attr(
                        config,
                        "channels_in",
                        channels_hidden,
                        protocol=ChannelsIn,
                    )
                if config.channels_out == -1:
                    propagate_attr(
                        config,
                        "channels_out",
                        channels_hidden,
                        protocol=ChannelsOut,
                    )
            if (
                isinstance(self.proj_out, ChannelsIn)
                and self.proj_out.channels_in == -1
            ):
                self.proj_out.channels_in = channels_hidden
            if (
                self.channels_out != -1
                and isinstance(self.proj_out, ChannelsOut)
                and self.proj_out.channels_out == -1
            ):
                self.proj_out.channels_out = self.channels_out
            finalized = super().finalize()
            # After the cascade: a composed head derives its output width from
            # its last element inside its own finalize, so reading it earlier
            # yields -1 and the stack would report the hidden width.
            if finalized.channels_out == -1:
                finalized.channels_out = (
                    finalized.proj_out.channels_out
                    if isinstance(finalized.proj_out, ChannelsOut)
                    and finalized.proj_out.channels_out != -1
                    else channels_hidden
                )
            return finalized

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Sum the projections and every block of the finalized list.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            assert isinstance(self.block, list)
            projections = [p for p in (self.proj_in, self.proj_out) if p is not None]
            return sum(
                (
                    cost(
                        c,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        dtype=dtype,
                        **kwargs,
                    )
                    for c in (*projections, *self.block)
                ),
                Cost(),
            )

    def __init__(self, config: Config) -> None:
        if not isinstance(config.block, list):
            raise TypeError("A finalized Transformer config must contain a block list.")
        if len(config.block) != config.num_layers:
            raise ValueError(
                f"block list length {len(config.block)} != num_layers={config.num_layers}.",
            )
        super().__init__()
        self.channels_in = config.channels_in
        self.channels_out = config.channels_out
        self.num_layers = config.num_layers
        self.proj_in = config.proj_in.make() if config.proj_in is not None else None
        blocks: list[nn.Module] = []
        for block_config in config.block:
            block = block_config.make()
            if not isinstance(block, nn.Module):
                raise TypeError("A Transformer block config must build an nn.Module.")
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.proj_out = config.proj_out.make() if config.proj_out is not None else None
        # Absorb checkpoints minted before the ``proj_in`` / ``proj_out`` rename.
        absorb_legacy_keys(self, {"in_proj": "proj_in", "out_proj": "proj_out"})

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        for module in (self.proj_in, *self.blocks, self.proj_out):
            if isinstance(module, HasResetParameters):
                module.reset_parameters()

    def project_to_logits(self, hidden: Tensor, **kwargs: object) -> Tensor:
        """Apply the head, or return hidden states when absent.

        Args:
          hidden: Hidden.
          **kwargs: Kwargs.

        Returns:
          result: The Tensor.

        """
        return hidden if self.proj_out is None else self.proj_out(hidden, **kwargs)

    @override
    def forward(self, x: Tensor, /, **kwargs: object) -> Tensor:
        """Apply input projection, blocks, and head."""
        if self.proj_in is not None:
            x = self.proj_in(x)
        for block in self.blocks:
            x = cast(Tensor, block(x, **kwargs))
        return self.project_to_logits(x, **kwargs)


def head_is_tied(config: Transformer.Config) -> bool:
    """Return whether the head borrows the embedding, so HF ships no ``lm_head``.

    Args:
      config: A transformer config, finalized or not.

    Returns:
      tied: True when ``proj_out`` is, or ends in, a ``TiedLinear``.

    """
    head = config.proj_out
    if isinstance(head, Sequential.Config):
        elements = head.elements
        head = elements[-1] if isinstance(elements, list) else elements
    return isinstance(head, TiedLinear.Config)
