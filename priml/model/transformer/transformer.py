"""Transformer stack with optional input and output projections.

Attention configs and masks determine causality; the wrapper does not impose it.
Without projections, inputs and outputs are continuous hidden states. For a
language model, inject an ``Embedding`` as ``in_proj`` and a head as
``out_proj``. The head owns its final normalization; the stack applies none of
its own.

``block`` accepts a template broadcast over ``num_layers`` or an explicit
per-layer list. Each layer receives its own depth index.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, cast, override

from configgle import Fig, Makeable
from torch import Tensor, nn

from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    HasDepthIndex,
    Resettable,
    TensorBlockConfig,
    TensorModule,
    propagate_attr,
)
from priml.model.transformer.block import TransformerBlock


class Transformer(nn.Module):
    """Optional projections around a configurable transformer stack."""

    class Config(Fig["Transformer"], kw_only=False):
        channels_in: int = -1
        """Input feature width, or embedding width for integer-token inputs."""

        channels_out: int = -1
        """Output width after ``out_proj``.

        Inferred from an ``out_proj`` that states one, else the hidden width.
        A head that cannot state one (a borrowed weight) takes it from here,
        so a language model with a tied head sets its vocabulary here."""

        _: KW_ONLY

        num_layers: int = -1
        """Number of stacked transformer blocks."""

        in_proj: Makeable[TensorModule] | None = None
        """Input projection or embedding; None accepts hidden states directly."""

        block: TensorBlockConfig | list[TensorBlockConfig] = field(
            default_factory=TransformerBlock.Config,
        )
        """Block template or explicit per-layer list of length num_layers."""

        out_proj: Makeable[TensorModule] | None = None
        """Head applied after the last block; None returns hidden states.

        Owns its normalization: a bare norm, a projection, or a norm composed
        in front of a projection."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                if isinstance(self.in_proj, ChannelsIn):
                    self.channels_in = self.in_proj.channels_in
                elif isinstance(self.in_proj, ChannelsOut):
                    self.channels_in = self.in_proj.channels_out
                if self.channels_in == -1:
                    self.channels_in = self.channels_out
            if self.num_layers == -1 and isinstance(self.block, list):
                self.num_layers = len(self.block)
            if isinstance(self.in_proj, ChannelsIn) and self.in_proj.channels_in == -1:
                self.in_proj.channels_in = self.channels_in
            if (
                isinstance(self.in_proj, ChannelsOut)
                and self.in_proj.channels_out == -1
            ):
                self.in_proj.channels_out = self.channels_in
            channels_hidden = (
                self.in_proj.channels_out
                if isinstance(self.in_proj, ChannelsOut)
                else self.channels_in
            )
            templates = (
                self.block
                if isinstance(self.block, list)
                else [self.block] * max(self.num_layers, 0)
            )
            block_configs: list[TensorBlockConfig] = []
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
                isinstance(self.out_proj, ChannelsIn)
                and self.out_proj.channels_in == -1
            ):
                self.out_proj.channels_in = channels_hidden
            if (
                self.channels_out != -1
                and isinstance(self.out_proj, ChannelsOut)
                and self.out_proj.channels_out == -1
            ):
                self.out_proj.channels_out = self.channels_out
            finalized = super().finalize()
            # After the cascade: a composed head derives its output width from
            # its last element inside its own finalize, so reading it earlier
            # yields -1 and the stack would report the hidden width.
            if finalized.channels_out == -1:
                finalized.channels_out = (
                    finalized.out_proj.channels_out
                    if isinstance(finalized.out_proj, ChannelsOut)
                    and finalized.out_proj.channels_out != -1
                    else channels_hidden
                )
            return finalized

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
        self.in_proj = config.in_proj.make() if config.in_proj is not None else None
        blocks: list[nn.Module] = []
        for block_config in config.block:
            block = block_config.make()
            if not isinstance(block, nn.Module):
                raise TypeError("A Transformer block config must build an nn.Module.")
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.out_proj = config.out_proj.make() if config.out_proj is not None else None

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        for module in (self.in_proj, *self.blocks, self.out_proj):
            if isinstance(module, Resettable):
                module.reset_parameters()

    def project_to_logits(self, hidden: Tensor, **kwargs: object) -> Tensor:
        """Apply the head, or return hidden states when absent.

        Args:
          hidden: Hidden.
          **kwargs: Kwargs.

        Returns:
          result: The Tensor.

        """
        return hidden if self.out_proj is None else self.out_proj(hidden, **kwargs)

    @override
    def forward(self, x: Tensor, /, **kwargs: object) -> Tensor:
        """Apply input projection, blocks, and head."""
        if self.in_proj is not None:
            x = self.in_proj(x)
        for block in self.blocks:
            x = cast(Tensor, block(x, **kwargs))
        return self.project_to_logits(x, **kwargs)
