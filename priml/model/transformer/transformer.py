"""Transformer stack with optional input and output projections.

Attention configs and masks determine causality; the wrapper does not impose it.
Without projections, inputs and outputs are continuous hidden states. For a
language model, inject an Embedding and a vocabulary projection (or "tied").

``block`` accepts a template broadcast over ``num_layers`` or an explicit
per-layer list. Each layer receives its own depth index.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Literal, Self, cast, override

from configgle import Fig, Makeable
from torch import Tensor, nn

from priml.model.custom_types import (
    ChannelsIn,
    ChannelsInOutConfig,
    ChannelsOut,
    HasDepthIndex,
    Resettable,
    TensorBlockConfig,
    TensorModule,
    has_weight,
    propagate_attr,
)
from priml.model.embedding import Embedding
from priml.model.norm import RMSNorm
from priml.model.transformer.block import TransformerBlock


class Transformer(nn.Module):
    """Optional projections around a configurable transformer stack."""

    class Config(Fig["Transformer"], kw_only=False):
        channels_in: int = -1
        """Input feature width, or embedding width for integer-token inputs."""

        channels_out: int = -1
        """Output feature width after the optional output projection."""

        _: KW_ONLY

        vocab_size: int = -1
        """Optional vocabulary size for inferring embedding and output widths."""

        num_layers: int = -1
        """Number of stacked transformer blocks."""

        in_proj: Makeable[TensorModule] | None = None
        """Input projection or embedding; None accepts hidden states directly."""

        block: TensorBlockConfig | list[TensorBlockConfig] = field(
            default_factory=TransformerBlock.Config
        )
        """Block template or explicit per-layer list of length num_layers."""

        final_norm: Makeable[TensorModule] = field(default_factory=RMSNorm.Config)
        """Normalization after the last block."""

        out_proj: ChannelsInOutConfig | Literal["tied"] | None = None
        """Output projection, input-embedding weight transpose, or no projection."""

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
            if (
                isinstance(self.in_proj, Embedding.Config)
                and self.in_proj.num_embeddings == -1
            ):
                self.in_proj.num_embeddings = self.vocab_size
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
                        config, "channels_in", channels_hidden, protocol=ChannelsIn
                    )
                if config.channels_out == -1:
                    propagate_attr(
                        config, "channels_out", channels_hidden, protocol=ChannelsOut
                    )
            if (
                isinstance(self.final_norm, ChannelsIn)
                and self.final_norm.channels_in == -1
            ):
                propagate_attr(
                    self.final_norm, "channels_in", channels_hidden, protocol=ChannelsIn
                )
            if isinstance(self.out_proj, ChannelsInOutConfig):
                if self.out_proj.channels_in == -1:
                    self.out_proj.channels_in = channels_hidden
                if self.out_proj.channels_out == -1:
                    self.out_proj.channels_out = (
                        self.channels_out
                        if self.channels_out != -1
                        else self.vocab_size
                        if self.vocab_size != -1
                        else channels_hidden
                    )
                if self.channels_out == -1:
                    self.channels_out = self.out_proj.channels_out
            elif self.channels_out == -1:
                self.channels_out = (
                    self.in_proj.num_embeddings
                    if self.out_proj == "tied"
                    and isinstance(self.in_proj, Embedding.Config)
                    else channels_hidden
                )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if (
            config.vocab_size != -1
            and isinstance(config.out_proj, ChannelsOut)
            and config.out_proj.channels_out != config.vocab_size
        ):
            raise ValueError(
                f"out_proj.channels_out={config.out_proj.channels_out} must equal "
                f"vocab_size={config.vocab_size}."
            )
        if not isinstance(config.block, list):
            raise TypeError("A finalized Transformer config must contain a block list.")
        if len(config.block) != config.num_layers:
            raise ValueError(
                f"block list length {len(config.block)} != num_layers={config.num_layers}."
            )
        super().__init__()
        self.vocab_size = config.vocab_size
        self.channels_in = config.channels_in
        self.num_layers = config.num_layers
        self.in_proj = config.in_proj.make() if config.in_proj is not None else None
        blocks: list[nn.Module] = []
        for block_config in config.block:
            block = block_config.make()
            if not isinstance(block, nn.Module):
                raise TypeError("A Transformer block config must build an nn.Module.")
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = config.final_norm.make()
        self.out_proj: TensorModule | Literal["tied"] | None = (
            config.out_proj.make()
            if isinstance(config.out_proj, ChannelsInOutConfig)
            else config.out_proj
        )
        if self.out_proj == "tied" and (
            not has_weight(self.in_proj)
            or self.in_proj.weight.ndim != 2
            or self.in_proj.weight.shape[-1]
            != (config.block[-1].channels_out if config.block else config.channels_in)
        ):
            raise ValueError(
                "A tied out_proj requires an embedding-compatible in_proj weight."
            )

    def reset_parameters(self) -> None:
        for module in (self.in_proj, self.final_norm, *self.blocks, self.out_proj):
            if isinstance(module, Resettable):
                module.reset_parameters()

    def project_to_logits(self, hidden: Tensor, **kwargs: object) -> Tensor:
        """Apply the output projection, or return hidden states when absent."""
        if self.out_proj == "tied":
            assert has_weight(self.in_proj)
            return hidden @ self.in_proj.weight.T
        if self.out_proj is None:
            return hidden
        return self.out_proj(hidden, **kwargs)

    @override
    def forward(self, x: Tensor, /, **kwargs: object) -> Tensor:
        """Apply input projection, blocks, final norm, and output projection."""
        if self.in_proj is not None:
            x = self.in_proj(x)
        for block in self.blocks:
            x = cast(Tensor, block(x, **kwargs))
        return self.project_to_logits(self.final_norm(x, **kwargs), **kwargs)
