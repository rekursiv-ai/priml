"""Causal language model: embed -> blocks -> final_norm -> lm_head.

Duck-types to :func:`priml.model.generate.generate` (which calls
``embed``, iterates ``blocks``, and invokes ``final_norm`` /
``project_to_logits``).

``block`` accepts a single ``Makeable`` (broadcast ``num_layers``
times, with per-layer ``depth_index`` set via ``copy_tree``) or an explicit
``list[Makeable]`` of length ``num_layers`` for per-layer variation
(e.g. Kimi-K2's first-layer-dense-then-MoE). Matches
:class:`priml.model.Sequential.Config.elements`.

Example::

    CausalLM.Config(
        vocab_size=151_936,
        channels_in=128,
        num_layers=2,
        block=TransformerBlock.Config(causal=True),
        tie_embeddings=True,
    ).make()
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
    TensorBlockConfig,
    TensorModule,
    propagate_attr,
)
from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.transformer.block import TransformerBlock


class CausalLM(nn.Module):
    """Decoder-only transformer LM."""

    class Config(Fig["CausalLM"], kw_only=False):
        channels_in: int = -1
        """Model width (embedding + residual stream dim)."""

        channels_out: int = -1
        """Residual-stream width (-1 to infer from channels_in)."""

        _: KW_ONLY

        vocab_size: int = -1
        """Token vocabulary size."""

        num_layers: int = -1
        """Number of stacked transformer blocks."""

        block: TensorBlockConfig | list[TensorBlockConfig] = field(
            default_factory=TransformerBlock.Config,
        )
        """Block template (broadcast ``num_layers`` times) or explicit
        per-layer list (length must equal ``num_layers``)."""

        final_norm: Makeable[TensorModule] = field(default_factory=RMSNorm.Config)
        """Normalization applied to the final hidden state."""

        tie_embeddings: bool = False
        """Reuse the token embedding matrix as the output projection."""

        lm_head: Makeable[TensorModule] | None = None
        """Explicit output projection. Ignored when ``tie_embeddings``."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            if self.channels_in != self.channels_out:
                raise ValueError(
                    f"channels_in={self.channels_in} must equal "
                    f"channels_out={self.channels_out} for CausalLM."
                )
            if self.vocab_size < 1:
                raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}.")
            if self.num_layers < 1:
                raise ValueError(f"num_layers must be > 0, got {self.num_layers}.")
            if self.channels_in < 1:
                raise ValueError(f"channels_in must be > 0, got {self.channels_in}.")
            if isinstance(self.block, list) and len(self.block) != self.num_layers:
                raise ValueError(
                    f"block list length {len(self.block)} != "
                    f"num_layers={self.num_layers}.",
                )
            templates = self.block if isinstance(self.block, list) else [self.block]
            block_configs: list[TensorBlockConfig] = []
            for index in range(self.num_layers):
                template = templates[index] if len(templates) > 1 else templates[0]
                config = template.copy_tree()
                if isinstance(config, HasDepthIndex):
                    config.depth_index = ((index, self.num_layers),)
                block_configs.append(config)
            self.block = block_configs
            for index, cfg in enumerate(block_configs):
                if cfg.channels_in not in (-1, self.channels_in):
                    raise ValueError(
                        f"block[{index}].channels_in={cfg.channels_in} must equal "
                        f"channels_in={self.channels_in}."
                    )
                if cfg.channels_out not in (-1, self.channels_in):
                    raise ValueError(
                        f"block[{index}].channels_out={cfg.channels_out} must equal "
                        f"channels_in={self.channels_in}."
                    )
                propagate_attr(
                    cfg, "channels_in", self.channels_in, protocol=ChannelsIn
                )
                propagate_attr(
                    cfg, "channels_out", self.channels_in, protocol=ChannelsOut
                )
            for cfg in (self.final_norm,):
                propagate_attr(
                    cfg, "channels_in", self.channels_in, protocol=ChannelsIn
                )
            if self.tie_embeddings:
                self.lm_head = None
            elif self.lm_head is not None:
                propagate_attr(
                    self.lm_head, "channels_in", self.channels_in, protocol=ChannelsIn
                )
                propagate_attr(
                    self.lm_head, "channels_out", self.vocab_size, protocol=ChannelsOut
                )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.channels_in = config.channels_in
        self.num_layers = config.num_layers
        self.tie_embeddings = config.tie_embeddings

        self.embed = Embedding.Config(
            channels_out=config.channels_in,
            num_embeddings=config.vocab_size,
            shard="vocab",
        ).make()
        block_configs = config.block
        assert isinstance(block_configs, list)
        blocks: list[nn.Module] = []
        for block_config in block_configs:
            block = block_config.make()
            if not isinstance(block, nn.Module):
                raise TypeError("A CausalLM block config must build an nn.Module.")
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = config.final_norm.make()
        if config.tie_embeddings:
            self.lm_head: TensorModule | None = None
        elif config.lm_head is not None:
            self.lm_head = config.lm_head.make()
        else:
            self.lm_head = Linear.Config(
                channels_in=config.channels_in,
                channels_out=config.vocab_size,
                bias=False,
                shard="vocab",
            ).make()

    def reset_parameters(self) -> None:
        for m in (self.embed, self.final_norm):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for block in self.blocks:
            if hasattr(block, "reset_parameters"):
                block.reset_parameters()
        if self.lm_head is not None and hasattr(self.lm_head, "reset_parameters"):
            self.lm_head.reset_parameters()

    def project_to_logits(self, hidden: Tensor) -> Tensor:
        """Map ``hidden`` to vocab logits. Ties to ``embed`` when configured."""
        if self.tie_embeddings:
            return hidden @ self.embed.weight.T
        assert self.lm_head is not None
        return self.lm_head(hidden)

    @override
    def forward(self, tokens: Tensor, **kwargs: object) -> Tensor:
        """Full forward pass: tokens -> logits."""
        x: Tensor = self.embed(tokens)
        for block in self.blocks:
            x = cast(Tensor, block(x, **kwargs))
        x = self.final_norm(x)
        return self.project_to_logits(x)
