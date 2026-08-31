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

    from priml.model.attention.self_attention import SelfAttention

    CausalLM.Config(
        vocab_size=151_936,
        channels_in=128,
        num_layers=2,
        block=TransformerBlock.Config(
            attn=SelfAttention.Config(causal=True),
        ),
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
            if isinstance(self.block, list):
                templates = self.block
            else:
                templates = [self.block] * max(self.num_layers, 0)
            block_configs: list[TensorBlockConfig] = []
            for index, template in enumerate(templates):
                config = template.copy_tree()
                if isinstance(config, HasDepthIndex):
                    config.depth_index = ((index, self.num_layers),)
                block_configs.append(config)
            self.block = block_configs
            for cfg in block_configs:
                if cfg.channels_in == -1:
                    propagate_attr(
                        cfg, "channels_in", self.channels_in, protocol=ChannelsIn
                    )
                if cfg.channels_out == -1:
                    propagate_attr(
                        cfg, "channels_out", self.channels_in, protocol=ChannelsOut
                    )
            if (
                isinstance(self.final_norm, ChannelsIn)
                and self.final_norm.channels_in == -1
            ):
                propagate_attr(
                    self.final_norm,
                    "channels_in",
                    self.channels_in,
                    protocol=ChannelsIn,
                )
            if self.tie_embeddings:
                self.lm_head = None
            elif self.lm_head is not None:
                if (
                    isinstance(self.lm_head, ChannelsIn)
                    and self.lm_head.channels_in == -1
                ):
                    propagate_attr(
                        self.lm_head,
                        "channels_in",
                        self.channels_in,
                        protocol=ChannelsIn,
                    )
                if (
                    isinstance(self.lm_head, ChannelsOut)
                    and self.lm_head.channels_out == -1
                ):
                    propagate_attr(
                        self.lm_head,
                        "channels_out",
                        self.vocab_size,
                        protocol=ChannelsOut,
                    )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.channels_in != config.channels_out:
            raise ValueError(
                f"channels_in={config.channels_in} must equal "
                f"channels_out={config.channels_out} for CausalLM."
            )
        # Only what this config owns. A child's WIDTH is the child's invariant:
        # ``finalize`` propagates into every ``-1`` slot, a slot the caller set
        # explicitly is rejected by that child's own ``__init__``, and a residual
        # mismatch surfaces as a torch shape error naming both operands.
        #
        # The head's OUTPUT width is the exception, because it is checked against
        # ``vocab_size`` -- this config's own field, which no child can see. It
        # also fails silently: a wrong width yields logits of the wrong size and
        # forward succeeds, so nothing downstream names the cause.
        if (
            isinstance(config.lm_head, ChannelsOut)
            and config.lm_head.channels_out != config.vocab_size
        ):
            raise ValueError(
                f"lm_head.channels_out={config.lm_head.channels_out} must equal "
                f"vocab_size={config.vocab_size}."
            )
        block_configs = config.block
        if not isinstance(block_configs, list):
            raise TypeError("A finalized CausalLM config must contain a block list.")
        if len(block_configs) != config.num_layers:
            raise ValueError(
                f"block list length {len(block_configs)} != "
                f"num_layers={config.num_layers}.",
            )
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

    def project_to_logits(self, hidden: Tensor, **kwargs: object) -> Tensor:
        """Map ``hidden`` to vocab logits. Ties to ``embed`` when configured."""
        if self.tie_embeddings:
            return hidden @ self.embed.weight.T
        assert self.lm_head is not None
        return self.lm_head(hidden, **kwargs)

    @override
    def forward(self, tokens: Tensor, **kwargs: object) -> Tensor:
        """Full forward pass: tokens -> logits."""
        x: Tensor = self.embed(tokens)
        for block in self.blocks:
            x = cast(Tensor, block(x, **kwargs))
        x = self.final_norm(x, **kwargs)
        return self.project_to_logits(x, **kwargs)
