"""Causal language model: embed -> blocks -> final_norm -> lm_head.

Duck-types to :func:`priml.model.generate.generate` (which calls
``embed``, iterates ``blocks``, and invokes ``final_norm`` /
``project_to_logits``).

``block`` accepts a single ``Makeable`` (broadcast ``num_layers``
times, with per-layer ``depth`` set via a deep copy) or an explicit
``list[Makeable]`` of length ``num_layers`` for per-layer variation
(e.g. Kimi-K2's first-layer-dense-then-MoE). Matches
:class:`priml.model.Sequential.Config.elements`.

Example::

    CausalLM.Config(
        vocab_size=151_936,
        channels=128,
        num_layers=2,
        block=TransformerBlock.Config(causal=True),
        tie_embeddings=True,
    ).make()
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import KW_ONLY, field
from typing import Any, Self, cast, override

from configgle import Fig, Makeable
from torch import Tensor, nn

from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    TensorModule,
    propagate_attr,
)
from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.transformer import TransformerBlock


class CausalLM(nn.Module):
    """Decoder-only transformer LM."""

    class Config(Fig["CausalLM"], kw_only=False):
        vocab_size: int = -1
        """Token vocabulary size."""

        _: KW_ONLY

        channels: int = -1
        """Model width (embedding + residual stream dim)."""

        num_layers: int = -1
        """Number of stacked transformer blocks."""

        block: Makeable[nn.Module] | list[Makeable[nn.Module]] = field(
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
            if self.vocab_size < 1:
                raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}.")
            if self.num_layers < 1:
                raise ValueError(f"num_layers must be > 0, got {self.num_layers}.")
            if self.channels < 1:
                raise ValueError(f"channels must be > 0, got {self.channels}.")
            if isinstance(self.block, list) and len(self.block) != self.num_layers:
                raise ValueError(
                    f"block list length {len(self.block)} != "
                    f"num_layers={self.num_layers}.",
                )
            # Propagate channels into nested configs before super().finalize()
            # (post-super mutation wouldn't stick — configgle freezes children).
            block_configs = self.block if isinstance(self.block, list) else [self.block]
            for cfg in (*block_configs, self.final_norm):
                propagate_attr(cfg, "channels_in", self.channels, protocol=ChannelsIn)
            if self.lm_head is not None:
                propagate_attr(
                    self.lm_head, "channels_in", self.channels, protocol=ChannelsIn
                )
                propagate_attr(
                    self.lm_head, "channels_out", self.vocab_size, protocol=ChannelsOut
                )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.channels = config.channels
        self.num_layers = config.num_layers
        self.tie_embeddings = config.tie_embeddings

        self.embed = Embedding.Config(
            channels_out=config.channels,
            num_embeddings=config.vocab_size,
            shard="vocab",
        ).make()
        # Resolve block configs: scalar → broadcast num_layers × deepcopy;
        # list → one config per layer. ``depth`` is backdoored via
        # object.__setattr__ since the finalized Fig is frozen.
        raw_block = config.block
        explicit_list = isinstance(raw_block, list)
        # ``cast`` narrows the Makeable|list[Makeable] union for ty;
        # basedpyright already narrows from the isinstance branch but
        # ty disagrees, so the cast is load-bearing for ty only.
        templates: list[Makeable[nn.Module]] = cast(  # pyright: ignore[reportUnnecessaryCast]
            "list[Makeable[nn.Module]]",
            raw_block if explicit_list else [raw_block],
        )
        self.blocks = nn.ModuleList()
        for i in range(config.num_layers):
            template = templates[i] if explicit_list else templates[0]
            block_cfg = deepcopy(template)
            if hasattr(block_cfg, "depth"):
                object.__setattr__(block_cfg, "depth", i)
            self.blocks.append(block_cfg.make())
        self.final_norm = config.final_norm.make()
        if config.tie_embeddings:
            self.lm_head: TensorModule | None = None
        elif config.lm_head is not None:
            self.lm_head = config.lm_head.make()
        else:
            self.lm_head = Linear.Config(
                channels_in=config.channels,
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
    def forward(self, tokens: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Full forward pass: tokens -> logits."""
        del args, kwargs
        x: Tensor = self.embed(tokens)
        for block in self.blocks:
            out = block(x)
            x = (
                cast("Tensor", out[0])
                if isinstance(out, tuple)
                else cast("Tensor", out)
            )
        x = self.final_norm(x)
        return self.project_to_logits(x)
