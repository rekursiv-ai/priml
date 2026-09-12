"""Compose a native causal language backbone with added attention streams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import field
from typing import Self, cast, override

from configgle import Fig
from torch import Tensor, nn

from priml.model.attention.multi_stream import (
    MultiStreamAttention,
    _validate_native_state,
)
from priml.model.attention.self_attention import (
    AttentionProjections,
    SelfAttention,
)
from priml.model.custom_types import Resettable, TransformerConfig
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.mmdit import MMDiTBlock, MMDiTStream
from priml.model.transformer.transformer import Transformer


class MMDiTGraft(nn.Module):
    """Compose language and continuous modality streams with joint attention.

    Stream zero is language. Modality adapters remain caller-owned. Required
    per-stream masks determine visibility; freezing alone does not preserve
    language computation. No KV-cache generation path is provided.
    """

    class Config(Fig["MMDiTGraft"]):
        backbone: TransformerConfig = field(default_factory=Transformer.Config)
        """Transformer architecture exposing projections and blocks."""

        streams: list[MMDiTStream.Config] = field(
            default_factory=lambda: [MMDiTStream.Config()],
        )
        """Additional stream templates, copied independently into each layer."""

        block: list[MMDiTBlock.Config] = field(
            default_factory=list[MMDiTBlock.Config],
            init=False,
        )
        """Joint layers derived from the backbone and stream templates."""

        @override
        def finalize(self) -> Self:
            source = self.backbone.finalize()
            self.block = []
            if isinstance(source.block, list):
                for layer in source.block:
                    if not isinstance(layer, TransformerBlock.Config):
                        continue
                    attn = layer.attn
                    if not isinstance(attn, SelfAttention.Config):
                        continue
                    language = MMDiTStream.Config()
                    language.attn = AttentionProjections.Config().update(
                        attn,
                        skip_missing=True,
                    )
                    language.attn.causal = False
                    language.norm1 = layer.norm1.copy_tree()
                    language.norm2 = layer.norm2.copy_tree()
                    language.ffn = layer.ffn.copy_tree()
                    joint = MMDiTBlock.Config()
                    joint.channels_in = layer.channels_in
                    joint.depth_index = layer.depth_index
                    joint.attn = MultiStreamAttention.Config()
                    joint.attn.num_heads = attn.num_heads
                    joint.attn.num_heads_kv = attn.num_heads_kv
                    joint.attn.channels_head = attn.channels_head
                    joint.attn.attn_kernel = attn.attn_kernel.copy_tree()
                    joint.streams = [
                        language,
                        *(stream.copy_tree() for stream in self.streams),
                    ]
                    self.block.append(joint)
            for stream in self.streams:
                if stream.channels_in == -1 and self.block:
                    stream.channels_in = self.block[0].channels_in
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        source = config.backbone
        if not config.streams:
            raise ValueError("A graft requires at least one additional stream.")
        if not isinstance(source.block, list) or not source.block:
            raise ValueError("A graft requires a nonempty language backbone.")
        for layer in source.block:
            if not isinstance(layer, TransformerBlock.Config) or not layer.prenorm:
                raise ValueError("Grafting requires native prenorm transformer blocks.")
            if not isinstance(layer.attn, SelfAttention.Config):
                raise TypeError("Grafting requires native SelfAttention blocks.")
        self.num_streams = len(config.streams) + 1
        self.in_proj = source.in_proj.make() if source.in_proj is not None else None
        self.blocks = nn.ModuleList(block.make() for block in config.block)
        self.out_proj = source.out_proj.make() if source.out_proj is not None else None

    def load_backbone(self, source: Transformer) -> None:
        """Load native language weights, preflighting all layers before copying.

        The source must match Config.backbone, including non-tensor settings
        such as norm epsilon and rotary frequencies. Modality parameters and
        existing requires_grad flags are unchanged.

        Args:
          source: Pre-trained Transformer with native architecture (prenorm,
            self-attention, no modality branches).

        """
        if any(
            not isinstance(block, TransformerBlock)
            or not block.prenorm
            or not isinstance(block.attn, SelfAttention)
            for block in source.blocks
        ):
            raise ValueError("Loading requires native prenorm SelfAttention blocks.")
        target = self._backbone_view()
        _validate_native_state(target, source=source)
        target.load_state_dict(source.state_dict(), strict=True)

    def load_backbone_state(self, state_dict: Mapping[str, Tensor]) -> None:
        """Load native language weights keyed as a ``Transformer`` names them.

        Args:
          state_dict: State dict.

        """
        self._backbone_view().load_state_dict(state_dict, strict=True)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze only the language stream, embedding, and head.

        Args:
          freeze: Freeze.

        """
        self._backbone_view().requires_grad_(not freeze)

    def reset_parameters(self) -> None:
        """Reset all owned modules using their configured initializers."""
        for module in (self.in_proj, *self.blocks, self.out_proj):
            if isinstance(module, Resettable):
                module.reset_parameters()

    @override
    def forward(
        self,
        tokens: Tensor,
        streams: Sequence[Tensor],
        *,
        attn_mask: Tensor | Sequence[Tensor | None],
        c: Tensor | Sequence[Tensor | None] | None = None,
        cos_sin: Sequence[tuple[Tensor, Tensor] | None] | None = None,
        **kwargs: object,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Compute joint blocks using explicit per-stream visibility.

        Masks address concatenated keys, language first; use additive 0/-inf
        masks for kernel-independent semantics. Conditioning includes a None
        entry for language. Rotary factors may be supplied per stream.

        Returns:
          logits: Language logits, shaped [batch, language_length, vocab_size].
          streams: Updated continuous modality hidden states.

        """
        if len(streams) != self.num_streams - 1:
            raise ValueError(
                f"Expected {self.num_streams - 1} modality streams, got {len(streams)}.",
            )
        language = self.in_proj(tokens) if self.in_proj is not None else tokens
        hidden: tuple[Tensor, ...] = (language, *streams)
        for block in self.blocks:
            hidden = block(hidden, c=c, cos_sin=cos_sin, attn_mask=attn_mask, **kwargs)
        language = hidden[0]
        logits = (
            language if self.out_proj is None else self.out_proj(language, **kwargs)
        )
        return logits, hidden[1:]

    def _backbone_view(self) -> nn.ModuleDict:
        blocks = [
            nn.ModuleDict(
                {
                    "attn": block.attn.streams[0],
                    "norm1": block.norms1[0],
                    "norm2": block.norms2[0],
                    "ffn": block.ffns[0],
                },
            )
            for block in self.blocks
        ]
        modules: dict[str, nn.Module] = {}
        if self.in_proj is not None:
            modules["in_proj"] = cast(nn.Module, self.in_proj)
        modules["blocks"] = nn.ModuleList(blocks)
        if self.out_proj is not None:
            modules["out_proj"] = cast(nn.Module, self.out_proj)
        return nn.ModuleDict(modules)
