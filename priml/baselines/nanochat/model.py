"""A decoder-only language model: embed, blocks, project to the vocabulary.

Every piece the stack is built from is a slot filled from priml -- the block,
its attention and feed-forward, the tables, the projection, the mix. What lives
HERE is only what a block cannot know: where in the stack it sits.

Three things follow from that position and nowhere else. Which layers attend
over a window and which see the whole context, since that is a property of
depth. Which layers read a value embedding, derived from a stride so a fork
changing the depth does not carry indices for a stack that no longer exists.
And the tables those layers read, sized to the attention's inner width because
they are added to its VALUES.

References:
    https://arxiv.org/abs/2410.17897
      Zhou et al. Value Residual Learning.
    https://arxiv.org/abs/2004.05150
      Beltagy et al. Longformer: The Long-Document Transformer.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import field
from functools import partial
from typing import Any, Self, override

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.model.custom_types import (
    AttentionBlock,
    ChannelsIn,
    ChannelsOut,
    HeadGeometry,
    TensorModule,
    propagate_attr,
)
from priml.model.embedding import Embedding
from priml.model.init import normal, unit_fan_in_uniform
from priml.model.linear import Linear
from priml.model.narrow_embedding import NarrowEmbedding
from priml.model.norm import RMSNorm
from priml.model.relu_squared import ReluSquared
from priml.model.residual_mix import ResidualMix
from priml.model.rope import RoPE
from priml.model.softcap import SoftCap
from priml.model.transformer import TransformerBlock
from priml.model.value_gated_attention import (
    ValueGatedAttention,
)


class NanoChatLM(nn.Module):
    """Decoder-only transformer with windowed attention and value embeddings.

    A forward embeds the tokens, runs every block over a residual stream
    re-mixed with the original embedding at each layer, and projects the result
    to soft-capped vocabulary logits.
    """

    class Config(Fig["NanoChatLM"]):
        """Shape, the injected block, and the per-layer patterns."""

        vocab_size: int = 8192
        """Token vocabulary size."""

        max_seq_len: int = 2048
        """Context length; also the long attention window."""

        channels_in: int = 512
        """Model width, and the width every block inherits."""

        num_layers: int = 8
        """Blocks in the stack."""

        block: AttentionBlock | Sequence[AttentionBlock] = field(
            default_factory=lambda: TransformerBlock.Config(
                attn=ValueGatedAttention.Config(),
                ffn=ReluSquared.Config(),
            ),
        )
        """Block template (broadcast ``num_layers`` times) or per-layer list.

        Typed by what the stack READS from a layer rather than by a class: it
        pushes a reach and a gate into the attention, and sizes its tables off
        that attention's heads. Any block exposing one qualifies, so the
        architecture stays a value here rather than an edit to this class."""

        value_embedding_stride: int = 0
        """Every Nth layer reads a value embedding, counting BACK from the last.

        0 gives none of them one. Counting back so the deepest layer always
        gets a table: the embedding is a path from the raw tokens to the
        output, worth the most where the residual stream is most processed.

        A STRIDE rather than a list of indices, because indices computed
        against one depth go stale the moment a fork changes ``num_layers`` --
        the list is a snapshot, this is the rule that produced it."""

        embedding: Makeable[nn.Module] = field(
            default_factory=lambda: NarrowEmbedding.Config(
                inner=Embedding.Config(init_weight=partial(normal, std=1.0)),
            ),
        )
        """Token embedding table.

        Unit variance, which is the recipe's (``train.py:150``): the table feeds
        an RMS norm that divides its scale out, so only the relative spread
        survives.

        Wrapped rather than bare, because the recipe holds its tables narrower
        than it draws them and that ordering belongs to whatever owns both
        steps. A rung wanting a full-precision table leaves the wrapper's
        ``dtype`` at None or supplies the bare table."""

        norm: Makeable[TensorModule] = field(
            default_factory=lambda: RMSNorm.Config(eps=None),
        )
        """Normalization on the embedding and before the output projection."""

        lm_head: Makeable[TensorModule] = field(
            default_factory=lambda: SoftCap.Config(
                inner=Linear.Config(
                    bias=False,
                    init_weight=partial(normal, std=0.001),
                ),
            ),
        )
        """Output projection to vocabulary logits.

        Capped, and the cap travels with the projection: bounding what one
        confident token contributes to the gradient is what keeps the run
        stable at a large learning rate. A rung wanting an uncapped readout
        supplies the bare projection."""

        rope: RoPE.Config = field(
            default_factory=lambda: RoPE.Config(hf_inv_freq=True),
        )
        """Rotary position embedding driving every layer's queries and keys.

        Its width is pushed down from the block's attention at finalize, since
        the model builds the factors and the heads consume them."""

        mix: ResidualMix.Config = field(default_factory=ResidualMix.Config)
        """Per-layer mix of the running stream with the token embedding.

        What lets a deep stack keep a path back to its input without a skip
        connection per layer. Its depth is pushed down at finalize."""

        @property
        def template(self) -> TransformerBlock.Config:
            """The single block config every layer is copied from.

            Raises rather than narrowing silently: a caller reaching for the
            template after a per-layer LIST was supplied is asking which of
            them it is, and the honest answer is that the question is wrong.

            Raises:
              TypeError: ``block`` holds a per-layer list, or a block that is
                not a transformer.

            """
            if not isinstance(self.block, TransformerBlock.Config):
                raise TypeError(
                    f"block is {type(self.block).__name__}, so there is no one "
                    "template; index it, or set a TransformerBlock.Config.",
                )
            return self.block

        @property
        def value_embedding_layers(self) -> list[int]:
            """Layers reading a value embedding, from the stride.

            Derived rather than stored, so the two can never disagree: a fork
            changing ``num_layers`` re-reads the RULE instead of carrying
            indices computed against a stack that no longer exists.
            """
            if self.value_embedding_stride <= 0:
                return []
            return sorted(
                range(self.num_layers - 1, -1, -self.value_embedding_stride),
            )

        @override
        def finalize(self) -> Self:
            if self.vocab_size <= 0:
                raise ValueError(f"vocab_size must be positive; got {self.vocab_size}.")
            if self.max_seq_len < 2:
                raise ValueError(
                    f"max_seq_len must be at least two; got {self.max_seq_len}.",
                )
            if self.num_layers <= 0 or self.channels_in <= 0:
                raise ValueError(
                    "num_layers and channels_in must be positive; got "
                    f"{self.num_layers} and {self.channels_in}.",
                )
            if self.value_embedding_stride < 0:
                raise ValueError(
                    "value_embedding_stride must be nonnegative; got "
                    f"{self.value_embedding_stride}.",
                )
            if not isinstance(self.block, Sequence):
                self.block = [self.block.copy_tree() for _ in range(self.num_layers)]
            if len(self.block) != self.num_layers:
                raise ValueError(
                    f"block names {len(self.block)} layers for "
                    f"num_layers={self.num_layers}.",
                )
            gated = set(self.value_embedding_layers)
            last = self.num_layers - 1
            for layer, block in enumerate(self.block):
                propagate_attr(
                    block, "channels_in", self.channels_in, protocol=ChannelsIn
                )
                propagate_attr(
                    block, "channels_out", self.channels_in, protocol=ChannelsOut
                )
                propagate_attr(block, "depth", layer)
                attention = block.attn
                if isinstance(attention, ValueGatedAttention.Config):
                    attention.max_seq_len = self.max_seq_len
                    if layer == last:
                        attention.window = self.max_seq_len
                    attention.gated = layer in gated
            propagate_attr(self.embedding, "channels_out", self.channels_in)
            propagate_attr(self.embedding, "num_embeddings", self.vocab_size)
            propagate_attr(self.lm_head, "channels_in", self.channels_in)
            propagate_attr(self.lm_head, "channels_out", self.vocab_size)
            propagate_attr(
                self.norm, "channels_in", self.channels_in, protocol=ChannelsIn
            )
            self.mix.num_layers = self.num_layers
            self.rope.channels_head = _head_shape(self.block[0], self.channels_in)[0]
            finalized = super().finalize()
            # AFTER the cascade: a block that left ``heads`` at its sentinel
            # derives it in its own finalize, so checking earlier would compare
            # a fallback rather than the width the layer will actually use.
            #
            # The value-embedding tables and the rotary factors are built ONCE,
            # to layer 0's geometry, and handed to every layer -- so a stack
            # disagreeing on head shape is a contradiction settled here. Left to
            # run time it surfaces as a reshape failure naming a tensor size
            # rather than the layer.
            assert isinstance(finalized.block, list)
            _reject_ragged_heads(finalized.block, channels_in=finalized.channels_in)
            return finalized

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        assert isinstance(config.block, list)
        # The table is read as the attention's VALUES, so it spans every head,
        # not one: a per-head width would reshape to the wrong sequence length.
        _, width = _head_shape(config.block[0], config.channels_in)
        # Construction order fixes the global-RNG draw order, so a seeded init
        # is reproducible: tokens, head, blocks, value embeddings.
        embedding = config.embedding.make()
        assert isinstance(embedding, nn.Module)
        self.embed = embedding
        self.lm_head = config.lm_head.make()
        self.blocks = nn.ModuleList([block.make() for block in config.block])
        self.value_embeds = nn.ModuleDict(
            {
                str(layer): self._value_table(config, width=width)
                for layer in config.value_embedding_layers
            },
        )
        self.norm_embed = config.norm.make()
        self.norm_out = config.norm.make()
        self.mix = config.mix.make()
        self.rope = config.rope.make()

    @staticmethod
    def _value_table(config: Config, *, width: int) -> nn.Module:
        """One value-embedding table, narrowed like the token table.

        Sized and built here rather than injected: there is one per
        PARTICIPATING layer, and which layers those are is derived from the
        stride, so the count is not something a config can state ahead of the
        depth. The narrowing follows the token table's, since both are read as
        lookups into the same stream.
        """
        table = NarrowEmbedding.Config(
            inner=Embedding.Config(init_weight=unit_fan_in_uniform),
        )
        table.channels_out = width
        table.num_embeddings = config.vocab_size
        if isinstance(config.embedding, NarrowEmbedding.Config):
            table.dtype = config.embedding.dtype
        built = table.make()
        assert isinstance(built, nn.Module)
        return built

    def reset_parameters(self) -> None:
        """Re-initialize everything this model constructed.

        Meta-device materialization drives init through here alone
        (``train/parallelism.py``), so a child left out trains on ``to_empty``'s
        garbage -- which that path detects by poisoning with NaN and auditing
        after, but only after the run has been set up.
        """
        for module in (
            self.embed,
            self.lm_head,
            self.norm_embed,
            self.norm_out,
            self.mix,
            *self.blocks,
            *self.value_embeds.values(),
        ):
            module.reset_parameters()

    @override
    def forward(self, tokens: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Map ``[B, S]`` token ids to ``[B, S, vocab_size]`` logits.

        Args:
          tokens: Input token ids.
          *args: Ignored; present for the priml model call contract.
          **kwargs: Ignored; present for the priml model call contract.

        Returns:
          logits: Soft-capped vocabulary logits, in float32.

        """
        del args, kwargs
        length = tokens.shape[-1]
        if length > self.config.max_seq_len:
            raise ValueError(
                f"Input length {length} exceeds max_seq_len={self.config.max_seq_len}.",
            )
        cos_sin = self.rope(torch.arange(length, device=tokens.device))
        x = self.norm_embed(self.embed(tokens))
        original = x
        for layer, block in enumerate(self.blocks):
            x = self.mix(x, original, layer=layer)
            # ``ModuleDict`` is not a Mapping -- it has no ``get`` -- so
            # membership is tested before the lookup.
            name = str(layer)
            gated = name in self.value_embeds
            out = block(
                x,
                cos_sin=cos_sin,
                value_embedding=self.value_embeds[name](tokens) if gated else None,
            )
            assert isinstance(out, Tensor)
            x = out
        return self.lm_head(self.norm_out(x))

    def flops_per_token(self) -> int:
        """Estimated forward-plus-backward FLOPs for one token.

        Counts each matrix parameter's multiply-accumulate three times (once
        forward, twice backward) and adds the attention scores, which scale
        with the window rather than with any parameter count. Lookup tables are
        excluded: a gather does no arithmetic.

        Returns:
          flops: FLOPs attributable to one token of one sequence.

        """
        gathered = {
            id(parameter)
            for module in (self.embed, *self.value_embeds.values(), self.mix)
            for parameter in module.parameters()
        }
        matrix = sum(
            parameter.numel()
            for parameter in self.parameters()
            if id(parameter) not in gathered
        )
        config = self.config
        assert isinstance(config.block, list)
        # ``heads * channels_head``, read from the block rather than derived
        # from the model width: the attention's inner width is decoupled from
        # the residual stream, so dividing would miscount every model where
        # they differ.
        _, inner = _head_shape(config.block[0], config.channels_in)
        attention = sum(12 * inner * _window(block) for block in self.blocks)
        return 6 * matrix + attention


def _window(block: nn.Module) -> int:
    """How far back a built block attends.

    Args:
      block: One layer of the stack.

    Returns:
      window: Keys its queries reach over.

    Raises:
      TypeError: The block's attention declares no window, so its cost cannot
        be attributed and a guess would be reported as a measurement.

    """
    attention = getattr(block, "attn", None)
    config = getattr(attention, "config", None)
    window = getattr(config, "window", None)
    if not isinstance(window, int):
        raise TypeError(f"{type(block).__name__} declares no attention window.")
    return window


def _head_shape(block: object, channels_in: int) -> tuple[int, int]:
    """A block's ``(channels_head, heads * channels_head)``.

    Asked OF THE BLOCK: where its attention sits is its own business, and a
    reader reaching for a fixed attribute gets the wrong answer from every
    wrapper. A block declaring no geometry has no attention and is treated as
    one head of the full width.

    Args:
      block: The block config the model was handed.
      channels_in: Model width, for a block declaring no geometry.

    Returns:
      shape: Per-head width, and that width across every head.

    """
    if not isinstance(block, HeadGeometry):
        return channels_in, channels_in
    return block.channels_head, block.heads * block.channels_head


def _reject_ragged_heads(
    blocks: Sequence[AttentionBlock],
    *,
    channels_in: int,
) -> None:
    """Raise unless every block agrees on its attention's head geometry.

    Args:
      blocks: The per-layer block configs.
      channels_in: Model width, the fallback for a block declaring no heads.

    Raises:
      ValueError: Two layers declare different head shapes. The model sizes its
        shared tensors from layer 0, so the disagreement is unbuildable -- and
        naming it here beats a reshape error deep inside the forward.

    """
    shapes = {_head_shape(block, channels_in) for block in blocks}
    if len(shapes) > 1:
        raise ValueError(
            "every block must declare the same attention head geometry, since "
            "the value embeddings and rotary factors are shared across layers; "
            f"got (channels_head, heads * channels_head) of {sorted(shapes)}.",
        )
