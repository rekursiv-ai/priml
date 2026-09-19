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

from collections.abc import Callable, Sequence
from dataclasses import field
from functools import partial
from typing import Protocol, Self, cast, override

import functools

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn

import torch

from priml.baselines.nanochat.ngram import (
    HashedNgramTables,
    NgramSource,
    clear_marked_sinks,
)
from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    reduction_cost,
    traffic,
)
from priml.model.attention.rope import RoPE
from priml.model.attention.value_gated_attention import (
    ValueGatedAttention,
)
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    EmbeddingConfig,
    HasAttention,
    HasDepthIndex,
    HasResetParameters,
    HeadGeometry,
    TensorModule,
    propagate_attr,
)
from priml.model.embedding import Embedding
from priml.model.init import normal, unit_fan_in_uniform
from priml.model.linear import Linear
from priml.model.narrow_embedding import NarrowEmbedding
from priml.model.norm import RMSNorm
from priml.model.residual_mix import ResidualMix
from priml.model.softcap import SoftCap
from priml.model.special import Identity
from priml.model.swiglu import SwiGLUReluSquared
from priml.model.transformer.block import TransformerBlock


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

        block: HasAttention | Sequence[HasAttention] = field(
            default_factory=lambda: TransformerBlock.Config(
                attn=ValueGatedAttention.Config(),
                ffn=SwiGLUReluSquared.Config(round_to=1),
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

        embedding: EmbeddingConfig = field(
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
        ``dtype`` at None or supplies the bare table. The value tables are
        held at whatever ``dtype`` this declares."""

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

        rope: RoPE.Config = field(default_factory=RoPE.Config)
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
                    block,
                    "channels_in",
                    self.channels_in,
                    protocol=ChannelsIn,
                )
                propagate_attr(
                    block,
                    "channels_out",
                    self.channels_in,
                    protocol=ChannelsOut,
                )
                propagate_attr(
                    block,
                    "depth_index",
                    ((layer, self.num_layers),),
                    protocol=HasDepthIndex,
                )
                attention = block.attn
                if isinstance(attention, ValueGatedAttention.Config):
                    attention.max_seq_len = self.max_seq_len
                    if layer == last:
                        attention.window = self.max_seq_len
                    attention.gated = layer in gated
                # Ahead of the cascade: an attention left at its ``num_heads``
                # sentinel derives the count inside its own finalize, and the
                # rotary factors and value tables below are sized from it.
                if not getattr(block, "_finalized", False):
                    block.finalize()
            # The value-embedding tables and the rotary factors are built ONCE,
            # to layer 0's geometry, and handed to every layer -- so a stack
            # disagreeing on head shape is a contradiction settled here. Left to
            # run time it surfaces as a reshape failure naming a tensor size
            # rather than the layer.
            _reject_ragged_heads(self.block)
            propagate_attr(self.embedding, "channels_out", self.channels_in)
            propagate_attr(self.embedding, "channels_in", self.vocab_size)
            propagate_attr(self.lm_head, "channels_in", self.channels_in)
            propagate_attr(self.lm_head, "channels_out", self.vocab_size)
            propagate_attr(
                self.norm,
                "channels_in",
                self.channels_in,
                protocol=ChannelsIn,
            )
            self.mix.num_layers = self.num_layers
            self.mix.channels_in = self.channels_in
            self.rope.channels_head = cast(
                HeadGeometry,
                self.block[0].attn,
            ).channels_head
            self._propagate_layer_table_widths()
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Sum the tables, both norms, the mix, every block, and the head.

            What the stack owns and no child can cost: the value tables, one
            per gated layer and sized to the attention's inner width, and the
            residual mix at the model width. The gate that reads a value table
            is the attention's own and is costed there.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            assert isinstance(self.block, list)
            table = _value_table_config(self, width=_inner_width(self.block[0]))
            child_cost = functools.partial(
                cost,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            return sum(
                (child_cost(child) for child in (*self.block, self.lm_head)),
                child_cost(self.embedding)
                + child_cost(self.norm).tile(2, copies=2)
                + child_cost(self.mix)
                + child_cost(self.rope)
                + child_cost(table).tile(
                    len(self.value_embedding_layers),
                    copies=len(self.value_embedding_layers),
                ),
            )

        def _propagate_layer_table_widths(self) -> None:
            """Propagate attention-value widths into subclass-owned layer tables."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        assert isinstance(config.block, list)
        # The table is read as the attention's VALUES, so it spans every head,
        # not one: a per-head width would reshape to the wrong sequence length.
        width = _inner_width(config.block[0])
        # Construction order fixes the global-RNG draw order, so a seeded init
        # is reproducible: tokens, head, blocks, value embeddings.
        embedding = config.embedding.make()
        # Registration: ``nn.Module.__setattr__`` only adopts a child that IS a
        # Module, so a plain-callable embedding would silently never train.
        assert isinstance(embedding, nn.Module)
        self.embed: TensorModule = embedding
        self.lm_head = config.lm_head.make()
        blocks: list[nn.Module] = []
        for block in config.block:
            built = block.make()
            assert isinstance(built, nn.Module)
            blocks.append(built)
        self.blocks = nn.ModuleList(blocks)
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
        # The rotation table depends on the sequence length and nothing else,
        # so a forward that rebuilds it pays an outer product, a cosine, a
        # sine, and two casts for a tensor it already had. Measured on a 5090:
        # 12.5 of the 13.1 ms/step this recipe stood above its reference.
        #
        # Built lazily rather than here: the module is constructed on meta and
        # materialized later, and a table built on meta holds no values.
        self._rotation: tuple[Tensor, Tensor] | None = None

    # Sized and built here rather than injected: there is one per PARTICIPATING layer,
    # and which layers those are is derived from the stride, so the count is not
    # something a config can state ahead of the depth. The narrowing follows the token
    # table's, since both are read as lookups into the same stream.
    @classmethod
    def _value_table(cls, config: Config, *, width: int) -> nn.Module:
        """One value-embedding table, narrowed like the token table."""
        built = _value_table_config(config, width=width).make()
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
            if isinstance(module, HasResetParameters):
                module.reset_parameters()
        # The rotation table is derived from the rope's frequencies, which a
        # device move rebuilds (rope.py:390-393) because the transcendental
        # differs by a bit between CPU and CUDA. Dropping it here keeps a
        # materialized model from reading factors its rope no longer holds.
        self._rotation = None

    @override
    def forward(self, tokens: Tensor, *args: object, **kwargs: object) -> Tensor:
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
        cos_sin = self._rotation_table(length, device=tokens.device)
        x = self.norm_embed(self.embed(tokens))
        original = x
        for layer, block in enumerate(self.blocks):
            x = self.mix(x, original=original, layer=layer)
            # ``ModuleDict`` is not a Mapping -- it has no ``get`` -- so
            # membership is tested before the lookup.
            name = str(layer)
            gated = name in self.value_embeds
            block_call = cast(_BlockCallable, block)
            out = block_call(
                x,
                cos_sin=cos_sin,
                value_embedding=self.value_embeds[name](tokens) if gated else None,
            )
            x = out
        return self.lm_head(self.norm_out(x))

    # Held for ``max_seq_len`` and SLICED, so a short batch reads a prefix of the same
    # table rather than minting another. That is the reference's own arrangement
    # (train.py:143-145, 269) and it matters for more than speed: the factors come from
    # a transcendental, so a table built at one length and one built at another need not
    # agree bit for bit in their common prefix.
    def _rotation_table(
        self,
        length: int,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Return the rotation factors for ``length`` positions, built once."""
        if self._rotation is None or self._rotation[0].device != device:
            # REBUILT on a device change, never moved there: the factors come
            # from a transcendental whose last bit differs between CPU and CUDA
            # (rope.py:395-401), so a moved table is not the table that device
            # would have produced.
            positions = torch.arange(self.config.max_seq_len, device=device)
            self._rotation = self.rope(positions)
        cos, sin = self._rotation
        return cos[:length], sin[:length]

    def flops_per_token(self) -> float:
        """Report matmul FLOPs per token at the configured context.

        Derives this rate from one complete invocation and excludes non-matmul
        arithmetic.

        Returns:
          flops: Whole-invocation matmul FLOPs divided by its token count.

        """
        whole = self.config.cost(
            seq_len=self.config.max_seq_len,
            batch_size=1,
            dtype=None,
        )["flops", "matmul"].sum()
        return whole / self.config.max_seq_len


def _value_table_config(
    config: NanoChatLM.Config,
    *,
    width: int,
) -> NarrowEmbedding.Config:
    """Configure one value-embedding table, narrowed like the token table."""
    table = NarrowEmbedding.Config(
        inner=Embedding.Config(init_weight=unit_fan_in_uniform),
    )
    table.channels_out = width
    table.channels_in = config.vocab_size
    table.dtype = config.embedding.dtype
    # Finalized here because it is not in the model's tree: nothing else pushes
    # the width into ``inner``, and an unfinalized table costs a -1 row.
    return table.finalize()


class _BlockCallable(Protocol):
    """Transformer-block call slice used where ``nn.Module`` is untyped."""

    def __call__(
        self,
        x: Tensor,
        *,
        cos_sin: tuple[Tensor, Tensor],
        value_embedding: Tensor | None,
    ) -> Tensor: ...


def _inner_width(block: HasAttention) -> int:
    """Return a block's attention inner width, ``num_heads * channels_head``."""
    geometry = cast(HeadGeometry, block.attn)
    return geometry.num_heads * geometry.channels_head


def _reject_ragged_heads(blocks: Sequence[HasAttention]) -> None:
    """Raise unless every block agrees on its attention's head geometry."""
    shapes = {
        (cast(HeadGeometry, block.attn).channels_head, _inner_width(block))
        for block in blocks
    }
    if len(shapes) > 1:
        raise ValueError(
            "every block must declare the same attention head geometry, since "
            "the value embeddings and rotary factors are shared across layers; "
            f"got (channels_head, num_heads * channels_head) of {sorted(shapes)}.",
        )


class OutputNormFeedForward(SwiGLUReluSquared):
    """Apply an injected transform to the FFN output; identity by default."""

    class Config(  # ty: ignore[inconsistent-mro] -- Makes re-parents make(); ty cannot model configgle's metaclass.  # pyright: ignore[reportGeneralTypeIssues] -- Makes re-parents make(); pyright cannot model configgle's metaclass.
        Makes["OutputNormFeedForward"],
        SwiGLUReluSquared.Config,
    ):
        norm_out: Makeable[TensorModule] = field(
            default_factory=Identity.Config,
        )
        """Parameter-free normalization of the output projection."""

        @override
        def finalize(self) -> Self:
            propagate_attr(
                self.norm_out,
                "channels_in",
                self.channels_out if self.channels_out > 0 else self.channels_in,
            )
            SwiGLUReluSquared.Config.finalize(self)
            return self

        @override
        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the feed-forward and its output normalization.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            return SwiGLUReluSquared.Config.cost(
                self,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + cost(
                self.norm_out,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.norm_out = config.norm_out.make()

    @override
    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.norm_out.reset_parameters()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        return self.norm_out(super().forward(x, **kwargs))


class ScaledSoftCap(SoftCap):
    """Cap in the projection dtype, then return float32 loss/evaluation logits."""

    class Config(Makes["ScaledSoftCap"], SoftCap.Config):
        output_cap: float = 15.0
        """Output amplitude, independent of the inherited input divisor ``cap``."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.output_cap = config.output_cap

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        return (
            self.output_cap * torch.tanh(self.inner(x, **kwargs) / self.cap)
        ).float()


class GatedResidualMix(ResidualMix):
    """Modulate the original-input skip by the running stream's mean."""

    class Config(Makes["GatedResidualMix"], ResidualMix.Config):
        gate_scale: float = 0.0
        """Initial gate scale; zero gives a neutral multiplier of one."""

        @override
        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the mean gate and its scalar parameter at every layer.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            base = super().cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            dt = dtype
            gate = (
                elementwise_cost(
                    primal=8 * seq_len * batch_size,
                    adjoint=8 * seq_len * batch_size,
                    channels=1,
                    inputs=6,
                    outputs=5,
                    adjoint_inputs=11,
                    adjoint_outputs=5,
                    params=1,
                    rows=seq_len * batch_size,
                    dtype=dt,
                )
                + elementwise_cost(
                    primal=0,
                    adjoint=2 * self.channels_in * seq_len * batch_size,
                    channels=self.channels_in,
                    inputs=0,
                    outputs=0,
                    adjoint_inputs=2,
                    adjoint_outputs=2,
                    rows=seq_len * batch_size,
                    dtype=dt,
                )
                + reduction_cost(
                    input_elements=self.channels_in * seq_len * batch_size,
                    output_groups=seq_len * batch_size,
                    dtype=dt,
                )
            )
            return base + gate.tile(self.num_layers, copies=self.num_layers)

    def __init__(self, config: Config) -> None:
        # The parent's constructor invokes reset before this added vector exists.
        nn.Module.__init__(self)
        self.config = config
        self.running = nn.Parameter(torch.empty(config.num_layers))
        self.original = nn.Parameter(torch.empty(config.num_layers))
        self.gate_scales = nn.Parameter(torch.empty(config.num_layers))
        self.gate_scale = config.gate_scale
        self.reset_parameters()

    @override
    def reset_parameters(self) -> None:
        super().reset_parameters()
        nn.init.constant_(self.gate_scales, self.gate_scale)

    @override
    def forward(
        self,
        x: Tensor,
        *,
        original: Tensor,
        layer: int,
        **kwargs: object,
    ) -> Tensor:
        """Mix the current residual with a gated original-input skip.

        Args:
          x: Current residual stream.
          original: Initial embedding stream with the same shape.
          layer: Index selecting the mixing coefficients.
          **kwargs: Unused model messages.

        Returns:
          mixed: Gated residual mixture with the shape of ``x``.

        """
        del kwargs
        gate = 2 * torch.sigmoid(
            self.gate_scales[layer] * x.float().mean(-1, keepdim=True),
        ).to(x.dtype)
        return self.running[layer] * x + self.original[layer] * gate * original


class SourceReuseTransformerBlock(TransformerBlock):
    """Read attention from a saved stream while keeping the current residual."""

    class Config(
        Makes["SourceReuseTransformerBlock"],
        TransformerBlock.Config,
    ):
        """Configure a prenorm block accepting an earlier attention source."""

    @override
    def _forward(self, x: Tensor, **kwargs: object) -> Tensor:
        source = kwargs.pop("attention_source", None)
        if source is None:
            return super()._forward(x, **kwargs)
        if not self.prenorm:
            raise ValueError("Expected self.prenorm.")
        assert isinstance(source, Tensor)
        attention = self.attn(self.norm1(source, **kwargs), **kwargs)
        assert isinstance(attention, Tensor)
        x = x + attention
        return x + self.ffn(self.norm2(x, **kwargs), **kwargs)


class MemoryNanoChatLM(NanoChatLM):
    """Decoder with optional per-layer memory and pooling, disabled by default."""

    class Config(Makes["MemoryNanoChatLM"], NanoChatLM.Config):
        fused_ngram: bool = False
        """Fuse hashed value gathers and accumulate table gradients directly in FP32."""

        ngram_dirty_clear: bool = False
        """Mark touched sink rows in the backward and clear only those rows."""

        bigrams: dict[str, HashedNgramTables.Config] = field(
            default_factory=dict[str, HashedNgramTables.Config],
        )
        """Bigram table configs, keyed by the receiving layer index."""

        trigrams: dict[str, HashedNgramTables.Config] = field(
            default_factory=dict[str, HashedNgramTables.Config],
        )
        """Trigram table configs, keyed by the receiving layer index."""

        num_pool_layers: int = 1
        """Final layers pooled; the last is the unweighted residual stream."""

        dtype: torch.dtype | None = None
        """Optional whole-model storage dtype; None preserves component dtypes."""

        attention_source_layers: tuple[int, ...] = ()
        """Layers whose attention reads the saved earlier block output."""

        attention_source_after_layer: int = 0
        """Save the residual after this zero-based layer, without detaching it."""

        @override
        def finalize(self) -> Self:
            if self.num_pool_layers < 1 or self.num_pool_layers > self.num_layers:
                raise ValueError("num_pool_layers must be between one and num_layers.")
            if self.attention_source_layers and (
                self.attention_source_after_layer < 0
                or any(
                    layer <= self.attention_source_after_layer
                    or layer >= self.num_layers
                    for layer in self.attention_source_layers
                )
            ):
                raise ValueError("Attention source must precede every receiving layer.")
            return super().finalize()

        @override
        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost inherited decoding, additional lookup tables, and layer pooling.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            batch = kwargs
            rows = seq_len * batch_size
            dt = dtype
            total = super().cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **batch,
            )
            for table in (*self.bigrams.values(), *self.trigrams.values()):
                total += cost(
                    table,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **batch,
                )
            pooled = self.num_pool_layers - 1
            if pooled:
                c = self.channels_in
                total += (
                    traffic(
                        "primal",
                        "elementwise",
                        elements=5 * c * rows + 1,
                        flops=2 * c * rows,
                        dtype=dt,
                    )
                    + traffic(
                        "adjoint",
                        "elementwise",
                        elements=6 * c * rows + 1,
                        flops=2 * c * rows,
                        dtype=dt,
                    )
                    + reduction_cost(
                        input_elements=c * rows,
                        output_groups=rows,
                        dtype=dt,
                        phase="adjoint",
                    )
                    + reduction_cost(
                        input_elements=rows,
                        output_groups=1,
                        dtype=dt,
                        phase="adjoint",
                    )
                    + Cost(params=1, params_active=1)
                ).tile(pooled, copies=pooled)
            return total

        @override
        def _propagate_layer_table_widths(self) -> None:
            assert isinstance(self.block, Sequence)
            for name, table in (*self.bigrams.items(), *self.trigrams.items()):
                layer = int(name)
                if layer < 0 or layer >= len(self.block):
                    raise ValueError(
                        f"Memory table layer {layer} is outside "
                        f"the {len(self.block)}-layer model.",
                    )
                table.channels_out = _inner_width(self.block[layer])

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.bigrams = nn.ModuleDict(
            {name: cfg.make() for name, cfg in config.bigrams.items()},
        )
        self.trigrams = nn.ModuleDict(
            {name: cfg.make() for name, cfg in config.trigrams.items()},
        )
        self.pool_weights = (
            nn.Parameter(torch.zeros(config.num_pool_layers - 1))
            if config.num_pool_layers > 1
            else None
        )
        self.pool_start = config.num_layers - config.num_pool_layers
        self.dtype = config.dtype
        self.attention_source_layers = config.attention_source_layers
        self.attention_source_after_layer = config.attention_source_after_layer
        self.fused_ngram = config.fused_ngram
        self.ngram_dirty_clear = config.ngram_dirty_clear
        self._finalize_storage()

    @override
    def reset_parameters(self) -> None:
        if self.dtype is not None:
            self.float()
        super().reset_parameters()
        device = self._materialized_device()
        if device is not None:
            self.materialize_rotation_table(device=device)
        for table in self._tables():
            table.reset_parameters()
        if self.pool_weights is not None:
            nn.init.zeros_(self.pool_weights)
        self._finalize_storage()

    def _finalize_storage(self) -> None:
        """Apply storage dtype and allocate fused buffers without redrawing state."""
        if self.dtype is not None:
            self.to(dtype=self.dtype)
        if self.fused_ngram:
            for table in self._tables():
                table.prepare_gradient_sinks(dirty_bitmaps=self.ngram_dirty_clear)

    def _tables(self) -> list[HashedNgramTables]:
        """Every n-gram table, bigrams then trigrams; ``ModuleDict`` erases the type."""
        tables: list[HashedNgramTables] = []
        for table in (*self.bigrams.values(), *self.trigrams.values()):
            assert isinstance(table, HashedNgramTables)
            tables.append(table)
        return tables

    @override
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear parameter gradients and persistent FP32 table-gradient buffers.

        Call after the optimizer consumes the buffers. Selective clearing also resets
        the row flags used by sparse updates.

        Args:
          set_to_none: Drop parameter gradients instead of zeroing them.

        """
        super().zero_grad(set_to_none=set_to_none)
        for table in self._tables():
            if self.ngram_dirty_clear and table.gradient_bitmaps:
                clear_marked_sinks(table.gradient_sinks, table.gradient_bitmaps)
                continue
            for sink in table.gradient_sinks:
                sink.zero_()

    @override
    def forward(self, tokens: Tensor, *args: object, **kwargs: object) -> Tensor:
        """Compute causal token logits with configured memories and layer pooling.

        Args:
          tokens: Token IDs with sequence on the last axis.
          *args: Unused model arguments.
          **kwargs: Unused model messages.

        Returns:
          logits: Token logits with vocabulary on the last axis.

        """
        del args, kwargs
        length = tokens.shape[-1]
        if length > self.config.max_seq_len:
            raise ValueError(
                f"Input length {length} exceeds max_seq_len={self.config.max_seq_len}.",
            )
        cos_sin = self._rotation_table(length, device=tokens.device)
        x = self.norm_embed(self.embed(tokens))
        original = x
        attention_source = x
        pooled: Tensor | None = None
        for layer, block in enumerate(self.blocks):
            name = str(layer)
            x = self.mix(x, original=original, layer=layer)
            messages: dict[str, object] = {"cos_sin": cos_sin}
            if name in self.value_embeds:
                messages["value_embedding"] = self.value_embeds[name](tokens)
            if layer in self.attention_source_layers:
                messages["attention_source"] = attention_source
            if self.fused_ngram:
                if name in self.bigrams or name in self.trigrams:
                    messages["fused_tables"] = self._fused_sources(name, tokens)
            else:
                if name in self.bigrams:
                    messages["bigram_value"] = self.bigrams[name](tokens)
                if name in self.trigrams:
                    messages["trigram_value"] = self.trigrams[name](tokens)
            x = cast(Tensor, block(x, **messages))
            if layer == self.attention_source_after_layer:
                attention_source = x
            if (
                self.pool_weights is not None
                and self.pool_start <= layer < len(self.blocks) - 1
            ):
                weighted = self.pool_weights[layer - self.pool_start] * x
                pooled = weighted if pooled is None else pooled + weighted
        if pooled is not None:
            x = x + pooled
        return self.lm_head(self.norm_out(x))

    def materialize_rotation_table(self, *, device: torch.device) -> None:
        """Build rotary factors on the model device before compilation.

        Allocating inside compilation risks CUDA-graph storage reuse. Moving factors
        between devices preserves the source device's transcendental rounding.

        Args:
          device: Materialized model device.

        """
        positions = torch.arange(self.config.max_seq_len, device=device)
        self._rotation = self.rope(positions)

    @override
    def _rotation_table(
        self,
        length: int,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Slice rotary factors, rebuilding outside compilation on device changes."""
        if (
            self._rotation is None
            or self._rotation[0].device != device
            or self._rotation[0].dtype != self.rope.dtype
        ):
            if torch.compiler.is_compiling():
                raise RuntimeError(
                    "The rotation table must be materialized outside compilation. "
                    "Call materialize_rotation_table(device=...) before the "
                    "first compiled forward.",
                )
            # Recompute on the target device; moving factors retains different
            # transcendental rounding.
            self.materialize_rotation_table(device=device)
        if self._rotation is None:
            raise ValueError("Expected self._rotation is not None.")
        cos, sin = self._rotation
        return cos[:length], sin[:length]

    @override
    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        """Rebuild factors after the rope's device or dtype changes."""
        super()._apply(fn, recurse)
        # Plain tuples are not moved by Module._apply. Whole-model precision
        # casts to float before base reset and back to BF16 afterwards.
        device = self._materialized_device()
        if device is None:
            self._rotation = None
        else:
            self.materialize_rotation_table(device=device)
        return self

    def _materialized_device(self) -> torch.device | None:
        """Return the first parameter's device, excluding meta tensors."""
        for tensor in self.parameters():
            return None if tensor.device.type == "meta" else tensor.device
        return None

    def _fused_sources(self, name: str, tokens: Tensor) -> list[NgramSource]:
        sources: list[NgramSource] = []
        for gate_index, tables in ((1, self.bigrams), (2, self.trigrams)):
            if name in tables:
                table = tables[name]
                assert isinstance(table, HashedNgramTables)
                sources.append(NgramSource(gate_index, table, table.indices(tokens)))
        return sources
