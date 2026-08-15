"""A decoder-only language model: embed, blocks, project to the vocabulary.

The block stack is priml's :class:`~priml.model.transformer.TransformerBlock`
with two sublayers supplied here, because both differ from priml's defaults in
ways that are the recipe rather than a preference:

* :class:`ReluSquared` -- a feed-forward whose nonlinearity is ``relu(x)**2``.
  It has no gate, so it is one matrix in and one out where SwiGLU is three.
* :class:`ValueGatedAttention` -- causal attention over a sliding WINDOW, with
  parameter-free norm on the queries and keys, and a per-head gate admitting a
  value embedding the model supplies.

The model itself owns three things a block cannot: the window pattern (a
property of a layer's position in the stack), the value-embedding tables (one
per participating layer, read from the input tokens directly), and the pair of
per-layer scalars mixing the running residual stream with the original
embedding. That last pair is what lets a deep stack keep a path back to the
input without a skip connection per layer.

Logits are soft-capped -- ``cap * tanh(logits / cap)`` -- which bounds the
gradient a single confident token can contribute and is what keeps the run
stable at a large learning rate.

References:
    https://arxiv.org/abs/2109.08668
      So et al. Primer: Searching for Efficient Transformer for Language
      Modeling. (The squared-ReLU feed-forward.)
    https://arxiv.org/abs/2410.17897
      Zhou et al. Value Residual Learning.
    https://arxiv.org/abs/2104.09864
      Su et al. RoFormer: Enhanced Transformer with Rotary Position Embedding.

"""

from __future__ import annotations

from dataclasses import field
from functools import partial
from typing import Any, Protocol, Self, override, runtime_checkable

import math

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.model.custom_types import ChannelsIn, TensorModule, propagate_attr
from priml.model.embedding import Embedding
from priml.model.init import InitFn, normal
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.rope import RoPE
from priml.model.transformer import TransformerBlock


@runtime_checkable
class AttentionKernel(Protocol):
    """Windowed causal attention over ``[B, S, heads, channels_head]``."""

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, *, window: int) -> Tensor: ...


def sdpa_attention(q: Tensor, k: Tensor, v: Tensor, *, window: int) -> Tensor:
    """Windowed causal attention through torch's dispatcher.

    Runs anywhere, which is what makes it the default. The cost is that a
    windowed layer has to say so with an explicit mask, and the flash backend
    refuses a mask -- so windowed layers land on the memory-efficient kernel
    while global ones reach flash. ``Flash3Attention`` expresses the same
    window as a kernel argument and keeps every layer on one kernel.

    Args:
      q: ``[B, S, heads, channels_head]`` queries.
      k: Keys, same shape.
      v: Values, same shape.
      window: Positions each query may look back over, itself included.

    Returns:
      out: Attention output, same shape as ``q``.

    """
    # SDPA wants [..., heads, S, channels_head].
    q, k, v = (t.movedim(-3, -2) for t in (q, k, v))
    length = q.shape[-2]
    if window >= length:
        out = functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        offset = torch.arange(length, device=q.device)
        offset = offset[:, None] - offset[None, :]
        # ``<=``, not ``<``: a fused kernel's ``window_size=(w, 0)`` admits w
        # tokens of history IN ADDITION to the query's own position, so the
        # exclusive form attends to one key fewer per row and is a different
        # model -- not a rounding difference.
        out = functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=(offset >= 0) & (offset <= window),
        )
    return out.movedim(-3, -2)


def apply_rotary(x: Tensor, *, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate the channel pairs of ``x`` by the angles ``cos``/``sin`` encode.

    Spelled here rather than delegating to
    :meth:`~priml.model.rope.RoPE.rotate`, for two reasons that are both
    numerics rather than taste:

    * **Direction.** priml's rotation is HuggingFace's ``+theta``; this recipe's
      is ``-theta`` (the sine enters the second half negated). The two are the
      same model under a channel permutation and DIFFERENT tensors, so a
      reproduction has to pick the one the weights were trained under.
    * **Precision.** priml's rotation upcasts to float32, accumulates there, and
      rounds once. This one accumulates in whatever precision the factors and
      the input arrive in -- half, under autocast -- which is what the fused
      reference kernel does, and differs from the upcast form in the last bits.

    Args:
      x: ``[..., S, heads, channels_head]`` queries or keys.
      cos: Cosine factors, broadcastable over ``x``'s first half.
      sin: Sine factors, same shape as ``cos``.

    Returns:
      rotated: ``x`` with each ``(i, i + half)`` channel pair rotated.

    """
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return torch.cat(
        [first * cos + second * sin, first * (-sin) + second * cos],
        dim=-1,
    )


def unit_fan_in_uniform(w: Tensor, *, depth: int = -1) -> None:
    """Uniform on ``+-sqrt(3 / fan_in)``, realizing a ``1/sqrt(fan_in)`` std.

    The initialization every projection in this baseline uses. ``depth`` is
    accepted and discarded: priml's layers pass it to every ``init_weight`` so
    depth-scaled schemes can use it, and this one does not scale with depth.

    Args:
      w: Tensor to initialize in place.
      depth: Ignored; present for the ``InitFn`` protocol.

    """
    del depth
    bound = 3**0.5 * w.shape[-1] ** -0.5
    nn.init.uniform_(w, -bound, bound)


class ReluSquared(nn.Module):
    """Feed-forward with a squared-ReLU nonlinearity and no gate.

    The output projection is zero-initialized, so a fresh block is the identity
    on its residual stream: the stack starts as shallow as the task needs and
    deepens as training proceeds.
    """

    class Config(Fig["ReluSquared"]):
        """Width, expansion, and initialization."""

        channels_in: int = -1
        """Model width; -1 inherits from the block."""

        expansion: int = 4
        """Hidden width as a multiple of ``channels_in``."""

        bias: bool = False
        """Include bias in both projections."""

        init_weight: InitFn = unit_fan_in_uniform
        """Initialization for the input projection."""

        depth: int = -1
        """Block depth index, accepted for the priml block contract."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.channels_in <= 0:
            raise ValueError(
                f"channels_in must be positive; got {config.channels_in}. It is "
                "normally inherited from the block during finalize.",
            )
        hidden = config.expansion * config.channels_in
        self.up_proj = Linear.Config(
            channels_in=config.channels_in,
            channels_out=hidden,
            bias=config.bias,
            init_weight=config.init_weight,
        ).make()
        self.down_proj = Linear.Config(
            channels_in=hidden,
            channels_out=config.channels_in,
            bias=config.bias,
            init_weight=nn.init.zeros_,
        ).make()

    def reset_parameters(self) -> None:
        """Re-initialize both projections."""
        self.up_proj.reset_parameters()
        self.down_proj.reset_parameters()

    @override
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        return self.down_proj(functional.relu(self.up_proj(x)).square())


class ValueGatedAttention(nn.Module):
    """Windowed causal attention with normalized queries/keys and value gating.

    Two departures from priml's
    :class:`~priml.model.attention.SelfAttention`, both load-bearing here:

    * **A window.** A layer attends only to the last ``window`` positions.
      Restricting most layers and leaving a few global keeps attention
      affordable at long context while preserving a path to any position.
    * **A value gate.** When the caller supplies a value embedding, each head
      admits it through a learned scalar read from the first few channels of
      the layer's own input, so a head decides per token how much to consult
      the raw token identity rather than the processed stream.

    Whether a layer HAS a gate is declared by ``gated``, which the model sets
    from its value-embedding layer set. A gate on a layer that receives no
    embedding is never read, so it would sit in the optimizer's matrix group
    collecting weight decay and contributing nothing -- the parameter count and
    the partition would both differ from a recipe that omits it, which is a
    difference a reproduction cannot carry.
    """

    class Config(Fig["ValueGatedAttention"]):
        """Head geometry, the gate width, and the injected norm."""

        channels_in: int = -1
        """Model width; -1 inherits from the block."""

        heads: int = -1
        """Attention heads; -1 derives from ``channels_in // channels_head``."""

        channels_head: int = 128
        """Per-head width."""

        gate_channels: int = 32
        """Input channels feeding the value gate, clamped to the model width."""

        norm_qk: Makeable[TensorModule] = field(
            default_factory=lambda: RMSNorm.Config(elementwise_affine=False),
        )
        """Normalization applied to queries and keys before attention.

        Parameter-free by default: bounding the logits' scale is its whole job
        here, and a learned gain would duplicate the projection that produced
        them."""

        init_weight: InitFn = unit_fan_in_uniform
        """Initialization for the query, key, and value projections."""

        kernel: Makeable[AttentionKernel] = field(
            default_factory=lambda: PartialConfig(sdpa_attention),
        )
        """The attention kernel itself, injected rather than selected.

        A kernel is a different VALUE in this slot, not a mode flag: the
        reference recipe measured its score on FlashAttention-3, and a fused
        kernel reduces in a different order than a masked SDPA, so reproducing
        that number means issuing that kernel. The default runs anywhere; a
        rung reproducing a published result pins the one it was published
        with, and inherits its hardware requirement along with it."""

        gated: bool = True
        """Whether this layer builds the value gate at all.

        Set by the model from its value-embedding layer set: a layer that
        receives no embedding never reads the gate, so building one would add a
        parameter that trains on nothing and shifts the optimizer's partition
        away from the recipe being reproduced."""

        depth: int = -1
        """Block depth index, accepted for the priml block contract."""

        @override
        def finalize(self) -> Self:
            if self.channels_head <= 0 or self.channels_head % 2:
                raise ValueError(
                    "channels_head must be positive and even; got "
                    f"{self.channels_head}.",
                )
            if self.heads == -1:
                if self.channels_in % self.channels_head:
                    raise ValueError(
                        f"channels_in={self.channels_in} is not divisible by "
                        f"channels_head={self.channels_head}; set heads.",
                    )
                self.heads = self.channels_in // self.channels_head
            self.gate_channels = min(self.gate_channels, self.channels_in)
            # The norm sees one HEAD, not the residual stream, so it takes the
            # head width rather than the model width the block propagated.
            propagate_attr(
                self.norm_qk,
                "channels_in",
                self.channels_head,
                protocol=ChannelsIn,
            )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        if min(config.channels_in, config.heads, config.gate_channels) <= 0:
            raise ValueError(
                "channels_in, heads, and gate_channels must be positive; they "
                "are normally inherited from the block during finalize. Got "
                f"{config.channels_in}, {config.heads}, {config.gate_channels}.",
            )
        self.config = config
        inner = config.heads * config.channels_head
        projection = Linear.Config(
            channels_in=config.channels_in,
            channels_out=inner,
            bias=False,
            init_weight=config.init_weight,
        )
        self.proj_q = projection.copy_tree().make()
        self.proj_k = projection.copy_tree().make()
        self.proj_v = projection.copy_tree().make()
        self.proj_out = Linear.Config(
            channels_in=inner,
            channels_out=config.channels_in,
            bias=False,
            init_weight=nn.init.zeros_,
        ).make()
        # Zero-initialized: ``2 * sigmoid(0)`` is exactly 1, so a fresh gate
        # admits the value embedding unchanged and must learn to attenuate it.
        self.value_gate = (
            Linear.Config(
                channels_in=config.gate_channels,
                channels_out=config.heads,
                bias=False,
                init_weight=nn.init.zeros_,
            ).make()
            if config.gated
            else None
        )
        self.norm_q = config.norm_qk.make()
        self.norm_k = config.norm_qk.make()
        # Resolved once, here: a pinned kernel validates a built artifact and
        # the device it will run on, and doing that per layer per step would
        # pay for the check every forward.
        self.attention = config.kernel.make()

    def reset_parameters(self) -> None:
        """Re-initialize every projection."""
        for module in (self.proj_q, self.proj_k, self.proj_v, self.proj_out):
            module.reset_parameters()
        if self.value_gate is not None:
            self.value_gate.reset_parameters()

    @override
    def forward(
        self,
        x: Tensor,
        *args: Any,
        cos_sin: tuple[Tensor, Tensor],
        value_embedding: Tensor | None = None,
        window: int = -1,
        **kwargs: Any,
    ) -> Tensor:
        """Attend over the last ``window`` positions.

        Args:
          x: ``[B, S, C]`` layer input.
          *args: Ignored; present for the block's call contract.
          cos_sin: Rotary ``(cos, sin)`` covering ``S`` positions.
          value_embedding: ``[B, S, heads * channels_head]`` added to the
            values through the per-head gate, or ``None`` for this layer.
          window: Positions each query may look back over; -1 is the full
            context.
          **kwargs: Ignored; present for the block's call contract.

        Returns:
          out: ``[B, S, C]`` attention output.

        """
        del args, kwargs
        config = self.config
        shape = (*x.shape[:-1], config.heads, config.channels_head)
        q = self.proj_q(x).view(shape)
        k = self.proj_k(x).view(shape)
        v = self.proj_v(x).view(shape)
        if value_embedding is not None:
            # A layer handed an embedding must have been built with a gate; the
            # model derives both from one layer set, so the absence of one is a
            # wiring error rather than a case to fall back from.
            assert self.value_gate is not None
            gate = 2 * torch.sigmoid(self.value_gate(x[..., : config.gate_channels]))
            v = v + gate.unsqueeze(-1) * value_embedding.view(shape)
        cos, sin = cos_sin
        q = apply_rotary(q, cos=cos, sin=sin)
        k = apply_rotary(k, cos=cos, sin=sin)
        q, k = self.norm_q(q), self.norm_k(k)
        # Left in [B, S, heads, channels_head]: that is what the pinned kernel
        # takes, and the portable one transposes internally. Transposing here
        # instead would make the fused path pay to undo it.
        out = self.attention(q, k, v, window=window if window > 0 else q.shape[-3])
        # Made contiguous before the head axes are merged. A portable kernel
        # returns a transposed view, and flattening that directly hands the
        # projection a strided tensor -- the matmul then reduces in a different
        # order than it does over a packed one, and the difference reaches this
        # layer's gradient. Copying once here costs a layer and buys an
        # ordering that does not depend on which kernel produced the input.
        return self.proj_out(out.contiguous().flatten(-2))


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

        channels: int = 512
        """Model width, and the width every block inherits."""

        num_layers: int = 8
        """Blocks in the stack."""

        block: TransformerBlock.Config = field(
            default_factory=lambda: TransformerBlock.Config(
                attn=ValueGatedAttention.Config(),
                ffn=ReluSquared.Config(),
                norm1=RMSNorm.Config(elementwise_affine=False),
                norm2=RMSNorm.Config(elementwise_affine=False),
            ),
        )
        """Block template, copied once per layer.

        Any module accepting ``(x, cos_sin=..., value_embedding=...,
        window=...)`` works, so changing the architecture is a value here
        rather than an edit to this class."""

        blocks: list[Makeable[nn.Module]] = field(
            default_factory=list[Makeable[nn.Module]],
        )
        """Per-layer blocks; empty copies ``block`` ``num_layers`` times."""

        window_pattern: str = "SSSL"
        """Cycled per-layer attention windows: L is the full context, S half.

        The final layer is always long regardless of the pattern -- it is the
        one that has to see the whole sequence to predict the next token."""

        value_embedding_stride: int = 0
        """Every Nth layer reads a value embedding, counting BACK from the last.

        0 gives none of them one. Counting back so the deepest layer always
        gets a table: the embedding is a path from the raw tokens to the
        output, worth the most where the residual stream is most processed.

        A STRIDE rather than a list of indices, because indices computed
        against one depth go stale the moment a fork changes ``num_layers`` --
        the list is a snapshot, this is the rule that produced it. Set
        ``value_embedding_layers`` directly to name specific layers."""

        value_embedding_layers: list[int] = field(default_factory=list[int])
        """Layers reading a value embedding; empty derives them from the stride.

        Set this to override the stride with an explicit set."""

        embedding: Makeable[nn.Module] = field(default_factory=Embedding.Config)
        """Token embedding table."""

        lm_head: Makeable[TensorModule] = field(
            default_factory=lambda: Linear.Config(
                bias=False,
                # Near-zero, unlike a hidden layer: the first steps' logits are
                # then near-uniform, so early gradients teach the body rather
                # than a confident wrong readout.
                init_weight=partial(normal, std=0.001),
            ),
        )
        """Output projection to vocabulary logits."""

        rope: RoPE.Config = field(
            # The reference builds its frequencies as ``1 / base**(i / d)`` over
            # an integer range. priml's default spells the same function as an
            # exponentiated float64 linspace, which is more accurate and NOT the
            # same bits -- so the reproduction takes the reference's spelling.
            default_factory=lambda: RoPE.Config(hf_inv_freq=True),
        )
        """Rotary position embedding driving every layer's queries and keys.

        Its width is pushed down from the block's attention at finalize, since
        the model builds the factors and the heads consume them."""

        rotary_dtype: torch.dtype | None = None
        """Dtype the rotary factors are rounded to before the rotation.

        Not a memory choice -- the table is two vectors -- but an arithmetic
        one: rounding the factors makes every product inside the rotation
        accumulate at that width, where leaving them in float32 promotes the
        whole rotation and rounds once at the end. The two differ in the last
        bits of every query and key, so a reproduction has to hold the factors
        at the width the reference held them -- ``exp000`` sets bfloat16.

        ``None`` by default, with :attr:`embedding_dtype`: the two are one
        decision, and a model narrowed by default cannot run outside autocast
        at all (a bfloat16 stream meets a float32 projection and the matmul
        refuses), so the default has to be the portable one and the recipe has
        to say what it narrowed."""

        logit_softcap: float = 15.0
        """Logit bound: ``cap * tanh(logits / cap)``."""

        residual_init: float = 1.0
        """Initial per-layer weight on the running residual stream."""

        skip_init: float = 0.1
        """Initial per-layer weight on the original token embedding."""

        embedding_dtype: torch.dtype | None = None
        """Dtype the token and value-embedding tables are held in.

        A lookup is a gather, not an arithmetic reduction, so half precision
        costs a table nothing it was using -- while halving the largest tensors
        in the model and the traffic to read them. ``exp000`` sets bfloat16
        because the recipe does.

        ``None`` by default rather than bfloat16, even though every budgeted
        run narrows it: a narrowed table makes the model runnable ONLY under
        autocast, since the half-precision stream it emits meets a float32
        projection one layer later. That is a property of the recipe's training
        loop, not of the architecture, so it belongs in the experiment that
        declares the loop."""

        @override
        def finalize(self) -> Self:
            if self.vocab_size <= 0:
                raise ValueError(f"vocab_size must be positive; got {self.vocab_size}.")
            if self.max_seq_len < 2:
                raise ValueError(
                    f"max_seq_len must be at least two; got {self.max_seq_len}.",
                )
            if self.num_layers <= 0 or self.channels <= 0:
                raise ValueError(
                    "num_layers and channels must be positive; got "
                    f"{self.num_layers} and {self.channels}.",
                )
            if not math.isfinite(self.logit_softcap) or self.logit_softcap <= 0:
                raise ValueError(
                    "logit_softcap must be finite and positive; got "
                    f"{self.logit_softcap}.",
                )
            if not self.window_pattern or set(self.window_pattern.upper()) - set("SL"):
                raise ValueError(
                    "window_pattern must hold only S and L; got "
                    f"{self.window_pattern!r}.",
                )
            if not self.blocks:
                self.blocks = [self.block.copy_tree() for _ in range(self.num_layers)]
            if len(self.blocks) != self.num_layers:
                raise ValueError(
                    f"blocks names {len(self.blocks)} layers for "
                    f"num_layers={self.num_layers}.",
                )
            if self.value_embedding_stride < 0:
                raise ValueError(
                    "value_embedding_stride must be nonnegative; got "
                    f"{self.value_embedding_stride}.",
                )
            # Two ways to say one thing, so saying both is a contradiction
            # rather than a precedence rule nobody can see in the print.
            if self.value_embedding_layers and self.value_embedding_stride:
                raise ValueError(
                    "set value_embedding_stride or value_embedding_layers, not "
                    f"both; got stride {self.value_embedding_stride} beside "
                    f"{self.value_embedding_layers}.",
                )
            # Derived HERE, where the depth is final: computing the indices in
            # an experiment factory snapshots whatever num_layers was then, so
            # a fork that changes the depth carries indices for a stack that no
            # longer exists.
            if self.value_embedding_stride:
                self.value_embedding_layers = sorted(
                    range(
                        self.num_layers - 1,
                        -1,
                        -self.value_embedding_stride,
                    ),
                )
                # Consumed, not kept: leaving it set would leave the finalized
                # config in the very both-set state the check above calls a
                # contradiction, and a reader could not tell a derived list
                # from one someone wrote.
                self.value_embedding_stride = 0
            if any(
                not 0 <= layer < self.num_layers
                for layer in self.value_embedding_layers
            ):
                raise ValueError(
                    f"value_embedding_layers {self.value_embedding_layers} names "
                    f"a layer outside the {self.num_layers}-layer stack.",
                )
            # The template stays a field, so the cascade finalizes it too: it
            # needs the width even though the copies are what get built.
            propagate_attr(self.block, "channels_in", self.channels)
            gated = set(self.value_embedding_layers)
            for layer, block in enumerate(self.blocks):
                propagate_attr(block, "channels_in", self.channels, protocol=ChannelsIn)
                propagate_attr(block, "depth", layer)
                # The gate is built only where a table will feed it. Decided
                # HERE because the layer set lives on the model: an attention
                # module does not know where in the stack it sits, and a
                # parameter that no forward reads would still take an optimizer
                # group and a share of the weight decay.
                attention = getattr(block, "attn", None)
                if isinstance(attention, ValueGatedAttention.Config):
                    attention.gated = layer in gated
            propagate_attr(self.embedding, "channels_out", self.channels)
            propagate_attr(self.embedding, "num_embeddings", self.vocab_size)
            propagate_attr(self.lm_head, "channels_in", self.channels)
            propagate_attr(self.lm_head, "channels_out", self.vocab_size)
            self.rope.channels_head = head_width(self.blocks[0], self.channels)
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
            _reject_ragged_heads(finalized.blocks, channels=finalized.channels)
            return finalized

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.windows = window_sizes(
            num_layers=config.num_layers,
            max_seq_len=config.max_seq_len,
            pattern=config.window_pattern,
        )
        # The table is read as the attention's VALUES, so it spans every head,
        # not one: a per-head width would reshape to the wrong sequence length.
        width = attention_width(config.blocks[0], config.channels)
        # Construction order fixes the global-RNG draw order, so a seeded init
        # is reproducible: tokens, head, blocks, value embeddings.
        embedding = config.embedding.make()
        assert isinstance(embedding, nn.Module)
        self.embed = embedding
        # A unit-variance table, against priml's 0.02: the embedding feeds an
        # RMS norm, so its scale is divided out and only its spread survives.
        nn.init.normal_(_weight(self.embed), std=1.0)
        self.lm_head = config.lm_head.make()
        self.blocks = nn.ModuleList([block.make() for block in config.blocks])
        self.value_embeds = nn.ModuleDict(
            {
                str(layer): Embedding.Config(
                    channels_out=width,
                    num_embeddings=config.vocab_size,
                ).make()
                for layer in config.value_embedding_layers
            },
        )
        for table in self.value_embeds.values():
            unit_fan_in_uniform(_weight(table))
        # Cast AFTER initialization so the draws themselves happen in full
        # precision: sampling straight into bfloat16 would quantize every value
        # to its ~3 significant digits and change the table's spread.
        if config.embedding_dtype is not None:
            self.embed.to(dtype=config.embedding_dtype)
            for table in self.value_embeds.values():
                table.to(dtype=config.embedding_dtype)
        self.residual_scale = nn.Parameter(
            torch.full((config.num_layers,), config.residual_init),
        )
        self.skip_scale = nn.Parameter(
            torch.full((config.num_layers,), config.skip_init),
        )
        self.rope = config.rope.make()

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
        if self.config.rotary_dtype is not None:
            cos_sin = (
                cos_sin[0].to(self.config.rotary_dtype),
                cos_sin[1].to(self.config.rotary_dtype),
            )
        # The table's OWN dtype carries the stream, and the per-layer scalars
        # ride along by 0-dim type promotion: a scalar tensor does not widen
        # the tensor it multiplies, so a float32 lambda against a bfloat16
        # stream stays bfloat16 without a cast. Writing the cast out
        # (``lambda.to(x.dtype)``) gives the same forward and a DIFFERENT
        # backward -- the cast is its own autograd node, and the gradient
        # accumulating through it rounds at each layer rather than once.
        x = _rms_norm(self.embed(tokens))
        skip = x
        for layer, block in enumerate(self.blocks):
            x = self.residual_scale[layer] * x + self.skip_scale[layer] * skip
            # ``ModuleDict`` is not a Mapping -- it has no ``get`` -- so
            # membership is tested before the lookup.
            name = str(layer)
            gated = name in self.value_embeds
            out = block(
                x,
                cos_sin=cos_sin,
                value_embedding=self.value_embeds[name](tokens) if gated else None,
                window=self.windows[layer],
            )
            assert isinstance(out, Tensor)
            x = out
        cap = self.config.logit_softcap
        logits = self.lm_head(_rms_norm(x)).float()
        return cap * torch.tanh(logits / cap)

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
            for module in (self.embed, *self.value_embeds.values())
            for parameter in module.parameters()
        }
        gathered |= {id(self.residual_scale), id(self.skip_scale)}
        matrix = sum(
            parameter.numel()
            for parameter in self.parameters()
            if id(parameter) not in gathered
        )
        # ``heads * channels_head``, read from the block rather than derived
        # from the model width: the attention's inner width is decoupled from
        # the residual stream, so dividing would miscount every model where
        # they differ.
        inner = attention_width(self.config.blocks[0], self.config.channels)
        # ``window_sizes`` returns only the full context or half of it, so no
        # window can exceed ``max_seq_len`` and no clamp is needed here.
        assert max(self.windows) <= self.config.max_seq_len
        attention = sum(12 * inner * window for window in self.windows)
        return 6 * matrix + attention


def window_sizes(*, num_layers: int, max_seq_len: int, pattern: str) -> list[int]:
    """Return each layer's attention window from a cycled short/long pattern.

    Args:
      num_layers: Blocks in the stack.
      max_seq_len: Full context length, and the long window.
      pattern: Cycled ``S`` (half context) and ``L`` (full context) symbols.

    Returns:
      windows: One window per layer; the last is always the full context.

    """
    width = {"L": max_seq_len, "S": max_seq_len // 2}
    pattern = pattern.upper()
    windows = [width[pattern[layer % len(pattern)]] for layer in range(num_layers)]
    windows[-1] = max_seq_len
    return windows


def head_width(block: Makeable[nn.Module], channels: int) -> int:
    """The per-head width a block's attention uses.

    The rotary factors are built by the MODEL and consumed per head, so the
    model reads the width off the block it was given rather than assuming its
    own. A block whose attention does not declare one is treated as
    single-head.

    Args:
      block: The block config the model was handed.
      channels: Model width, the single-head fallback.

    Returns:
      width: Per-head channel count.

    """
    attention = getattr(block, "attn", None)
    width = getattr(attention, "channels_head", None)
    return width if isinstance(width, int) and width > 0 else channels


def attention_width(block: Makeable[nn.Module], channels: int) -> int:
    """The width of a block's attention across every head.

    What a value-embedding table must be, since it is added to the VALUES and
    then split per head. Distinct from :func:`head_width`, which sizes the
    rotary factors -- confusing the two builds a table that reshapes to the
    wrong sequence length rather than failing at construction.

    Args:
      block: The block config the model was handed.
      channels: Model width, the fallback when heads are not declared.

    Returns:
      width: ``heads * channels_head``.

    """
    attention = getattr(block, "attn", None)
    heads = getattr(attention, "heads", None)
    width = head_width(block, channels)
    return heads * width if isinstance(heads, int) and heads > 0 else channels


def _reject_ragged_heads(
    blocks: list[Makeable[nn.Module]],
    *,
    channels: int,
) -> None:
    """Raise unless every block agrees on its attention's head geometry.

    Args:
      blocks: The per-layer block configs.
      channels: Model width, the fallback for a block declaring no heads.

    Raises:
      ValueError: Two layers declare different head shapes. The model sizes its
        shared tensors from layer 0, so the disagreement is unbuildable -- and
        naming it here beats a reshape error deep inside the forward.

    """
    shapes = {
        (head_width(block, channels), attention_width(block, channels))
        for block in blocks
    }
    if len(shapes) > 1:
        raise ValueError(
            "every block must declare the same attention head geometry, since "
            "the value embeddings and rotary factors are shared across layers; "
            f"got (channels_head, heads * channels_head) of {sorted(shapes)}.",
        )


def _rms_norm(x: Tensor) -> Tensor:
    """Parameter-free RMS normalization over the final dimension."""
    return functional.rms_norm(x, (x.shape[-1],))


def _weight(module: nn.Module) -> Tensor:
    """The ``weight`` tensor of an embedding-like module."""
    weight = module.weight
    assert isinstance(weight, Tensor)
    return weight
