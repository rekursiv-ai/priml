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
from typing import Any, Self, override

import math

from configgle import Fig, Makeable
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

    The gate exists on every layer even though only some receive a value
    embedding: it is ``heads * gate_channels`` parameters, and making its
    presence conditional would put a layer-index policy inside a module that
    does not otherwise know where it sits.
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
        self.value_gate = Linear.Config(
            channels_in=config.gate_channels,
            channels_out=config.heads,
            bias=False,
            init_weight=nn.init.zeros_,
        ).make()
        self.norm_q = config.norm_qk.make()
        self.norm_k = config.norm_qk.make()

    def reset_parameters(self) -> None:
        """Re-initialize every projection."""
        for module in (self.proj_q, self.proj_k, self.proj_v, self.proj_out):
            module.reset_parameters()
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
            gate = 2 * torch.sigmoid(self.value_gate(x[..., : config.gate_channels]))
            v = v + gate.unsqueeze(-1) * value_embedding.view(shape)
        cos, sin = cos_sin
        q, k = RoPE.rotate(q, k, cos, sin)
        q, k = self.norm_q(q), self.norm_k(k)
        # SDPA wants [..., heads, S, channels_head].
        q, k, v = (t.movedim(-3, -2) for t in (q, k, v))
        mask = _window_mask(q.shape[-2], window=window, device=q.device)
        out = functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=mask is None,
        )
        return self.proj_out(out.movedim(-3, -2).flatten(-2))


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

        value_embedding_layers: list[int] = field(default_factory=list[int])
        """Layers reading a value embedding; empty gives none of them one.

        Listed rather than derived from a stride, because which layers get one
        is the experimental question: the embedding is a path from the raw
        tokens to the output, so it is worth the most where the residual
        stream is most processed and the answer is not a formula."""

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

        rope: RoPE.Config = field(default_factory=RoPE.Config)
        """Rotary position embedding driving every layer's queries and keys.

        Its width is pushed down from the block's attention at finalize, since
        the model builds the factors and the heads consume them."""

        logit_softcap: float = 15.0
        """Logit bound: ``cap * tanh(logits / cap)``."""

        residual_init: float = 1.0
        """Initial per-layer weight on the running residual stream."""

        skip_init: float = 0.1
        """Initial per-layer weight on the original token embedding."""

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
            for layer, block in enumerate(self.blocks):
                propagate_attr(block, "channels_in", self.channels, protocol=ChannelsIn)
                propagate_attr(block, "depth", layer)
            # The value-embedding tables and the rotary factors are built ONCE,
            # to layer 0's geometry, then handed to every layer -- so a stack
            # whose layers disagree on head shape is a contradiction fixed at
            # construction. Left to run time it surfaces as a reshape failure
            # inside the forward, naming a size rather than the layer.
            _reject_ragged_heads(self.blocks, channels=self.channels)
            propagate_attr(self.embedding, "channels_out", self.channels)
            propagate_attr(self.embedding, "num_embeddings", self.vocab_size)
            propagate_attr(self.lm_head, "channels_in", self.channels)
            propagate_attr(self.lm_head, "channels_out", self.vocab_size)
            self.rope.channels_head = head_width(self.blocks[0], self.channels)
            return super().finalize()

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
        attention = sum(
            12 * inner * min(window, self.config.max_seq_len) for window in self.windows
        )
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


def _window_mask(length: int, *, window: int, device: torch.device) -> Tensor | None:
    """A causal mask restricted to ``window`` positions of history.

    Returns ``None`` for a window covering the whole sequence, so the caller's
    ``is_causal`` fast path applies instead of a materialized mask.
    """
    if window < 0 or window >= length - 1:
        return None
    offset = torch.arange(length, device=device)
    offset = offset[:, None] - offset[None, :]
    return (offset >= 0) & (offset <= window)


def _rms_norm(x: Tensor) -> Tensor:
    """Parameter-free RMS normalization over the final dimension."""
    return functional.rms_norm(x, (x.shape[-1],))


def _weight(module: nn.Module) -> Tensor:
    """The ``weight`` tensor of an embedding-like module."""
    weight = module.weight
    assert isinstance(weight, Tensor)
    return weight
