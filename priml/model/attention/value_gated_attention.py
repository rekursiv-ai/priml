"""Windowed causal attention with a per-head value-embedding gate.

Two departures from :class:`~priml.model.attention.self_attention.SelfAttention`, both
from the speedrun recipes rather than from taste:

* **A window.** A layer attends only to the last ``window`` positions.
  Restricting most layers and leaving a few global keeps attention affordable
  at long context while preserving a path to any position.
* **A value gate.** When the caller supplies a value embedding, each head
  admits it through a learned scalar read from the first few channels of the
  layer's own input, so a head decides per token how much to consult the raw
  token identity rather than the processed stream.

References:
    https://arxiv.org/abs/2410.17897
      Zhou et al. Value Residual Learning.
    https://arxiv.org/abs/2004.05150
      Beltagy et al. Longformer: The Long-Document Transformer.

"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, override

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.model.attention.rope import rotate_conjugate
from priml.model.attention.window import layer_window, window_mask
from priml.model.custom_types import (
    AttentionKernel,
    ChannelsIn,
    DepthIndex,
    TensorModule,
    propagate_attr,
)
from priml.model.init import InitFn, unit_fan_in_uniform
from priml.model.linear import Linear
from priml.model.norm import RMSNorm


def sdpa_attention(q: Tensor, k: Tensor, v: Tensor, *, window: int) -> Tensor:
    """Windowed causal attention through torch's dispatcher.

    Runs anywhere, which is what makes it the default. The cost is that a
    windowed layer has to say so with an explicit mask, and the flash backend
    refuses a mask -- so windowed layers land on the memory-efficient kernel
    while global ones reach flash. A fused kernel expresses the same window as
    an argument and keeps every layer on one kernel.

    Args:
      q: ``[B, S, num_heads, channels_head]`` queries.
      k: Keys, same shape.
      v: Values, same shape.
      window: Positions each query may look back over, itself included.

    Returns:
      out: Attention output, same shape as ``q``.

    """
    mask = window_mask(q, k, window=window)
    q, k, v = (t.movedim(-3, -2) for t in (q, k, v))
    out = functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        # SDPA refuses a mask beside ``is_causal``, and the window mask is
        # already causal.
        is_causal=mask is None,
    )
    return out.movedim(-3, -2)


class ValueGatedAttention(nn.Module):
    """Windowed causal attention with normalized queries/keys and value gating.

    Two departures from priml's
    :class:`~priml.model.attention.self_attention.SelfAttention`, both load-bearing here:

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

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        num_heads: int = -1
        """Attention num_heads; -1 derives from ``channels_in // channels_head``."""

        channels_head: int = 128
        """Per-head width."""

        gate_channels: int = 32
        """Input channels feeding the value gate; -1 reads the whole stream.

        A fixed slice rather than the model width, so the gate costs the same
        few weights at any size. It must therefore fit: a value wider than the
        model is rejected instead of clamped, since clamping would silently
        build a gate of a shape the recipe never specified."""

        norm_qk: Makeable[TensorModule] = field(default_factory=RMSNorm.Config)
        """Normalization applied to queries and keys before attention.

        Parameter-free, which is ``RMSNorm``'s own default: bounding the
        logits' scale is its whole job here, and a learned gain would duplicate
        the projection that produced them."""

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

        window: int = -1
        """Keys each query attends back over, itself included.

        -1 derives it from ``window_pattern`` at this layer's ``depth``; set it
        to fix one layer's reach regardless of the pattern."""

        window_pattern: str = "SSSL"
        """Cycled reach per layer: L is the full context, S half of it.

        Restricting most layers and leaving a few global keeps attention
        affordable at long context while preserving a path to any position.

        A PATTERN on the attention rather than a window, because a layer's
        reach is a property of the attention and its position -- both of which
        this config holds. The block hands down ``depth``, so the layer selects
        its own symbol and nothing above it needs to know windows exist."""

        max_seq_len: int = -1
        """Full context, and the reach an L layer takes; -1 inherits it."""

        gated: bool = True
        """Whether this layer builds the value gate at all.

        Set by the model from its value-embedding layer set: a layer that
        receives no embedding never reads the gate, so building one would add a
        parameter that trains on nothing and shifts the optimizer's partition
        away from the recipe being reproduced."""

        depth_index: DepthIndex = ()
        """Block depth index, accepted for the priml block contract."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            if self.channels_in != self.channels_out:
                raise ValueError(
                    f"channels_in={self.channels_in} must equal "
                    f"channels_out={self.channels_out} for ValueGatedAttention."
                )
            if self.channels_head <= 0 or self.channels_head % 2:
                raise ValueError(
                    "channels_head must be positive and even; got "
                    f"{self.channels_head}.",
                )
            if self.num_heads == -1:
                if self.channels_in % self.channels_head:
                    raise ValueError(
                        f"channels_in={self.channels_in} is not divisible by "
                        f"channels_head={self.channels_head}; set num_heads.",
                    )
                self.num_heads = self.channels_in // self.channels_head
            if self.window == -1 and self.max_seq_len > 0 and self.depth_index:
                self.window = layer_window(
                    depth_index=self.depth_index,
                    max_seq_len=self.max_seq_len,
                    pattern=self.window_pattern,
                )
            if self.gate_channels == -1:
                self.gate_channels = self.channels_in
            if self.gate_channels <= 0 or self.gate_channels > self.channels_in:
                raise ValueError(
                    f"gate_channels={self.gate_channels} must be positive and "
                    f"at most channels_in={self.channels_in}; the gate reads "
                    "that many leading channels of the layer input.",
                )
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
        if (
            -1 not in (config.channels_in, config.channels_out)
            and config.channels_in != config.channels_out
        ):
            raise ValueError(
                f"channels_in={config.channels_in} must equal "
                f"channels_out={config.channels_out} for ValueGatedAttention."
            )
        super().__init__()
        if min(config.channels_in, config.num_heads, config.gate_channels) <= 0:
            raise ValueError(
                "channels_in, num_heads, and gate_channels must be positive; they "
                "are normally inherited from the block during finalize. Got "
                f"{config.channels_in}, {config.num_heads}, {config.gate_channels}.",
            )
        self.config = config
        inner = config.num_heads * config.channels_head
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
                channels_out=config.num_heads,
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
        *,
        cos_sin: tuple[Tensor, Tensor],
        value_embedding: Tensor | None = None,
        window: int | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Attend over this layer's configured window.

        Args:
          x: ``[B, S, C]`` layer input.
          cos_sin: Rotary ``(cos, sin)`` covering ``S`` positions.
          value_embedding: ``[B, S, num_heads * channels_head]`` added to the
            values through the per-head gate, or ``None`` for this layer.
          window: Attention-window override.
          **kwargs: Open message bus forwarded to the attention kernel.

        Returns:
          out: ``[B, S, C]`` attention output.

        """
        config = self.config
        shape = (*x.shape[:-1], config.num_heads, config.channels_head)
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
        q = rotate_conjugate(q, cos=cos, sin=sin)
        k = rotate_conjugate(k, cos=cos, sin=sin)
        q, k = self.norm_q(q), self.norm_k(k)
        if window is None:
            window = config.window if config.window > 0 else q.shape[-3]
        out = self.attention(q, k, v, window=window, **kwargs)
        return self.proj_out(out.contiguous().flatten(-2))
