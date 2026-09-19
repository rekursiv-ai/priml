"""Windowed causal attention with a per-head value-embedding gate.

Two departures from :class:`~priml.model.attention.self_attention.SelfAttention`, both
from the speedrun recipes rather than from taste:

* **A window.** A layer attends to ``window`` previous positions plus itself.
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

from dataclasses import KW_ONLY, field, replace
from typing import Self, override

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
    reduction_cost,
    resolve_dtype,
    traffic,
)
from priml.model.attention.kernel import attention_kernel_cost
from priml.model.attention.rope import rotate_conjugate
from priml.model.attention.window import layer_window, window_mask
from priml.model.custom_types import (
    AttentionKernel,
    ChannelsIn,
    DepthIndex,
    TensorModule,
    infer_same_width,
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
      window: Previous positions each query may reach, in addition to itself.

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


class SdpaCausal:
    """:func:`sdpa_attention` as a slot value that costs itself.

    Holds no state. It exists so the default kernel is a config with a
    ``cost`` -- the standard two products, bound the way every other priml
    kernel binds them -- rather than a bare function the owner would have to
    cost on its behalf.
    """

    class Config(Fig["SdpaCausal"]):
        @classmethod
        def cost(
            cls,
            *,
            seq_len: int,
            batch_size: int = 1,
            dtype: torch.dtype | None,
            num_heads: int,
            channels_head: int,
            channels_v_head: int = -1,
            window: int = -1,
            dropout_p: float = 0.0,
            rows: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Cost the kernel from the shapes its owner hands it.

            See :func:`attention_kernel_cost` for every argument.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              num_heads: Query heads.
              channels_head: Width of each query/key head.
              channels_v_head: Value width; -1 uses the query/key width.
              window: Previous keys each query reaches, plus itself; negative is unbounded.
              dropout_p: Attention dropout rate.
              rows: Query rows sharing K/V; negative uses the modeled key count.
              **kwargs: The rest of the owner's bus, unread.

            Returns:
              cost: Whole-invocation cost over ``seq_len`` and ``batch_size``.

            """
            del kwargs, window
            return attention_kernel_cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                num_heads=num_heads,
                channels_head=channels_head,
                channels_v_head=channels_v_head,
                window=-1,
                dropout_p=dropout_p,
                rows=rows,
            )

    def __init__(self, config: Config) -> None:
        del config

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        window: int = -1,
        **kwargs: object,
    ) -> Tensor:
        """Attend over ``window`` previous positions plus self, causally.

        Remaining keyword arguments belong to the open model message bus; this
        kernel reads only the window it understands.
        """
        del kwargs
        return sdpa_attention(q, k, v, window=window)


class ValueGatedAttention(nn.Module):
    """Windowed causal attention with normalized queries/keys and value gating.

    Two departures from priml's
    :class:`~priml.model.attention.self_attention.SelfAttention`, both load-bearing here:

    * **A window.** A layer attends to ``window`` previous positions plus itself.
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
        """Input channel width."""

        channels_out: int = -1
        """Output channel width."""

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

        kernel: Makeable[AttentionKernel] = field(default_factory=SdpaCausal.Config)
        """The attention kernel itself, injected rather than selected.

        A kernel is a different VALUE in this slot, not a mode flag: the
        reference recipe measured its score on FlashAttention-3, and a fused
        kernel reduces in a different order than a masked SDPA, so reproducing
        that number means issuing that kernel. The default runs anywhere; a
        rung reproducing a published result pins the one it was published
        with, and inherits its hardware requirement along with it."""

        window: int = -1
        """Previous positions each query reaches, in addition to itself.

        Zero is self-only. -1 derives history from ``window_pattern`` when
        depth and context are known, otherwise leaves it unbounded."""

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
            infer_same_width(self)
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if (
                self.num_heads == -1
                and self.channels_head > 0
                and self.channels_in % self.channels_head == 0
            ):
                self.num_heads = self.channels_in // self.channels_head
            if self.window == -1 and self.max_seq_len > 0 and self.depth_index:
                self.window = layer_window(
                    depth_index=self.depth_index,
                    max_seq_len=self.max_seq_len,
                    pattern=self.window_pattern,
                )
            if self.gate_channels == -1:
                self.gate_channels = self.channels_in
            # The norm sees one HEAD, not the residual stream, so it takes the
            # head width rather than the model width the block propagated.
            propagate_attr(
                self.norm_qk,
                "channels_in",
                self.channels_head,
                protocol=ChannelsIn,
            )
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int = 1,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost four projections, the gate, two norms, and the kernel.

            The injected kernel owns its product geometry: masked SDPA is dense,
            while a local-window kernel can use fewer keys. Rotation and the
            value gate are counted here, since no child owns them.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation cost over ``seq_len`` and ``batch_size``.

            """
            rows = seq_len * batch_size
            dt = dtype
            inner = self.num_heads * self.channels_head
            total = matmul_cost(
                channels_in=self.channels_in,
                channels_out=inner,
                rows=rows,
                dtype=dt,
            ).tile(3, copies=3) + matmul_cost(
                channels_in=inner,
                channels_out=self.channels_in,
                rows=rows,
                dtype=dt,
            )
            if self.gated:
                total += matmul_cost(
                    channels_in=self.gate_channels,
                    channels_out=self.num_heads,
                    rows=rows,
                    dtype=dt,
                )
            total += cost(
                self.norm_qk,
                seq_len=seq_len,
                batch_size=batch_size * self.num_heads,
                dtype=dtype,
                **kwargs,
            )
            total += cost(
                self.norm_qk,
                seq_len=seq_len,
                batch_size=batch_size * self.num_heads,
                dtype=dtype,
                **kwargs,
            )
            total += cost(
                self.kernel,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                num_heads=self.num_heads,
                channels_head=self.channels_head,
                window=self.window,
            )
            total += elementwise_cost(
                primal=6 * inner * rows,
                adjoint=6 * inner * rows,
                channels=2 * inner,
                inputs=6,
                outputs=3,
                adjoint_inputs=6,
                adjoint_outputs=3,
                rows=rows,
                dtype=dt,
            )
            if self.gated:
                # Sigmoid/scale, then broadcast product and addition. The head
                # gradient reduces channels; the input slice merges two paths.
                total += (
                    traffic(
                        "primal",
                        "elementwise",
                        elements=5 * self.num_heads + 5 * inner,
                        flops=5 * self.num_heads + 2 * inner,
                        dtype=dt,
                    )
                    + traffic(
                        "adjoint",
                        "elementwise",
                        elements=6 * self.num_heads + 5 * inner + 3 * self.channels_in,
                        flops=5 * self.num_heads + 2 * inner + self.channels_in,
                        dtype=dt,
                    )
                    + reduction_cost(
                        input_elements=inner,
                        output_groups=self.num_heads,
                        dtype=dt,
                        phase="adjoint",
                    )
                ).tile(rows)
            return replace(total, bytes_state=resolve_dtype(dtype).itemsize * 2 * inner)

    def __init__(self, config: Config) -> None:
        # Only constraints torch cannot state: RoPE pairs the head width, and
        # the gate indexes into the layer input. A nonpositive extent is torch's
        # to reject (STYLE.md "Let the leaf complain").
        if config.channels_head % 2:
            raise ValueError(
                f"channels_head must be even; got {config.channels_head}.",
            )
        if config.num_heads == -1:
            raise ValueError(
                f"channels_in={config.channels_in} is not divisible by "
                f"channels_head={config.channels_head}; set num_heads.",
            )
        if config.gate_channels > config.channels_in:
            raise ValueError(
                f"gate_channels={config.gate_channels} must be "
                f"at most channels_in={config.channels_in}; the gate reads "
                "that many leading channels of the layer input.",
            )
        super().__init__()
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
        """Re-initialize every projection and injected normalization."""
        for module in (self.proj_q, self.proj_k, self.proj_v, self.proj_out):
            module.reset_parameters()
        if self.value_gate is not None:
            self.value_gate.reset_parameters()
        for norm in (self.norm_q, self.norm_k):
            norm.reset_parameters()

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
            if self.value_gate is None:
                raise ValueError("Expected self.value_gate is not None.")
            gate = 2 * torch.sigmoid(self.value_gate(x[..., : config.gate_channels]))
            v = v + gate.unsqueeze(-1) * value_embedding.view(shape)
        cos, sin = cos_sin
        q = rotate_conjugate(q, cos=cos, sin=sin)
        k = rotate_conjugate(k, cos=cos, sin=sin)
        q, k = self.norm_q(q), self.norm_k(k)
        if window is None:
            window = config.window if config.window >= 0 else q.shape[-3]
        out = self.attention(q, k, v, window=window, **kwargs)
        return self.proj_out(out.contiguous().flatten(-2))
