"""N-stream joint-attention block (MMDiT).

Multi-modal Diffusion Transformer block supporting an arbitrary number
of token streams. Composes MultiStreamAttention (joint attention) with
per-stream FFNs and optional adaLN-Zero conditioning.

References:
  [1] Esser et al., "Scaling Rectified Flow Transformers for
      High-Resolution Image Synthesis" (SD3), arXiv:2403.03206
  [2] Peebles & Xie, "Scalable Diffusion Models with Transformers"
      (DiT), arXiv:2212.09748

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, field
from typing import NamedTuple, Self, override

import copy

from configgle import Fig, Makeable
from torch import Tensor, nn

from priml.math.basic import broadcast_sequences
from priml.model.attention import MultiStreamAttention
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    propagate_attr,
)
from priml.model.linear import Linear
from priml.model.swiglu import SwiGLU


class AdaLNZero(nn.Module):
    """Adaptive LayerNorm-Zero modulation.

    Projects conditioning vector to 6 modulation parameters
    (scale, shift, gate for attn and FFN each). Zero-initialized
    so the block starts as identity at initialization.
    """

    class Output(NamedTuple):
        """Six modulation parameters from AdaLN-Zero."""

        attn_scale: Tensor
        attn_shift: Tensor
        attn_gate: Tensor
        ffn_scale: Tensor
        ffn_shift: Tensor
        ffn_gate: Tensor

    class Config(Fig["AdaLNZero"], kw_only=False):
        channels_in: int = -1
        """Input channels to modulate (output is 6x this)."""

        _: KW_ONLY

        cond_dim: int = -1
        """Conditioning input dimension."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.proj = Linear.Config(
            channels_in=config.cond_dim,
            channels_out=6 * config.channels_in,
            bias=True,
            init_weight=nn.init.zeros_,
            init_bias=nn.init.zeros_,
        ).make()

    def reset_parameters(self) -> None:
        self.proj.reset_parameters()

    @override
    def forward(self, c: Tensor) -> Output:
        """Compute 6 modulation parameters from conditioning vector.

        Args:
          c: [..., cond_dim] conditioning vector.

        Returns:
          params: Output with [..., 1, channels_in] tensors.

        """
        params = self.proj(self.act(c)).unsqueeze(-2)
        return type(self).Output(*params.chunk(6, dim=-1))


class MMDiTBlock(nn.Module):
    """N-stream joint-attention block with optional adaLN-Zero.

    Composes MultiStreamAttention with per-stream norms and FFNs.
    With adaLN-Zero (cond_dim > 0), gates are zero-initialized,
    making the block identity at init.
    """

    class Config(Fig["MMDiTBlock"], kw_only=False):
        channels_in: int = 768
        """Channel width shared across all streams."""

        _: KW_ONLY

        num_streams: int = 2
        """Number of parallel token streams."""

        attn: Makeable[nn.Module] = field(
            default_factory=MultiStreamAttention.Config,
        )
        """Multi-stream attention config."""

        cond_dim: int = 0
        """Conditioning dimension for adaLN-Zero (0 = disabled)."""

        ffn: Makeable[nn.Module] = field(
            default_factory=SwiGLU.Config,
        )
        """FFN config (instantiated once per stream)."""

        depth: int = -1
        """Block depth for depth-scaled init (-1 = no scaling)."""

        @override
        def finalize(self) -> Self:
            propagate_attr(
                self.attn,
                "channels_in",
                self.channels_in,
                protocol=ChannelsIn,
            )
            propagate_attr(self.attn, "num_streams", self.num_streams)
            propagate_attr(self.attn, "depth", self.depth)
            propagate_attr(
                self.ffn,
                "channels_in",
                self.channels_in,
                protocol=ChannelsIn,
            )
            propagate_attr(
                self.ffn,
                "channels_out",
                self.channels_in,
                protocol=ChannelsOut,
            )
            propagate_attr(self.ffn, "depth", self.depth)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        N = config.num_streams
        D = config.channels_in

        self.num_streams = N
        self.attn: MultiStreamAttention = config.attn.make()  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[invalid-assignment]

        self.norms1 = nn.ModuleList(
            nn.LayerNorm(D, elementwise_affine=False) for _ in range(N)
        )

        self.norms2 = nn.ModuleList(
            nn.LayerNorm(D, elementwise_affine=False) for _ in range(N)
        )

        # Per-stream FFNs (dims propagated in finalize).
        self.ffns = nn.ModuleList(copy.copy(config.ffn).make() for _ in range(N))

        # Optional per-stream adaLN-Zero.
        self.adalns: nn.ModuleList | None = None
        if config.cond_dim > 0:
            self.adalns = nn.ModuleList(
                AdaLNZero.Config(
                    channels_in=D,
                    cond_dim=config.cond_dim,
                ).make()
                for _ in range(N)
            )

    def reset_parameters(self) -> None:
        self.attn.reset_parameters()
        for modules in (self.norms1, self.norms2, self.ffns):
            for m in modules:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        if self.adalns is not None:
            for m in self.adalns:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

    @override
    def forward(
        self,
        xs: Sequence[Tensor],
        c: Tensor | Sequence[Tensor] | None = None,
        cos_sin: (Sequence[tuple[Tensor, Tensor] | None] | None) = None,
    ) -> tuple[Tensor, ...]:
        """Forward pass through the N-stream block.

        Args:
          xs: Per-stream tokens, each [..., S_i, channels_in].
          c: Conditioning for adaLN. Single tensor broadcasts to
            all streams, or a sequence of one per stream.
          cos_sin: Per-stream RoPE (cos, sin) pairs. None entries
            skip positional encoding for that stream.

        Returns:
          ys: Per-stream output tokens, same shapes as xs.

        """
        N = self.num_streams

        (cs,) = broadcast_sequences(c if c is not None else [None] * N)
        if len(cs) == 1:
            cs = cs * N

        mods: list[AdaLNZero.Output | None] = [None] * N
        if self.adalns is not None:
            for i, ci in enumerate(cs):
                if ci is not None:
                    mods[i] = self.adalns[i](ci)

        normed: list[Tensor] = []
        for i, x in enumerate(xs):
            mod = mods[i]
            h = self.norms1[i](x)
            if mod is not None:
                h = h * (1 + mod.attn_scale) + mod.attn_shift
            normed.append(h)

        attn_outs = self.attn(normed, cos_sin=cos_sin)

        results: list[Tensor] = []
        for i in range(N):
            mod = mods[i]
            out = attn_outs[i]
            if mod is not None:
                out = mod.attn_gate * out
            y = xs[i] + out

            h = self.norms2[i](y)
            if mod is not None:
                h = h * (1 + mod.ffn_scale) + mod.ffn_shift
            ffn_out = self.ffns[i](h)
            if mod is not None:
                ffn_out = mod.ffn_gate * ffn_out
            results.append(y + ffn_out)

        return tuple(results)
