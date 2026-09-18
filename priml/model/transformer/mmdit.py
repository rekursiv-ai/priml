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
from functools import partial
from typing import TYPE_CHECKING, NamedTuple, Protocol, Self, override
from typing_extensions import ParamSpec

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
)
from priml.model.attention.multi_stream import (
    MultiStreamAttention,
    _validate_native_state,
)
from priml.model.attention.self_attention import AttentionProjections
from priml.model.custom_types import (
    AttentionKernel,
    ChannelsHead,
    ChannelsIn,
    ChannelsOut,
    DepthIndex,
    HasDepthIndex,
    HasResetParameters,
    NumHeads,
    TensorModule,
    infer_same_width,
    propagate_attr,
)
from priml.model.linear import Linear
from priml.model.norm import LayerNorm
from priml.model.swiglu import SwiGLU


if TYPE_CHECKING:
    from priml.model.transformer.block import TransformerBlock


_AttentionKwargs = ParamSpec("_AttentionKwargs", default=...)


class _JointAttention(Protocol[_AttentionKwargs]):
    """Joint stream attention preserving its implementation's keyword bus."""

    streams: nn.ModuleList[AttentionProjections]
    proj_outs: nn.ModuleList[Linear]
    attn_kernel: AttentionKernel

    def reset_parameters(self) -> None: ...

    def __call__(
        self,
        xs: Sequence[Tensor],
        /,
        *args: _AttentionKwargs.args,
        **kwargs: _AttentionKwargs.kwargs,
    ) -> tuple[Tensor, ...]: ...


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
        """Residual width to modulate; projection output is six times this width."""

        _: KW_ONLY

        cond_dim: int = -1
        """Conditioning input dimension."""

        proj: Makeable[TensorModule] = field(
            default_factory=partial(
                Linear.Config,
                bias=True,
                init_weight=nn.init.zeros_,
                init_bias=nn.init.zeros_,
            ),
        )
        """Conditioning projection; widths filled by ``finalize``.

        Zero-initialized by default -- that is what "Zero" names, and it is what
        makes the block an identity at init rather than a random perturbation."""

        @override
        def finalize(self) -> Self:
            if isinstance(self.proj, ChannelsIn) and self.proj.channels_in == -1:
                self.proj.channels_in = self.cond_dim
            if isinstance(self.proj, ChannelsOut) and self.proj.channels_out == -1:
                self.proj.channels_out = 6 * self.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Price the projection; the SiLU is elementwise.

            Counted per token like every leaf, though ``c`` is often one vector
            per sequence: a per-sequence conditioning amortizes this over the
            sequence, so the per-token figure is an upper bound.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            return cost(
                self.proj,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + elementwise_cost(
                primal=5 * self.cond_dim,
                adjoint=5 * self.cond_dim,
                channels=self.cond_dim,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.proj = config.proj.make()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        self.proj.reset_parameters()

    @override
    def forward(self, c: Tensor, **kwargs: object) -> Output:
        """Compute 6 modulation parameters from conditioning vector.

        Args:
          c: [..., cond_dim] conditioning vector.
          **kwargs: Open message bus forwarded to the projection.

        Returns:
          attn_scale: Attention scale offset, shaped [..., 1, channels_in].
          attn_shift: Attention shift, shaped [..., 1, channels_in].
          attn_gate: Attention residual gate, shaped [..., 1, channels_in].
          ffn_scale: FFN scale offset, shaped [..., 1, channels_in].
          ffn_shift: FFN shift, shaped [..., 1, channels_in].
          ffn_gate: FFN residual gate, shaped [..., 1, channels_in].

        """
        params = self.proj(self.act(c), **kwargs).unsqueeze(-2)
        return type(self).Output(*params.chunk(6, dim=-1))


class MMDiTStream(nn.Module):
    """Own one stream's residual branches; joint attention builds its attn subtree."""

    class Config(Fig["MMDiTStream"]):
        channels_in: int = -1
        """Residual width, inherited from the joint block."""

        attn: AttentionProjections.Config = field(
            default_factory=AttentionProjections.Config,
        )
        """Stream attention leaves, built and registered by the joint attention."""

        norm1: Makeable[TensorModule] = field(default_factory=LayerNorm.Config)
        """Normalization before attention."""

        norm2: Makeable[TensorModule] = field(default_factory=LayerNorm.Config)
        """Normalization before the feed-forward branch."""

        ffn: Makeable[TensorModule] = field(default_factory=SwiGLU.Config)
        """Feed-forward branch for this stream."""

        adaln: Makeable[AdaLNZero] | None = None
        """Optional conditioning modulation; None has no conditioning parameters."""

        depth_index: DepthIndex = ()
        """Block depth for initialization of this stream's branches."""

        @override
        def finalize(self) -> Self:
            for child in (self.norm1, self.norm2, self.ffn, self.adaln):
                if isinstance(child, ChannelsIn) and child.channels_in == -1:
                    child.channels_in = self.channels_in
                if isinstance(child, ChannelsOut) and child.channels_out == -1:
                    child.channels_out = self.channels_in
                if isinstance(child, HasDepthIndex) and not child.depth_index:
                    child.depth_index = self.depth_index
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Sum the residual branches; ``attn`` is priced by the joint attention.

            The joint attention registers ``attn`` as one of its own streams and
            builds it, so it prices it too; counting it here would charge every
            stream's projections twice.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            children = (self.norm1, self.norm2, self.ffn)
            total = sum(
                (
                    cost(
                        child,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        dtype=dtype,
                        **kwargs,
                    )
                    for child in children
                ),
                Cost(),
            )
            if self.adaln is not None:
                total += cost(
                    self.adaln,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
            # Per branch: scale offset, scale, shift, gate, residual.
            modulated = self.adaln is not None
            adds = (10 if modulated else 2) * self.channels_in
            return total + elementwise_cost(
                primal=adds,
                adjoint=adds,
                channels=2 * self.channels_in,
                inputs=9 if modulated else 2,
                outputs=5 if modulated else 1,
                adjoint_inputs=13 if modulated else 2,
                adjoint_outputs=7 if modulated else 1,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        norm1 = config.norm1.make()
        norm2 = config.norm2.make()
        ffn = config.ffn.make()
        assert isinstance(norm1, nn.Module)
        assert isinstance(norm2, nn.Module)
        assert isinstance(ffn, nn.Module)
        self.norm1 = norm1
        self.norm2 = norm2
        self.ffn = ffn
        self.adaln = config.adaln.make() if config.adaln is not None else None


class MMDiTBlock(nn.Module):
    """N-stream joint-attention block with optional adaLN-Zero.

    Composes MultiStreamAttention with per-stream norms and FFNs.
    With adaLN-Zero (cond_dim > 0), gates are zero-initialized,
    making the block identity at init.
    """

    class Config(Fig["MMDiTBlock"], kw_only=False):
        channels_in: int = -1
        """Input channel width."""

        channels_out: int = -1
        """Output channel width."""

        _: KW_ONLY

        num_streams: int = 2
        """Number of parallel token streams when streams is empty."""

        attn: Makeable[_JointAttention] = field(
            default_factory=MultiStreamAttention.Config,
        )
        """Multi-stream attention config.

        Declared at the class the block actually uses, not ``nn.Module``: the
        body reads members only multi-stream attention has, and the wider
        declaration made every one of them an assertion.
        """

        streams: list[MMDiTStream.Config] = field(
            default_factory=list[MMDiTStream.Config],
        )
        """Explicit stream-owned subtrees; their length determines the stream count."""

        cond_dim: int = 0
        """Implicit-stream conditioning dimension (0 = disabled).

        Explicit streams own adaln.
        """

        ffn: Makeable[TensorModule] = field(default_factory=SwiGLU.Config)
        """FFN template for implicit streams; explicit streams own their FFNs."""

        depth_index: DepthIndex = ()
        """Block depth for depth-scaled init (empty = no scaling)."""

        @property
        def num_heads(self) -> int:
            """Return the joint attention's head count."""
            return self.attn.num_heads if isinstance(self.attn, NumHeads) else 1

        @property
        def channels_head(self) -> int:
            """Return the joint attention's per-head channel width."""
            if isinstance(self.attn, ChannelsHead):
                return self.attn.channels_head
            return self.channels_in

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            if self.streams:
                self.num_streams = len(self.streams)
                if isinstance(self.attn, MultiStreamAttention.Config):
                    self.attn.streams = [stream.attn for stream in self.streams]
                for stream in self.streams:
                    stream.channels_in = self.channels_in
                    stream.depth_index = self.depth_index
            propagate_attr(
                self.attn,
                "channels_in",
                self.channels_in,
                protocol=ChannelsIn,
            )
            propagate_attr(self.attn, "num_streams", self.num_streams)
            propagate_attr(
                self.attn,
                "depth_index",
                self.depth_index,
                protocol=HasDepthIndex,
            )
            if not self.streams:
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
                propagate_attr(
                    self.ffn,
                    "depth_index",
                    self.depth_index,
                    protocol=HasDepthIndex,
                )
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Sum the joint attention and every stream's residual branches.

            ``seq_len`` is the JOINT key length -- every stream's
            tokens concatenated -- as :class:`MultiStreamAttention` defines it,
            so a caller holding per-stream lengths passes their sum. Every
            stream's branches are summed, matching the joint attention's
            convention that each stream's token pays its own projections.
            Implicit streams are priced from the same templates ``__init__``
            builds.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            total = cost(
                self.attn,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            if self.streams:
                return total + sum(
                    (
                        cost(
                            s,
                            seq_len=seq_len,
                            batch_size=batch_size,
                            dtype=dtype,
                            **kwargs,
                        )
                        for s in self.streams
                    ),
                    Cost(),
                )
            stream = MMDiTStream.Config()
            stream.channels_in = self.channels_in
            stream.norm1 = LayerNorm.Config(channels_in=self.channels_in)
            stream.norm2 = LayerNorm.Config(channels_in=self.channels_in)
            stream.ffn = self.ffn
            if self.cond_dim > 0:
                adaln = AdaLNZero.Config()
                adaln.channels_in = self.channels_in
                adaln.cond_dim = self.cond_dim
                stream.adaln = adaln.finalize()
            return total + cost(
                stream,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ).tile(self.num_streams, copies=self.num_streams)

    def __init__(self, config: Config) -> None:
        super().__init__()
        N = config.num_streams
        D = config.channels_in

        self.num_streams = N
        self.attn = config.attn.make()
        self.adalns: nn.ModuleDict | None = None
        if config.streams:
            streams = [stream.make() for stream in config.streams]
            # Register leaves at the established checkpoint paths, without aliases.
            self.norms1 = nn.ModuleList(stream.norm1 for stream in streams)
            self.norms2 = nn.ModuleList(stream.norm2 for stream in streams)
            self.ffns = nn.ModuleList(stream.ffn for stream in streams)
            if any(stream.adaln is not None for stream in streams):
                self.adalns = nn.ModuleDict(
                    {
                        str(i): stream.adaln
                        for i, stream in enumerate(streams)
                        if stream.adaln is not None
                    },
                )
            return

        self.norms1 = nn.ModuleList(
            LayerNorm.Config(channels_in=D).make() for _ in range(N)
        )

        self.norms2 = nn.ModuleList(
            LayerNorm.Config(channels_in=D).make() for _ in range(N)
        )

        # Per-stream FFNs (dims propagated in finalize).
        ffns: list[nn.Module] = []
        for _ in range(N):
            ffn = config.ffn.make()
            assert isinstance(ffn, nn.Module)
            ffns.append(ffn)
        self.ffns = nn.ModuleList(ffns)

        # Optional per-stream adaLN-Zero.
        if config.cond_dim > 0:
            self.adalns = nn.ModuleDict(
                {
                    str(i): AdaLNZero.Config(
                        channels_in=D,
                        cond_dim=config.cond_dim,
                    ).make()
                    for i in range(N)
                },
            )

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        self.attn.reset_parameters()
        for modules in (self.norms1, self.norms2, self.ffns):
            for m in modules:
                if isinstance(m, HasResetParameters):
                    m.reset_parameters()
        if self.adalns is not None:
            for m in self.adalns.values():
                if isinstance(m, HasResetParameters):
                    m.reset_parameters()

    def load_stream(self, index: int, *, source: TransformerBlock) -> None:
        """Copy a native prenorm transformer into one unconditioned stream.

        Configure identical architecture first: normalization epsilon, FFN
        activation, rotary frequencies and attention policies are not weights.
        All native state keys and tensor shapes are checked before any copy;
        unrelated streams and parameter requires_grad flags remain unchanged.

        Args:
          index: Explicit destination stream index.
          source: Native prenorm TransformerBlock, already loaded if pretrained.

        """
        if not self.attn.streams:
            raise ValueError("Native loading requires explicit streams.")
        if index < 0 or index >= self.num_streams:
            raise ValueError(f"Invalid stream index {index}.")
        if not source.prenorm:
            raise ValueError("Native stream loading requires a prenorm transformer.")
        if self.adalns is not None and str(index) in self.adalns:
            raise ValueError("Native stream loading requires an unconditioned stream.")
        target = nn.ModuleDict(
            {
                "attn": self.attn.streams[index],
                "norm1": self.norms1[index],
                "norm2": self.norms2[index],
                "ffn": self.ffns[index],
            },
        )
        _validate_native_state(target, source=source)
        target.load_state_dict(source.state_dict(), strict=True)

    @override
    def forward(
        self,
        xs: Sequence[Tensor],
        *,
        c: Tensor | Sequence[Tensor | None] | None = None,
        cos_sin: (Sequence[tuple[Tensor, Tensor] | None] | None) = None,
        **kwargs: object,
    ) -> tuple[Tensor, ...]:
        """Forward pass through the N-stream block.

        Args:
          xs: Per-stream tokens, each [..., S_i, channels_in].
          c: Conditioning for adaLN. A single tensor or one-element sequence
            broadcasts to all streams; otherwise supply one entry per stream.
          cos_sin: Per-stream RoPE (cos, sin) pairs. None entries
            skip positional encoding for that stream.
          **kwargs: Open message bus forwarded to every sublayer.

        Returns:
          ys: Per-stream output tokens, same shapes as xs.

        """
        N = self.num_streams
        if len(xs) != N:
            raise ValueError(f"Expected {N} streams, got {len(xs)}.")

        if self.adalns is not None and c is None:
            # Skipping AdaLN would add both sublayers UNGATED, which is not the
            # identity-at-init this block documents. Zeros give that identity.
            raise ValueError(
                "conditioning is required when cond_dim > 0; pass a tensor "
                "per stream, or one to broadcast.",
            )

        cs: list[Tensor | None] = list(c) if isinstance(c, Sequence) else [c] * N
        if len(cs) == 1:
            cs = cs * N
        if len(cs) != N:
            raise ValueError(
                f"Got {len(cs)} conditioning tensors for {N} streams; supply "
                "one per stream, or a single tensor to broadcast.",
            )

        mods: list[AdaLNZero.Output | None] = [None] * N
        if self.adalns is not None:
            for key, adaln in self.adalns.items():
                i = int(key)
                ci = cs[i]
                if ci is None:
                    raise ValueError(f"conditioning is required for stream {i}.")
                mods[i] = adaln(ci, **kwargs)

        normed: list[Tensor] = []
        for i, x in enumerate(xs):
            mod = mods[i]
            h = self.norms1[i](x, **kwargs)
            if mod is not None:
                h = h * (1 + mod.attn_scale) + mod.attn_shift
            normed.append(h)

        attn_outs = self.attn(normed, cos_sin=cos_sin, **kwargs)

        results: list[Tensor] = []
        for i in range(N):
            mod = mods[i]
            out = attn_outs[i]
            if mod is not None:
                out = mod.attn_gate * out
            y = xs[i] + out

            h = self.norms2[i](y, **kwargs)
            if mod is not None:
                h = h * (1 + mod.ffn_scale) + mod.ffn_shift
            ffn_out = self.ffns[i](h, **kwargs)
            if mod is not None:
                ffn_out = mod.ffn_gate * ffn_out
            results.append(y + ffn_out)

        return tuple(results)
