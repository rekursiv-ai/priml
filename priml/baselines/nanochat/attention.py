"""Causal attention and fused normalization/rotary kernels.

The attention kernel itself is priml's: ``Flash4Attention`` in
``priml.model.attention.flash4``.

Triton parses source annotations as device code. Keep tuple annotations quoted
and omit future annotations, which makes the formatter remove those quotes.
"""

from collections.abc import Callable
from dataclasses import field
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, Self, cast, override

from configgle import Makeable, Makes
from torch import Tensor, nn

import torch

from priml.baselines.nanochat.ngram import (
    NgramSource,
    ngram_mix,
)
from priml.cost import (
    Cost,
    cost,
    matmul_cost,
    reduction_cost,
    traffic,
)
from priml.kernel import jit_kernel
from priml.model.attention.rope import rotate_conjugate
from priml.model.attention.value_gated_attention import ValueGatedAttention
from priml.model.custom_types import LayerCache, TensorModule, propagate_attr
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.special import Identity


if TYPE_CHECKING:
    from triton import language
    from triton.language.extra.cuda import libdevice

    import triton
else:
    from wrapt import lazy_import

    triton = lazy_import("triton")
    language = lazy_import("triton.language")
    libdevice = lazy_import("triton.language.extra.cuda.libdevice")


class _FusedContext(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


class CausalAttention(ValueGatedAttention):
    """Normalize Q/K before RoPE, with optional output gates and value memories."""

    class Config(Makes["CausalAttention"], ValueGatedAttention.Config):
        head_gate: Linear.Config | None = None
        """Optional per-head output gate; absent in the ungated variant."""

        fused_qk_rope: bool = False
        """Fuse parameter-free RMSNorm and RoPE, reducing in float32.

        ``norm_qk`` must be RMSNorm with epsilon None or float32 epsilon.
        In this path, None selects the float32 accumulation dtype's epsilon.
        """

        bigram: bool = False
        """Build a gate reading the second gate-width slice of the input."""

        trigram: bool = False
        """Build a gate reading the third gate-width slice of the input."""

        norm_out: Makeable[TensorModule] = field(
            default_factory=Identity.Config,
        )
        """Per-head output normalization, preceding the head gate."""

        @override
        def finalize(self) -> Self:
            width = self.channels_in if self.channels_in > 0 else self.channels_out
            if self.head_gate is not None:
                self.head_gate.channels_in = (
                    self.gate_channels if self.gate_channels >= 0 else width
                )
                self.head_gate.channels_out = (
                    self.num_heads
                    if self.num_heads > 0
                    else width // self.channels_head
                )
            propagate_attr(self.norm_out, "channels_in", self.channels_head)
            return super().finalize()

        @override
        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int = 1,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost base attention plus memory gates, output norm, and head gate.

            Ngram tables own their lookups; this module owns the gates and value
            mixing. Fused kernels retain the same logical unfused accounting.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              total: Whole-invocation work, traffic, and owned parameters.

            """
            rows = seq_len * batch_size
            total = super().cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + cost(
                self.norm_out,
                seq_len=seq_len,
                batch_size=batch_size * self.num_heads,
                dtype=dtype,
                **kwargs,
            )
            memory = int(self.bigram) + int(self.trigram)
            total += (
                matmul_cost(
                    channels_in=self.gate_channels,
                    channels_out=self.num_heads,
                    rows=rows,
                    dtype=dtype,
                )
                + _value_mix_cost(
                    heads=self.num_heads,
                    channels_head=self.channels_head,
                    channels_in=self.channels_in,
                    rows=rows,
                    dtype=dtype,
                    add=True,
                )
            ).tile(memory, copies=memory)
            if self.head_gate is not None:
                total += cost(
                    self.head_gate,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                ) + _value_mix_cost(
                    heads=self.num_heads,
                    channels_head=self.channels_head,
                    channels_in=self.channels_in,
                    rows=rows,
                    dtype=dtype,
                    add=False,
                )
            return total

    def __init__(self, config: Config) -> None:
        norm = config.norm_qk
        fused_norm_is_valid = (
            isinstance(norm, RMSNorm.Config)
            and type(norm) is RMSNorm.Config
            and not norm.elementwise_affine
            and norm.eps in (None, torch.finfo(torch.float32).eps)
        )
        if config.fused_qk_rope and not fused_norm_is_valid:
            raise ValueError(
                "fused_qk_rope requires norm_qk to be parameter-free RMSNorm "
                "with epsilon None or float32 epsilon.",
            )
        required_slices = 3 if config.trigram else 2 if config.bigram else 1
        if required_slices * config.gate_channels > config.channels_in:
            raise ValueError(
                f"{required_slices} gate slices of width {config.gate_channels} "
                f"do not fit channels_in={config.channels_in}.",
            )
        super().__init__(config)
        gate = Linear.Config(
            channels_in=config.gate_channels,
            channels_out=config.num_heads,
            init_weight=nn.init.zeros_,
        )
        self.bigram_gate = gate.make() if config.bigram else None
        self.trigram_gate = gate.make() if config.trigram else None
        self.head_gate = (
            config.head_gate.make() if config.head_gate is not None else None
        )
        self.norm_out = config.norm_out.make()
        self.fused_qk_rope = config.fused_qk_rope

    @override
    def reset_parameters(self) -> None:
        super().reset_parameters()
        for gate in (self.bigram_gate, self.trigram_gate, self.head_gate):
            if gate is not None:
                gate.reset_parameters()
        self.norm_out.reset_parameters()

    @override
    def forward(
        self,
        x: Tensor,
        *,
        cos_sin: tuple[Tensor, Tensor],
        value_embedding: Tensor | None = None,
        window: int | None = None,
        cache: LayerCache | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Apply causal attention with normalized rotary queries and keys.

        Args:
          x: Residual stream with channels last.
          cos_sin: Rotary factors by position and half-channel.
          value_embedding: Optional token-specific attention values.
          window: History distance; None uses the configured attention window.
          cache: The block's decode cache; must be ``None``.
          **kwargs: Memory values and unconsumed messages for the attention kernel.

        Returns:
          output: Projected attention output with the shape of ``x``.

        Raises:
          TypeError: If ``cache`` is given.

        """
        # Overrides the base forward whole, so it repeats the base's refusal:
        # ``cache`` must not reach a kernel that names only its window.
        if cache is not None:
            raise TypeError(f"{type(self).__name__} keeps no decode cache.")
        bigram_value = _optional_message(kwargs, "bigram_value", Tensor)
        trigram_value = _optional_message(kwargs, "trigram_value", Tensor)
        fused_tables = _fused_sources(kwargs.pop("fused_tables", None))
        cfg = self.config
        shape = (*x.shape[:-1], cfg.num_heads, cfg.channels_head)
        q, k = self.proj_q(x).view(shape), self.proj_k(x).view(shape)
        cos, sin = cos_sin
        if self.fused_qk_rope:
            q, k = fused_qk_norm_rope(q, k, cos, sin)
        else:
            q, k = (
                rotate_conjugate(self.norm_q(q), cos=cos, sin=sin),
                rotate_conjugate(self.norm_k(k), cos=cos, sin=sin),
            )
        v = self.proj_v(x).view(shape)
        for index, (value, gate) in enumerate(
            (
                (value_embedding, self.value_gate),
                (bigram_value, self.bigram_gate),
                (trigram_value, self.trigram_gate),
            ),
        ):
            if value is not None:
                if gate is None:
                    raise ValueError("Expected gate is not None.")
                start = index * cfg.gate_channels
                weight = 2 * torch.sigmoid(
                    gate(x[..., start : start + cfg.gate_channels]),
                )
                v = v + weight.unsqueeze(-1) * value.view(shape)
        if fused_tables is not None:
            logits: list[Tensor] = []
            weights: list[Tensor] = []
            indices: list[Tensor] = []
            sinks: list[Tensor] = []
            bitmaps: list[Tensor] = []
            for gate_index, table, hashed in fused_tables:
                gate = self.bigram_gate if gate_index == 1 else self.trigram_gate
                if gate is None:
                    raise ValueError(
                        f"fused source {gate_index} has no gate on this attention.",
                    )
                start = gate_index * cfg.gate_channels
                logits.append(gate(x[..., start : start + cfg.gate_channels]))
                weights.extend(part.weight for part in table.tables)
                indices.extend(hashed)
                sinks.extend(table.gradient_sinks)
                bitmaps.extend(table.gradient_bitmaps)
            v = ngram_mix(v.contiguous(), logits, weights, indices, sinks, bitmaps)
        out = self.norm_out(
            self.attention(
                q,
                k,
                v,
                window=cfg.window if window is None else window,
                **kwargs,
            ),
        )
        if self.head_gate is not None:
            head_weights = 2 * torch.sigmoid(
                self.head_gate(x[..., : cfg.gate_channels]),
            )
            out = out * head_weights.unsqueeze(-1)
        return self.proj_out(out.contiguous().flatten(-2))


@torch.library.custom_op("priml_nanochat::qk_norm_rope", mutates_args=())
def fused_qk_norm_rope(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> "tuple[Tensor, Tensor]":
    """Normalize Q/K in FP32 and rotate, rounding when storing the result.

    Args:
      q: Queries shaped ``[batch, tokens, heads, channels]``.
      k: Keys with the same shape; half the channel width must be a power of two.
      cos: Rotary cosine factors by position and half-channel.
      sin: Rotary sine factors matching ``cos``.

    Returns:
      queries: Contiguous normalized and rotated queries in the input dtype.
      keys: Contiguous normalized and rotated keys in the input dtype.

    """
    if q.shape != k.shape:
        raise ValueError("Expected q.shape == k.shape.")
    if q.ndim != 4:
        raise ValueError("Expected q.ndim == 4.")
    half = q.shape[-1] // 2
    if q.shape[-1] % 2 != 0:
        raise ValueError("Expected q.shape[-1] % 2 == 0.")
    if half <= 0:
        raise ValueError("Expected half > 0.")
    if half & half - 1 != 0:
        raise ValueError("Expected half & (half - 1) == 0.")
    if q.is_cuda:
        return _qk_forward_cuda(q, k, cos, sin)
    return (
        _qk_reference(q, cos, sin).contiguous(),
        _qk_reference(k, cos, sin).contiguous(),
    )


def _qk_reference(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    co = cos.reshape(-1, half)[: x.shape[1]].float()[None, :, None, :]
    si = sin.reshape(-1, half)[: x.shape[1]].float()[None, :, None, :]
    a, b = x[..., :half].float(), x[..., half:].float()
    inv = torch.rsqrt(
        (a * a + b * b).sum(-1, keepdim=True) / x.shape[-1] + 1.1920928955078125e-7,
    )
    a, b = a * inv, b * inv
    return torch.cat((a * co + b * si, b * co - a * si), dim=-1).to(x.dtype)


@torch.library.custom_op("priml_nanochat::qk_backward", mutates_args=())
def _qk_backward(
    dq: Tensor,
    dk: Tensor,
    q: Tensor,
    k: Tensor,
    rotation: list[Tensor],
) -> "tuple[Tensor, Tensor]":
    cos, sin = rotation
    if q.is_cuda:
        return _qk_backward_cuda(dq, dk, q, k, cos=cos, sin=sin)
    # CPU cat can preserve channels-last cotangents. Normalize inside the opaque
    # operator so AOT cannot elide the copy using its declared contiguous strides.
    return (
        _qk_backward_reference(dq, q, cos, sin).contiguous(),
        _qk_backward_reference(dk, k, cos, sin).contiguous(),
    )


def _qk_backward_reference(dy: Tensor, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    co = cos.reshape(-1, half)[: x.shape[1]].float()[None, :, None, :]
    si = sin.reshape(-1, half)[: x.shape[1]].float()[None, :, None, :]
    a, b = x[..., :half].float(), x[..., half:].float()
    da, db = dy[..., :half].float(), dy[..., half:].float()
    inv = torch.rsqrt(
        (a * a + b * b).sum(-1, keepdim=True) / x.shape[-1] + 1.1920928955078125e-7,
    )
    a, b = a * inv, b * inv
    ga, gb = da * co - db * si, da * si + db * co
    dot = (a * ga + b * gb).sum(-1, keepdim=True) / x.shape[-1]
    return torch.cat(((ga - a * dot) * inv, (gb - b * dot) * inv), dim=-1).to(x.dtype)


def _qk_setup(
    ctx: _FusedContext,
    inputs: tuple[Tensor, Tensor, Tensor, Tensor],
    output: object,
) -> None:
    del output
    ctx.save_for_backward(*inputs)


@fused_qk_norm_rope.register_fake
def _qk_fake(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> "tuple[Tensor, Tensor]":
    del cos, sin
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
    )


@_qk_backward.register_fake
def _qk_backward_fake(
    dq: Tensor,
    dk: Tensor,
    q: Tensor,
    k: Tensor,
    rotation: list[Tensor],
) -> "tuple[Tensor, Tensor]":
    del dq, dk, rotation
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
    )


def _register_qk_autograd[FunctionT: Callable[..., object]](
    function: FunctionT,
) -> FunctionT:
    fused_qk_norm_rope.register_autograd(function, setup_context=_qk_setup)
    return function


@_register_qk_autograd
def _qk_autograd(
    ctx: _FusedContext,
    dq: Tensor,
    dk: Tensor,
) -> "tuple[Tensor, Tensor, None, None]":
    q, k, cos, sin = ctx.saved_tensors
    a, b = _qk_backward(dq, dk, q, k, [cos, sin])
    return a, b, None, None


def _qk_forward_cuda(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> "tuple[Tensor, Tensor]":
    """Launch fused query/key RMS normalization and rotary embeddings."""
    (q, k) = (q.contiguous(), k.contiguous())
    (b, t, h, d) = q.shape
    rows = b * t * h

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(rows, meta["block"]),)

    (qo, ko) = (torch.empty_like(q), torch.empty_like(k))
    _compiled_qk_forward()[grid](
        buffers=(q, k, cos.contiguous(), sin.contiguous(), qo, ko),
        n_rows=rows,
        geometry=(h, t),
        constants=(1.1920928955078125e-07, d // 2, rows % 32 == 0),
    )
    return (qo, ko)


def _qk_backward_cuda(
    dq: Tensor,
    dk: Tensor,
    q: Tensor,
    k: Tensor,
    *,
    cos: Tensor,
    sin: Tensor,
) -> "tuple[Tensor, Tensor]":
    """Launch inverse rotation and recomputed RMS normalization gradients."""
    (q, k) = (q.contiguous(), k.contiguous())
    (b, t, h, d) = q.shape
    rows = b * t * h

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(rows, meta["block"]),)

    (qo, ko) = (torch.empty_like(q), torch.empty_like(k))
    _compiled_qk_backward()[grid](
        buffers=(
            dq.contiguous(),
            dk.contiguous(),
            q,
            k,
            cos.contiguous(),
            sin.contiguous(),
            qo,
            ko,
        ),
        n_rows=rows,
        geometry=(h, t),
        constants=(1.1920928955078125e-07, d // 2, rows % 32 == 0),
    )
    return (qo, ko)


@lru_cache(maxsize=1)
def _compiled_qk_forward() -> "triton.Autotuner":
    return triton.autotune(
        configs=[
            triton.Config({"block": b}, num_warps=w, num_stages=1)
            for b in (4, 8, 16, 32)
            for w in (2, 4, 8)
        ],
        key=["n_rows"],
    )(jit_kernel(_qk_norm_rope_fwd_triton))


def _qk_norm_rope_fwd_triton(
    buffers: "tuple[language.tensor, ...]",
    n_rows: int,
    geometry: "tuple[int, int]",
    constants: "tuple[float, language.constexpr, bool]",
    block: "language.constexpr",
) -> None:
    """Normalize and rotate query/key rows, loading each rotary half once."""
    q_ptr, k_ptr, cos_ptr, sin_ptr, qo_ptr, ko_ptr = buffers
    n_head, seq_len = geometry
    eps = constants[0]
    half: language.constexpr = constants[1]
    even: language.constexpr = constants[2]
    pid = language.program_id(0)
    rows = pid * block + language.arange(0, block)
    j = language.arange(0, half)
    base = rows[:, None] * (2 * half) + j[None, :]
    if even:
        q1 = language.load(q_ptr + base).to(language.float32)
        q2 = language.load(q_ptr + base + half).to(language.float32)
        k1 = language.load(k_ptr + base).to(language.float32)
        k2 = language.load(k_ptr + base + half).to(language.float32)
    else:
        m = (rows < n_rows)[:, None]
        q1 = language.load(q_ptr + base, mask=m, other=0.0).to(language.float32)
        q2 = language.load(q_ptr + base + half, mask=m, other=0.0).to(language.float32)
        k1 = language.load(k_ptr + base, mask=m, other=0.0).to(language.float32)
        k2 = language.load(k_ptr + base + half, mask=m, other=0.0).to(language.float32)
    inv_d = 1.0 / (2 * half)
    rq = libdevice.rsqrt(language.sum(q1 * q1 + q2 * q2, 1) * inv_d + eps)[:, None]
    rk = libdevice.rsqrt(language.sum(k1 * k1 + k2 * k2, 1) * inv_d + eps)[:, None]
    pos = rows // n_head % seq_len
    cs_off = pos[:, None] * half + j[None, :]
    if even:
        co = language.load(cos_ptr + cs_off).to(language.float32)
        si = language.load(sin_ptr + cs_off).to(language.float32)
    else:
        m = (rows < n_rows)[:, None]
        co = language.load(cos_ptr + cs_off, mask=m, other=0.0).to(language.float32)
        si = language.load(sin_ptr + cs_off, mask=m, other=0.0).to(language.float32)
    (q1n, q2n) = (q1 * rq, q2 * rq)
    (k1n, k2n) = (k1 * rk, k2 * rk)
    qo1 = q1n * co + q2n * si
    qo2 = q2n * co - q1n * si
    ko1 = k1n * co + k2n * si
    ko2 = k2n * co - k1n * si
    if even:
        language.store(qo_ptr + base, qo1)
        language.store(qo_ptr + base + half, qo2)
        language.store(ko_ptr + base, ko1)
        language.store(ko_ptr + base + half, ko2)
    else:
        m = (rows < n_rows)[:, None]
        language.store(qo_ptr + base, qo1, mask=m)
        language.store(qo_ptr + base + half, qo2, mask=m)
        language.store(ko_ptr + base, ko1, mask=m)
        language.store(ko_ptr + base + half, ko2, mask=m)


@lru_cache(maxsize=1)
def _compiled_qk_backward() -> "triton.Autotuner":
    return triton.autotune(
        configs=[
            triton.Config({"block": b}, num_warps=w, num_stages=1)
            for b in (4, 8, 16, 32)
            for w in (2, 4, 8)
        ],
        key=["n_rows"],
    )(jit_kernel(_qk_norm_rope_bwd_triton))


def _qk_norm_rope_bwd_triton(
    buffers: "tuple[language.tensor, ...]",
    n_rows: int,
    geometry: "tuple[int, int]",
    constants: "tuple[float, language.constexpr, bool]",
    block: "language.constexpr",
) -> None:
    """Undo rotation and differentiate RMS normalization using saved projections."""
    dq_ptr, dk_ptr, q_ptr, k_ptr, cos_ptr, sin_ptr, dqo_ptr, dko_ptr = buffers
    n_head, seq_len = geometry
    eps = constants[0]
    half: language.constexpr = constants[1]
    even: language.constexpr = constants[2]
    pid = language.program_id(0)
    rows = pid * block + language.arange(0, block)
    j = language.arange(0, half)
    base = rows[:, None] * (2 * half) + j[None, :]
    if even:
        q1 = language.load(q_ptr + base).to(language.float32)
        q2 = language.load(q_ptr + base + half).to(language.float32)
        k1 = language.load(k_ptr + base).to(language.float32)
        k2 = language.load(k_ptr + base + half).to(language.float32)
        dq1 = language.load(dq_ptr + base).to(language.float32)
        dq2 = language.load(dq_ptr + base + half).to(language.float32)
        dk1 = language.load(dk_ptr + base).to(language.float32)
        dk2 = language.load(dk_ptr + base + half).to(language.float32)
    else:
        m = (rows < n_rows)[:, None]
        q1 = language.load(q_ptr + base, mask=m, other=0.0).to(language.float32)
        q2 = language.load(q_ptr + base + half, mask=m, other=0.0).to(language.float32)
        k1 = language.load(k_ptr + base, mask=m, other=0.0).to(language.float32)
        k2 = language.load(k_ptr + base + half, mask=m, other=0.0).to(language.float32)
        dq1 = language.load(dq_ptr + base, mask=m, other=0.0).to(language.float32)
        dq2 = language.load(dq_ptr + base + half, mask=m, other=0.0).to(
            language.float32,
        )
        dk1 = language.load(dk_ptr + base, mask=m, other=0.0).to(language.float32)
        dk2 = language.load(dk_ptr + base + half, mask=m, other=0.0).to(
            language.float32,
        )
    pos = rows // n_head % seq_len
    cs_off = pos[:, None] * half + j[None, :]
    if even:
        co = language.load(cos_ptr + cs_off).to(language.float32)
        si = language.load(sin_ptr + cs_off).to(language.float32)
    else:
        m = (rows < n_rows)[:, None]
        co = language.load(cos_ptr + cs_off, mask=m, other=0.0).to(language.float32)
        si = language.load(sin_ptr + cs_off, mask=m, other=0.0).to(language.float32)
    inv_d = 1.0 / (2 * half)
    rq = libdevice.rsqrt(language.sum(q1 * q1 + q2 * q2, 1) * inv_d + eps)[:, None]
    rk = libdevice.rsqrt(language.sum(k1 * k1 + k2 * k2, 1) * inv_d + eps)[:, None]
    (qh1, qh2) = (q1 * rq, q2 * rq)
    (kh1, kh2) = (k1 * rk, k2 * rk)
    gq1 = dq1 * co - dq2 * si
    gq2 = dq1 * si + dq2 * co
    gk1 = dk1 * co - dk2 * si
    gk2 = dk1 * si + dk2 * co
    sq = language.sum(qh1 * gq1 + qh2 * gq2, 1)[:, None]
    sk = language.sum(kh1 * gk1 + kh2 * gk2, 1)[:, None]
    dqr1 = (gq1 - qh1 * inv_d * sq) * rq
    dqr2 = (gq2 - qh2 * inv_d * sq) * rq
    dkr1 = (gk1 - kh1 * inv_d * sk) * rk
    dkr2 = (gk2 - kh2 * inv_d * sk) * rk
    if even:
        language.store(dqo_ptr + base, dqr1)
        language.store(dqo_ptr + base + half, dqr2)
        language.store(dko_ptr + base, dkr1)
        language.store(dko_ptr + base + half, dkr2)
    else:
        m = (rows < n_rows)[:, None]
        language.store(dqo_ptr + base, dqr1, mask=m)
        language.store(dqo_ptr + base + half, dqr2, mask=m)
        language.store(dko_ptr + base, dkr1, mask=m)
        language.store(dko_ptr + base + half, dkr2, mask=m)


def _value_mix_cost(
    *,
    heads: int,
    channels_head: int,
    channels_in: int,
    rows: int,
    dtype: torch.dtype | None,
    add: bool,
) -> Cost:
    """Cost scaled sigmoid gates, broadcast products, and optional value additions."""
    inner = heads * channels_head
    dt = dtype
    return (
        traffic(
            "primal",
            "elementwise",
            elements=rows * (5 * heads + (5 if add else 3) * inner),
            flops=rows * (5 * heads + (2 if add else 1) * inner),
            dtype=dt,
        )
        + traffic(
            "adjoint",
            "elementwise",
            elements=rows * (6 * heads + 5 * inner + 3 * channels_in),
            flops=rows * (5 * heads + 2 * inner + channels_in),
            dtype=dt,
        )
        + reduction_cost(
            input_elements=rows * inner,
            output_groups=rows * heads,
            dtype=dt,
            phase="adjoint",
        )
    )


def _optional_message[T](
    kwargs: dict[str, object],
    name: str,
    kind: type[T],
) -> T | None:
    """Pop an optional message, raising TypeError when it is the wrong type."""
    value = kwargs.pop(name, None)
    if value is not None and not isinstance(value, kind):
        raise TypeError(
            f"{name} must be {kind.__name__} or None; got {type(value).__name__}.",
        )
    return value


def _fused_sources(value: object) -> list[NgramSource] | None:
    """Narrow the ``fused_tables`` message, raising TypeError on any other shape."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(
            f"fused_tables must be list or None; got {type(value).__name__}.",
        )
    sources: list[NgramSource] = []
    for source in cast(list[object], value):
        if not isinstance(source, NgramSource):
            raise TypeError(
                f"fused_tables must hold NgramSource; got {type(source).__name__}.",
            )
        sources.append(source)
    return sources
