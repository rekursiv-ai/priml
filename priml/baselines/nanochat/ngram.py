"""Causal n-gram embeddings and fused table gather kernels.

Triton parses source annotations as device code. Keep tuple annotations quoted
and omit future annotations, which makes the formatter remove those quotes.
"""

from collections.abc import Callable
from dataclasses import field
from functools import lru_cache, partial
from importlib import import_module
from types import FunctionType
from typing import TYPE_CHECKING, NamedTuple, Protocol, Self, override

import math

from configgle import Fig, Makes
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.model.embedding import Embedding
from priml.model.narrow_embedding import NarrowEmbedding


if TYPE_CHECKING:
    from triton import language

    import triton
else:
    from wrapt import lazy_import

    # Triton adds 100 ms median on Colossus after torch (five fresh processes,
    # Python 3.14/Triton 3.6); CPU embeddings never need its CUDA kernels.
    triton = lazy_import("triton")
    language = lazy_import("triton.language")


class _FusedContext(Protocol):
    saved_tensors: tuple[Tensor, ...]
    sources: int
    marked: int

    def save_for_backward(self, *tensors: Tensor) -> None: ...


class NgramEmbedding(NarrowEmbedding):
    """Embed direct or hashed tokens and add optional context embeddings."""

    class Config(Makes["NgramEmbedding"], NarrowEmbedding.Config):
        multipliers: tuple[int, ...] = ()
        """Causal hash coefficients, current token first; empty uses direct IDs."""

        scale: float = 1.0
        """Multiplier applied to this lookup before adding its contexts."""

        contexts: dict[str, NarrowEmbedding.Config] = field(
            default_factory=dict[str, NarrowEmbedding.Config],
        )
        """Named context embeddings added in insertion order; empty adds none."""

        @override
        def finalize(self) -> Self:
            for context in self.contexts.values():
                context.channels_out = self.channels_out
                context.dtype = self.dtype
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.multipliers = config.multipliers
        self.num_embeddings = config.channels_in
        self.scale = config.scale
        self.contexts = nn.ModuleDict(
            {name: context.make() for name, context in config.contexts.items()},
        )

    @override
    def reset_parameters(self) -> None:
        """Reset this lookup and its contexts in their construction order."""
        super().reset_parameters()
        for context in self.contexts.values():
            assert isinstance(context, NarrowEmbedding)
            context.reset_parameters()

    @override
    def forward(self, tokens: Tensor, **kwargs: object) -> Tensor:
        """Add direct or hashed token lookups and configured context embeddings.

        Args:
          tokens: Token IDs with sequence on the last axis.
          **kwargs: Messages forwarded to the embedding components.

        Returns:
          embeddings: Token representations with channels on the last axis.

        """
        indices = tokens
        if self.multipliers:
            indices = tokens * self.multipliers[0]
            for lag, multiplier in enumerate(self.multipliers[1:], start=1):
                previous = functional.pad(tokens, (lag, 0))[..., :-lag]
                indices = indices + multiplier * previous
            indices = indices % self.num_embeddings
        output = super().forward(indices, **kwargs)
        if self.multipliers:
            prefix = len(self.multipliers) - 1
            output = torch.cat(
                (torch.zeros_like(output[..., :prefix, :]), output[..., prefix:, :]),
                dim=-2,
            )
        if self.scale != 1.0:
            output = self.scale * output
        for context in self.contexts.values():
            assert isinstance(context, NarrowEmbedding)
            output = output + context(tokens, **kwargs)
        return output


class HashedNgramTables(nn.Module):
    """Concatenate independently hashed tables over causal token n-grams."""

    class Config(Fig["HashedNgramTables"]):
        channels_out: int = -1
        """Concatenated width; inherited from the attention values."""

        num_embeddings: int = 8192
        """Buckets in each independent table."""

        hash_multipliers: tuple[tuple[int, ...], ...] = ((1,),)
        """Multipliers, oldest token first, for each hash."""

        table: Embedding.Config = field(default_factory=Embedding.Config)
        """Table template; width and initialization follow the full value width."""

        init_after: Callable[[Tensor], Tensor] | None = None
        """Optional transform after all table draws, preserving their RNG sequence."""

        @override
        def finalize(self) -> Self:
            if not self.hash_multipliers or self.channels_out % len(
                self.hash_multipliers,
            ):
                raise ValueError("Hash count must divide the value width.")
            if len({len(row) for row in self.hash_multipliers}) != 1:
                raise ValueError("All hashes must have the same n-gram order.")
            self.table.channels_out = self.channels_out // len(self.hash_multipliers)
            self.table.channels_in = self.num_embeddings
            bound = (3 / self.channels_out) ** 0.5
            self.table.init_weight = partial(nn.init.uniform_, a=-bound, b=bound)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.hash_multipliers = config.hash_multipliers
        self.num_embeddings = config.num_embeddings
        self.tables = nn.ModuleList(
            [config.table.make() for _ in config.hash_multipliers],
        )
        self.init_after = config.init_after
        if self.init_after is not None:
            for table in self.tables:
                self.init_after(table.weight)
        self.gradient_sinks: list[Tensor] = []
        self.gradient_bitmaps: list[Tensor] = []

    @override
    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        """Move auxiliary buffers with their tables while retaining FP32 sinks."""
        super()._apply(fn, recurse)
        # These non-checkpointed buffers are not Parameters. Applying the dtype
        # transform itself would quantize accumulated gradients, so follow only
        # the table's destination device. Meta materialization resets them later.
        for buffers in (self.gradient_sinks, self.gradient_bitmaps):
            for index, buffer in enumerate(buffers):
                device = self.tables[index].weight.device
                buffers[index] = (
                    torch.empty_like(buffer, device=device)
                    if buffer.is_meta
                    else buffer.to(device=device)
                )
        return self

    def prepare_gradient_sinks(self, *, dirty_bitmaps: bool = False) -> None:
        """Allocate FP32 table-gradient buffers after parameter materialization.

        Args:
          dirty_bitmaps: Allocate row flags for sparse updates and gradient clearing.

        """
        self.gradient_sinks = [
            torch.zeros_like(table.weight, dtype=torch.float32) for table in self.tables
        ]
        self.gradient_bitmaps = (
            [
                torch.zeros(
                    table.weight.shape[0],
                    dtype=torch.uint8,
                    device=table.weight.device,
                )
                for table in self.tables
            ]
            if dirty_bitmaps
            else []
        )

    def reset_parameters(self) -> None:
        """Initialize all tables, then apply the configured transform."""
        for table in self.tables:
            table.reset_parameters()
        if self.init_after is not None:
            for table in self.tables:
                self.init_after(table.weight)

    @override
    def forward(self, tokens: Tensor, **kwargs: object) -> Tensor:
        """Concatenate independently hashed table lookups.

        Args:
          tokens: Token IDs with sequence on the last axis.
          **kwargs: Unused model messages.

        Returns:
          embeddings: Concatenated table outputs with channels last.

        """
        del kwargs
        return torch.cat(
            [
                table(index)
                for table, index in zip(self.tables, self.indices(tokens), strict=True)
            ],
            dim=-1,
        )

    def indices(self, tokens: Tensor) -> list[Tensor]:
        """Compute causal XOR-hash indices for each embedding table.

        Args:
          tokens: Token IDs with sequence on the last axis.

        Returns:
          indices: One tensor of bucket IDs per table, shaped like ``tokens``.

        """
        order = len(self.hash_multipliers[0])
        shifted = [
            torch.cat((tokens[..., :lag], tokens[..., :-lag]), dim=-1)
            if lag
            else tokens
            for lag in reversed(range(order))
        ]
        indices: list[Tensor] = []
        for hash_multipliers in self.hash_multipliers:
            index = shifted[0] * hash_multipliers[0]
            for previous, prime in zip(shifted[1:], hash_multipliers[1:], strict=True):
                index = index ^ (previous * prime)
            indices.append(index % self.num_embeddings)
        return indices


class NgramSource(NamedTuple):
    """Describe one fused n-gram source and its precomputed indices."""

    gate_index: int
    table: HashedNgramTables
    indices: list[Tensor]


@torch.library.custom_op("priml_nanochat::ngram_mix", mutates_args=())
def ngram_mix(  # noqa: PLR0917 -- The operator schema fixes the positional arity.
    v: Tensor,
    gates: list[Tensor],
    weights: list[Tensor],
    indices: list[Tensor],
    sinks: list[Tensor],
    bitmaps: list[Tensor],
) -> Tensor:
    """Add gated n-gram values and accumulate table gradients in FP32 buffers.

    Args:
      v: Contiguous values shaped ``[batch, tokens, heads, channels]``.
      gates: Per-head gate logits for one or two n-gram sources.
      weights: Two half-width embedding tables per source.
      indices: Flattenable int64 lookup indices for each table.
      sinks: Persistent FP32 gradient buffers matching the tables.
      bitmaps: Optional row flags written by backward for selective updates.

    Returns:
      values: Mixed values with the shape and dtype of ``v``.

    """
    del bitmaps  # Part of the schema; only the BACKWARD writes it.
    _check_mix(v, gates, weights, indices, sinks)
    if v.is_cuda:
        return _mix_forward_cuda(v, gates, weights, indices)
    b, t, h, d = v.shape
    out = v.float()
    for source, gate in enumerate(gates):
        rows = torch.cat(
            [
                weights[2 * source + j][indices[2 * source + j].reshape(-1)]
                for j in (0, 1)
            ],
            dim=-1,
        ).reshape(b, t, h, d)
        out = out + 2 * torch.sigmoid(gate.float()).reshape(b, t, h, 1) * rows.float()
    return out.to(v.dtype)


def _check_mix(
    v: Tensor,
    gates: list[Tensor],
    weights: list[Tensor],
    indices: list[Tensor],
    sinks: list[Tensor],
) -> None:
    assert 1 <= len(gates) <= 2
    assert len(weights) == len(indices) == len(sinks) == 2 * len(gates)
    assert v.ndim == 4
    assert v.is_contiguous()
    b, t, h, d = v.shape
    assert weights[0].shape[1] * 2 == h * d
    for w, s in zip(weights, sinks, strict=True):
        assert w.shape == s.shape
        assert w.is_contiguous()
        assert s.is_contiguous()
        assert s.dtype == torch.float32
        assert w.dtype == v.dtype
    for index in indices:
        assert index.numel() == b * t
        assert index.dtype == torch.int64
        assert index.is_contiguous()
    for gate in gates:
        assert gate.shape[-1] == h
        assert gate.numel() == b * t * h
        assert gate.is_contiguous()


@torch.library.custom_op(
    "priml_nanochat::ngram_backward",
    mutates_args=("sinks", "bitmaps"),
)
def _mix_backward(  # noqa: PLR0917 -- The operator schema fixes the positional arity.
    dv: Tensor,
    gates: list[Tensor],
    weights: list[Tensor],
    indices: list[Tensor],
    sinks: list[Tensor],
    bitmaps: list[Tensor],
) -> "tuple[Tensor, Tensor]":
    dv = dv.contiguous()
    _check_mix(dv, gates, weights, indices, sinks)
    if dv.is_cuda:
        grads = _mix_backward_cuda(
            dv,
            gates,
            weights,
            indices,
            sinks,
            bitmaps=bitmaps or None,
        )
    else:
        grads = _mix_backward_reference(dv, gates, weights, indices, sinks)
        for bitmap, index in zip(bitmaps, indices, strict=False):
            flat = index.reshape(-1)
            bitmap[flat[flat != -1]] = 1
    # A list return prevents auto-functionalization of this mutating custom op.
    return grads[0], grads[1] if len(grads) == 2 else gates[0].new_empty(0)


def _mix_backward_reference(
    dv: Tensor,
    gates: list[Tensor],
    weights: list[Tensor],
    indices: list[Tensor],
    sinks: list[Tensor],
) -> list[Tensor]:
    b, t, h, d = dv.shape
    half = weights[0].shape[1]
    grads: list[Tensor] = []
    for source, gate in enumerate(gates):
        sig = torch.sigmoid(gate.float()).reshape(b, t, h)
        rows = (
            torch.cat(
                [
                    weights[2 * source + j][indices[2 * source + j].reshape(-1)]
                    for j in (0, 1)
                ],
                dim=-1,
            )
            .reshape(b, t, h, d)
            .float()
        )
        grads.append(
            ((dv.float() * rows).sum(-1) * (2 * sig * (1 - sig)))
            .to(gate.dtype)
            .reshape(gate.shape),
        )
        contribution = (dv.float() * (2 * sig).unsqueeze(-1)).reshape(-1, h * d)
        for j in (0, 1):
            index = indices[2 * source + j].reshape(-1)
            sinks[2 * source + j].index_add_(
                0,
                index,
                contribution[:, j * half : (j + 1) * half],
            )
    return grads


def _mix_setup(
    ctx: _FusedContext,
    inputs: tuple[
        Tensor,
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[Tensor],
    ],
    output: object,
) -> None:
    del output
    _, gates, weights, indices, sinks, bitmaps = inputs
    ctx.sources = len(gates)
    ctx.marked = len(bitmaps)
    ctx.save_for_backward(*gates, *weights, *indices, *sinks, *bitmaps)


def _register_mix_autograd[FunctionT: Callable[..., object]](
    function: FunctionT,
) -> FunctionT:
    ngram_mix.register_autograd(function, setup_context=_mix_setup)
    return function


def clear_marked_sinks(sinks: list[Tensor], bitmaps: list[Tensor]) -> None:
    """Clear marked gradient rows and their flags after the optimizer consumes them.

    Args:
      sinks: Persistent table-gradient buffers.
      bitmaps: One row-flag tensor per buffer.

    """
    if not sinks:
        return
    if sinks[0].is_cuda:
        _clear_marked_sinks_cuda(sinks, bitmaps)
        return
    for sink, bitmap in zip(sinks, bitmaps, strict=True):
        marked = bitmap != 0
        sink[marked] = 0
        bitmap.zero_()


def _mix_forward_cuda(
    v: Tensor,
    gates: list[Tensor],
    weights: list[Tensor],
    indices: list[Tensor],
) -> Tensor:
    """Gather and mix up to two factored n-gram sources in one launch."""
    (b, t, h, d) = v.shape
    half = weights[0].shape[1]
    tile = math.gcd(d, half)
    (g, w, i) = (
        _pad_ngram_sources(gates, 2),
        _pad_ngram_sources(weights, 4),
        _pad_ngram_sources(indices, 4),
    )
    out = torch.empty_like(v)
    _compiled_ngram_forward()[triton.cdiv(b * t, 16),](
        buffers=(v, out, g[0], g[1], *w, *i),
        n_rows=b * t,
        n_head=h,
        dimensions=(half, tile, h * d // tile, d, len(gates)),
        block=16,
        num_warps=4,
        num_stages=1,
    )
    return out


def _mix_backward_cuda(
    dv: Tensor,
    gates: list[Tensor],
    weights: list[Tensor],
    indices: list[Tensor],
    sinks: list[Tensor],
    *,
    bitmaps: list[Tensor] | None = None,
) -> list[Tensor]:
    """Compute gate gradients, scatter table gradients, and optionally mark rows."""
    (b, t, h, d) = dv.shape
    half = weights[0].shape[1]
    tile = math.gcd(d, half)
    grads = [torch.empty_like(g) for g in gates]
    (g, w, i, s, dg) = (
        _pad_ngram_sources(gates, 2),
        _pad_ngram_sources(weights, 4),
        _pad_ngram_sources(indices, 4),
        _pad_ngram_sources(sinks, 4),
        _pad_ngram_sources(grads, 2),
    )
    mark = bitmaps is not None
    bm = (
        _pad_ngram_sources(list(bitmaps), 4)
        if bitmaps is not None
        else _pad_ngram_sources(list(sinks), 4)
    )
    _compiled_ngram_backward()[triton.cdiv(b * t, 16),](
        buffers=(dv.contiguous(), g[0], g[1], *w, *i, dg[0], dg[1], *s, *bm),
        mark=mark,
        n_rows=b * t,
        n_head=h,
        dimensions=(half, tile, h * d // tile, d, len(gates)),
        block=16,
        num_warps=4,
        num_stages=1,
    )
    return grads


def _pad_ngram_sources(values: list[Tensor], length: int) -> list[Tensor]:
    return values + [values[0]] * (length - len(values))


def _jit_kernel(function: Callable[..., None]) -> "triton.JITFunction[..., None]":
    """Bind concrete language modules before Triton hashes and compiles the function."""
    assert isinstance(function, FunctionType)
    bound = FunctionType(
        function.__code__,
        function.__globals__
        | {
            "language": import_module("triton.language"),
            "libdevice": import_module("triton.language.extra.cuda.libdevice"),
        },
        function.__name__,
        function.__defaults__,
    )
    bound.__annotations__ = function.__annotations__
    return triton.jit(bound)


@lru_cache(maxsize=1)
def _compiled_ngram_forward() -> "triton.JITFunction[..., None]":
    return _jit_kernel(_ngram_mix_fwd_kernel)


def _ngram_mix_fwd_kernel(
    buffers: "tuple[language.tensor, ...]",
    n_rows: int,
    n_head: int,
    dimensions: "tuple[language.constexpr, ...]",
    block: "language.constexpr",
) -> None:
    """Gather factored n-gram tables and add gated values in one pass."""
    (
        v_ptr,
        out_ptr,
        gl0_ptr,
        gl1_ptr,
        w0_ptr,
        w1_ptr,
        w2_ptr,
        w3_ptr,
        i0_ptr,
        i1_ptr,
        i2_ptr,
        i3_ptr,
    ) = buffers
    half: language.constexpr = dimensions[0]
    tile: language.constexpr = dimensions[1]
    nt: language.constexpr = dimensions[2]
    hd: language.constexpr = dimensions[3]
    ns: language.constexpr = dimensions[4]
    pid = language.program_id(0)
    rows = pid * block + language.arange(0, block)
    rmask = rows < n_rows
    j = language.arange(0, tile)
    rm2 = rmask[:, None]
    i0 = language.load(i0_ptr + rows, mask=rmask, other=0)
    i1 = language.load(i1_ptr + rows, mask=rmask, other=0)
    (i2, i3) = (i0, i1)
    if ns > 1:
        i2 = language.load(i2_ptr + rows, mask=rmask, other=0)
        i3 = language.load(i3_ptr + rows, mask=rmask, other=0)
    for t in language.static_range(nt):
        head = t * tile // hd
        tab = t * tile // half
        loc = t * tile - tab * half
        voff = rows[:, None] * (nt * tile) + (t * tile + j[None, :])
        acc = language.load(v_ptr + voff, mask=rm2, other=0.0).to(language.float32)
        g0 = language.load(gl0_ptr + rows * n_head + head, mask=rmask, other=0.0).to(
            language.float32,
        )
        gate0 = 2.0 * language.sigmoid(g0)
        idx0 = i0 if tab == 0 else i1
        wp0 = w0_ptr if tab == 0 else w1_ptr
        w0v = language.load(
            wp0 + (idx0[:, None] * half + (loc + j[None, :])),
            mask=rm2,
            other=0.0,
        )
        acc += gate0[:, None] * w0v.to(language.float32)
        if ns > 1:
            g1 = language.load(
                gl1_ptr + rows * n_head + head,
                mask=rmask,
                other=0.0,
            ).to(language.float32)
            gate1 = 2.0 * language.sigmoid(g1)
            idx1 = i2 if tab == 0 else i3
            wp1 = w2_ptr if tab == 0 else w3_ptr
            w1v = language.load(
                wp1 + (idx1[:, None] * half + (loc + j[None, :])),
                mask=rm2,
                other=0.0,
            )
            acc += gate1[:, None] * w1v.to(language.float32)
        language.store(out_ptr + voff, acc, mask=rm2)


@lru_cache(maxsize=1)
def _compiled_ngram_backward() -> "triton.JITFunction[..., None]":
    return _jit_kernel(_ngram_mix_bwd_kernel)


def _ngram_mix_bwd_kernel(  # noqa: PLR0917 -- Triton JIT binds the kernel operands by position.
    buffers: "tuple[language.tensor, ...]",
    n_rows: int,
    n_head: int,
    dimensions: "tuple[language.constexpr, ...]",
    block: "language.constexpr",
    mark: "language.constexpr" = False,
) -> None:
    """Compute gate gradients and atomically scatter FP32 table gradients."""
    (
        dv_ptr,
        gl0_ptr,
        gl1_ptr,
        w0_ptr,
        w1_ptr,
        w2_ptr,
        w3_ptr,
        i0_ptr,
        i1_ptr,
        i2_ptr,
        i3_ptr,
        dg0_ptr,
        dg1_ptr,
        s0_ptr,
        s1_ptr,
        s2_ptr,
        s3_ptr,
        b0_ptr,
        b1_ptr,
        b2_ptr,
        b3_ptr,
    ) = buffers
    half: language.constexpr = dimensions[0]
    tile: language.constexpr = dimensions[1]
    nt: language.constexpr = dimensions[2]
    hd: language.constexpr = dimensions[3]
    ns: language.constexpr = dimensions[4]
    pid = language.program_id(0)
    rows = pid * block + language.arange(0, block)
    rmask = rows < n_rows
    j = language.arange(0, tile)
    rm2 = rmask[:, None]
    subs: language.constexpr = hd // tile
    i0 = language.load(i0_ptr + rows, mask=rmask, other=0)
    i1 = language.load(i1_ptr + rows, mask=rmask, other=0)
    mark0a = rmask & (i0 != -1)
    mark0b = rmask & (i1 != -1)
    keep0a = language.broadcast_to(mark0a[:, None], [block, tile])
    keep0b = language.broadcast_to(mark0b[:, None], [block, tile])
    (i2, i3) = (i0, i1)
    (keep1a, keep1b) = (keep0a, keep0b)
    (mark1a, mark1b) = (mark0a, mark0b)
    if ns > 1:
        i2 = language.load(i2_ptr + rows, mask=rmask, other=0)
        i3 = language.load(i3_ptr + rows, mask=rmask, other=0)
        mark1a = rmask & (i2 != -1)
        mark1b = rmask & (i3 != -1)
        keep1a = language.broadcast_to(mark1a[:, None], [block, tile])
        keep1b = language.broadcast_to(mark1b[:, None], [block, tile])
    # Mark under the same row predicate as the gradient atomic, once before channel
    # loops.
    if mark:
        one_u8 = language.full((block,), 1, language.uint8)
        language.store(b0_ptr + i0, one_u8, mask=mark0a)
        language.store(b1_ptr + i1, one_u8, mask=mark0b)
        if ns > 1:
            language.store(b2_ptr + i2, one_u8, mask=mark1a)
            language.store(b3_ptr + i3, one_u8, mask=mark1b)
    for h in language.static_range(nt // subs):
        g0 = language.load(gl0_ptr + rows * n_head + h, mask=rmask, other=0.0).to(
            language.float32,
        )
        s0 = language.sigmoid(g0)
        gate0 = 2.0 * s0
        acc0 = language.zeros([block], language.float32)
        (s1, gate1, acc1) = (s0, gate0, acc0)
        if ns > 1:
            g1 = language.load(gl1_ptr + rows * n_head + h, mask=rmask, other=0.0).to(
                language.float32,
            )
            s1 = language.sigmoid(g1)
            gate1 = 2.0 * s1
            acc1 = language.zeros([block], language.float32)
        for sub in language.static_range(subs):
            t = h * subs + sub
            tab = t * tile // half
            loc = t * tile - tab * half
            voff = rows[:, None] * (nt * tile) + (t * tile + j[None, :])
            dv = language.load(dv_ptr + voff, mask=rm2, other=0.0).to(language.float32)
            idx0 = i0 if tab == 0 else i1
            wp0 = w0_ptr if tab == 0 else w1_ptr
            sp0 = s0_ptr if tab == 0 else s1_ptr
            k0 = keep0a if tab == 0 else keep0b
            woff0 = idx0[:, None] * half + (loc + j[None, :])
            w0v = language.load(wp0 + woff0, mask=rm2, other=0.0).to(language.float32)
            acc0 += language.sum(dv * w0v, 1)
            language.atomic_add(
                sp0 + woff0,
                dv * gate0[:, None],
                mask=k0,
                sem="relaxed",
            )
            if ns > 1:
                idx1 = i2 if tab == 0 else i3
                wp1 = w2_ptr if tab == 0 else w3_ptr
                sp1 = s2_ptr if tab == 0 else s3_ptr
                k1 = keep1a if tab == 0 else keep1b
                woff1 = idx1[:, None] * half + (loc + j[None, :])
                w1v = language.load(wp1 + woff1, mask=rm2, other=0.0).to(
                    language.float32,
                )
                acc1 += language.sum(dv * w1v, 1)
                language.atomic_add(
                    sp1 + woff1,
                    dv * gate1[:, None],
                    mask=k1,
                    sem="relaxed",
                )
        language.store(
            dg0_ptr + rows * n_head + h,
            acc0 * (2.0 * s0 * (1.0 - s0)),
            mask=rmask,
        )
        if ns > 1:
            language.store(
                dg1_ptr + rows * n_head + h,
                acc1 * (2.0 * s1 * (1.0 - s1)),
                mask=rmask,
            )


@lru_cache(maxsize=1)
def _compiled_sink_clear() -> "triton.JITFunction[..., None]":
    return _jit_kernel(_clear_marked_sink_kernel)


def _clear_marked_sink_kernel(
    buffers: "tuple[language.tensor, ...]",
    n_cols: int,
    block: "language.constexpr",
    rows_per_program: "language.constexpr",
) -> None:
    """Clear marked rows with one warp; multiple warps can race on flag reset."""
    (sink_ptr, bitmap_ptr) = buffers
    pid = language.program_id(0)
    cols = language.arange(0, block)
    mask = cols < n_cols
    for k in language.static_range(rows_per_program):
        row = pid * rows_per_program + k
        if language.load(bitmap_ptr + row) != 0:
            language.store(sink_ptr + row * n_cols + cols, 0.0, mask=mask)
            language.store(bitmap_ptr + row, language.full((), 0, language.uint8))


def _clear_marked_sinks_cuda(
    sinks: list[Tensor],
    bitmaps: list[Tensor],
    rows_per_program: int = 8,
) -> None:
    """Clear marked rows in every fused table, replacing dense ``sink.zero_()``."""
    for sink, bitmap in zip(sinks, bitmaps, strict=True):
        (rows, cols) = sink.shape
        if rows % rows_per_program:
            raise ValueError(
                f"{rows} rows is not divisible by {rows_per_program}; the clear omits a "
                "bounds mask and would read past the table",
            )
        _compiled_sink_clear()[rows // rows_per_program,](
            buffers=(sink, bitmap),
            n_cols=cols,
            block=triton.next_power_of_2(cols),
            rows_per_program=rows_per_program,
            num_warps=1,
        )


@ngram_mix.register_fake
def _mix_fake(  # noqa: PLR0917 -- The operator schema fixes the positional arity.
    v: Tensor,
    gates: list[Tensor],
    weights: list[Tensor],
    indices: list[Tensor],
    sinks: list[Tensor],
    bitmaps: list[Tensor],
) -> Tensor:
    del gates, weights, indices, sinks, bitmaps
    return torch.empty_like(v)


@_mix_backward.register_fake
def _mix_backward_fake(  # noqa: PLR0917 -- The operator schema fixes the positional arity.
    dv: Tensor,
    gates: list[Tensor],
    weights: list[Tensor],
    indices: list[Tensor],
    sinks: list[Tensor],
    bitmaps: list[Tensor],
) -> "tuple[Tensor, Tensor]":
    del dv, weights, indices, sinks, bitmaps
    return torch.empty_like(gates[0]), torch.empty_like(gates[1]) if len(
        gates,
    ) == 2 else gates[0].new_empty(0)


@_register_mix_autograd
def _mix_autograd(
    ctx: _FusedContext,
    gradient: Tensor,
) -> "tuple[Tensor, list[Tensor], list[None], list[None], list[None], list[None]]":
    saved = list(ctx.saved_tensors)
    n = ctx.sources
    grads = _mix_backward(
        gradient,
        saved[:n],
        saved[n : 3 * n],
        saved[3 * n : 5 * n],
        saved[5 * n : 7 * n],
        saved[7 * n : 7 * n + ctx.marked],
    )
    # None suppresses BF16 embedding-gradient materialization; RMSProp reads sinks.
    return (
        gradient,
        list(grads[:n]),
        [None] * (2 * n),
        [None] * (2 * n),
        [None] * (2 * n),
        [None] * ctx.marked,
    )
