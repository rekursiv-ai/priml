"""INT8 per-token KV cache: the storage ``exp001`` swaps into the decode chain.

Every token's row of keys, and of values, is quantized symmetrically to
signed int8 against one scale computed from that row's absolute maximum.
The float32 scale's bits ride in the row's trailing columns, so ``k`` and
``v`` alone carry the whole cache and :func:`priml.baselines.kvcache.decode`
measured ``kv_bytes`` counts codes and scales together.

Attention never sees the codes: :meth:`Int8KVCache.update` returns the
dequantized valid slice in the incoming dtype, shaped exactly like the
float cache's answer, so the model's read path is unchanged.

The cache refuses :meth:`Int8KVCache.freeze`; a frozen view of the base
class would hand int8 codes and scale bits straight to attention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, override

from torch import Tensor

import torch

from priml.model.attention.kvcache import KVCache, alloc_layer_cache


if TYPE_CHECKING:
    from priml.model.attention.kvcache import HasModules
    from priml.model.custom_types import DepthIndex


SCALE_COLUMNS: Final = 4
"""Trailing int8 columns per row holding one float32 scale, bit for bit."""

_CODE_MAX: Final = 127.0
"""Magnitude of the largest code: symmetric int8 range is -127..127."""


class Int8KVCache(KVCache):
    """KV cache storing int8 codes and one packed scale per token.

    ``k`` is int8 ``[..., H, max_seq, D + SCALE_COLUMNS]``: the first ``D``
    columns hold one token's key codes, the last ``SCALE_COLUMNS`` that
    row's float32 scale, bit for bit. ``v`` is the same over its own
    channel count. Rows stay whole under FIFO eviction, so codes and scales
    always shift together.
    """

    __slots__ = ()

    @classmethod
    @override
    def alloc(
        cls,
        *,
        batch: int | tuple[int, ...],
        num_heads: int,
        max_seq: int,
        channels_head: int,
        channels_v_head: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        """Allocate packed int8 rows, including each row's float32 scale.

        Args:
          batch: Batch size or batch shape.
          num_heads: Number of cached heads.
          max_seq: Maximum sequence length.
          channels_head: Key channels per head, excluding the packed scale.
          channels_v_head: Value channels per head; defaults to key channels.
          device: Device for allocated tensors.
          dtype: Accepted for the base allocator contract; storage is int8.

        Returns:
          cache: An empty packed cache with zero length and tokens seen.

        """
        del dtype
        channels_v = channels_head if channels_v_head is None else channels_v_head
        for name, channels in (
            ("channels_head", channels_head),
            ("channels_v_head", channels_v),
        ):
            # The scale columns would turn a negative width into a positive one.
            if channels < 0:
                raise ValueError(f"{name} must be non-negative; got {channels}.")
        return super().alloc(
            batch=batch,
            num_heads=num_heads,
            max_seq=max_seq,
            channels_head=channels_head + SCALE_COLUMNS,
            channels_v_head=channels_v + SCALE_COLUMNS,
            device=device,
            dtype=torch.int8,
        )

    @override
    def freeze(self) -> KVCache:
        """Refuse a frozen view, which would serve raw codes to attention.

        Raises:
          NotImplementedError: Always; this cache has no float view to freeze.

        """
        raise NotImplementedError(
            "Int8KVCache cannot freeze: the shared-storage view would pass "
            "int8 codes and scale bits to attention unquantized.",
        )

    @override
    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize ``k`` and ``v`` into the cache, return the valid slice.

        Same contract as :meth:`KVCache.update`: the cache keeps the most
        recent ``max_seq`` tokens (FIFO when an update overflows), ``seen``
        counts every token ever written, and a rejected update leaves the
        cache as it was.

        Args:
          k: New keys ``[..., seq_len, channels_head]``, any float dtype.
          v: New values ``[..., seq_len, channels_v_head]``, likewise.

        Returns:
          k: Dequantized keys ``[..., length, channels_head]`` in ``k``'s dtype.
          v: Dequantized values ``[..., length, channels_v_head]`` in ``v``'s dtype.

        Raises:
          ValueError: ``k`` and ``v`` differ in length, or either cannot be
            written into the cache.

        """
        s = k.shape[-2]
        if v.shape[-2] != s:
            raise ValueError(
                f"Got {s} keys and {v.shape[-2]} values; keys and values pair "
                "up one per token.",
            )
        if s > self.max_seq:
            raise ValueError(
                f"Update length {s} exceeds cache capacity {self.max_seq}.",
            )
        packed: list[Tensor] = []
        for new, store in ((k, self.k), (v, self.v)):
            slot = store[..., :s, :-SCALE_COLUMNS].shape
            try:
                fits = torch.broadcast_shapes(new.shape, slot) == slot
            except RuntimeError:
                fits = False
            if not fits:
                raise ValueError(
                    f"Cannot write {tuple(new.shape)} into the cache's "
                    f"{tuple(slot)} slot.",
                )
            # Moved here, before eviction, so a value that cannot reach the
            # cache's device fails with the cache untouched.
            packed.append(_quantize(new.expand(slot)).to(store.device))
        overflow = self.length + s - self.max_seq
        if overflow > 0:
            keep = self.length - overflow
            self.k[..., :keep, :] = self.k[..., overflow : self.length, :].clone()
            self.v[..., :keep, :] = self.v[..., overflow : self.length, :].clone()
            self.length = keep
        end = self.length + s
        self.k[..., self.length : end, :] = packed[0]
        self.v[..., self.length : end, :] = packed[1]
        self.length = end
        self.seen += s
        return (
            _dequantized(self.k, end=end, dtype=k.dtype),
            _dequantized(self.v, end=end, dtype=v.dtype),
        )


def int8_layer_cache(
    model: HasModules,
    *,
    batch: int | tuple[int, ...],
    max_seq: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[DepthIndex, object]:
    """Allocate one :class:`Int8KVCache` per cached attention in ``model``.

    Each slot mirrors :func:`alloc_layer_cache` -- same keys, same shapes --
    widened by ``SCALE_COLUMNS`` so the scales live inside ``k`` and ``v``
    and the benchmark's byte count sees them.

    Args:
      model: Module tree containing cached attentions.
      batch: Batch size or batch shape.
      max_seq: Maximum sequence length.
      device: Device for allocated tensors.
      dtype: Accepted for the ``allocate_cache`` slot and ignored; storage
        is int8 by definition.

    Returns:
      cache: Mutable state keyed by each attention's full depth index.

    """
    del dtype
    result: dict[DepthIndex, object] = {}
    allocated = alloc_layer_cache(
        model,
        batch=batch,
        max_seq=0,
        device=device,
    )
    for key, state in allocated.items():
        assert isinstance(state, KVCache)
        result[key] = Int8KVCache.alloc(
            batch=tuple(state.k.shape[:-3]),
            num_heads=state.k.shape[-3],
            max_seq=max_seq,
            channels_head=state.k.shape[-1],
            channels_v_head=state.v.shape[-1],
            device=state.k.device,
        )
    return result


def _quantize(x: Tensor) -> Tensor:
    """Pack a float row as int8 codes followed by its float32 scale."""
    values = x.to(dtype=_working_dtype(x.dtype))
    # A zero-width row (MLA without a rotary key) has no peak to reduce.
    peak = (
        values.abs().amax(dim=-1, keepdim=True)
        if values.shape[-1]
        else values.new_zeros((*values.shape[:-1], 1))
    )
    scale = peak.div(_CODE_MAX)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = torch.round(values.div(scale)).clamp_(-_CODE_MAX, _CODE_MAX)
    codes = codes.to(torch.int8)
    return torch.cat((codes, scale.float().view(torch.int8)), dim=-1)


def _dequantized(store: Tensor, *, end: int, dtype: torch.dtype) -> Tensor:
    """Read the cache's first ``end`` rows back as floats in ``dtype``."""
    codes = store[..., :end, :-SCALE_COLUMNS]
    # A plain clone preserves singleton strides; dtype views also require
    # divisible strides on those axes, even when no rows are valid yet.
    scale = (
        store[..., :end, -SCALE_COLUMNS:]
        .clone(memory_format=torch.contiguous_format)
        .view(torch.float32)
    )
    return (codes.to(_working_dtype(dtype)) * scale).to(dtype)


# Every dtype narrower than float32 promotes to it; float8 cannot be asked.
def _working_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return ``dtype`` or float32, whichever is wider."""
    if dtype.itemsize < torch.float32.itemsize:
        return torch.float32
    return torch.promote_types(dtype, torch.float32)
