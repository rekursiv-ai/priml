"""Static pre-allocated key-value cache for autoregressive generation."""

from __future__ import annotations

from typing import override

from torch import Tensor

import torch


class KVCache:
    """Static pre-allocated key-value cache for autoregressive generation.

    Pre-allocates tensors of shape [..., H, max_seq, D] and writes new
    KV pairs at an offset. No copies per token — O(1) update.

    FIFO uses sliced copy (not torch.roll) for torch.compile compat.
    For maximum throughput, HuggingFace and vLLM use custom CUDA
    kernels (PagedAttention, etc.).
    """

    __slots__ = ("k", "length", "seen", "v")

    def __init__(self, k: Tensor, v: Tensor, length: int | None = None) -> None:
        self.k = k
        self.v = v
        self.length = length if length is not None else k.shape[-2]
        # Total tokens ever written. Unlike ``length`` (capped at
        # ``max_seq`` by FIFO eviction) ``seen`` grows monotonically, so
        # RoPE can keep assigning absolute positions past capacity.
        self.seen = self.length

    @classmethod
    def alloc(
        cls,
        *,
        batch: int | tuple[int, ...],
        heads: int,
        max_seq: int,
        channels_head: int,
        channels_v_head: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        """Pre-allocate an empty cache.

        ``channels_v_head`` defaults to ``channels_head`` (symmetric K/V,
        standard MHA/GQA). MLA needs independent dims (K concatenates
        qk_nope + qk_rope; V uses a separate channels_v_head).
        """
        if isinstance(batch, int):
            batch = (batch,)
        v_dim = channels_head if channels_v_head is None else channels_v_head
        k_shape = (*batch, heads, max_seq, channels_head)
        v_shape = (*batch, heads, max_seq, v_dim)
        return cls(
            k=torch.zeros(k_shape, device=device, dtype=dtype),
            v=torch.zeros(v_shape, device=device, dtype=dtype),
            length=0,
        )

    @property
    def max_seq(self) -> int:
        return self.k.shape[-2]

    def freeze(self) -> KVCache:
        """Return a frozen snapshot (update becomes a no-op)."""
        frozen = _FrozenKVCache(self.k, self.v, self.length)
        # Preserve the monotonic total so post-freeze RoPE keeps assigning
        # correct absolute positions; the constructor reset it to ``length``.
        frozen.seen = self.seen
        return frozen

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write new KV pairs into the cache and return the valid slice.

        If the cache is full, shifts old entries out (FIFO) to make room.
        Uses sliced copy instead of torch.roll for torch.compile compat.
        """
        s = k.shape[-2]
        if s > self.max_seq:
            raise ValueError(
                f"Update length {s} exceeds cache capacity {self.max_seq}.",
            )
        overflow = self.length + s - self.max_seq
        if overflow > 0:
            keep = self.length - overflow
            self.k[..., :keep, :] = self.k[..., overflow : self.length, :].clone()
            self.v[..., :keep, :] = self.v[..., overflow : self.length, :].clone()
            self.length = keep
        end = self.length + s
        self.k[..., self.length : end, :] = k
        self.v[..., self.length : end, :] = v
        self.length = end
        self.seen += s
        return self.k[..., :end, :], self.v[..., :end, :]


class _FrozenKVCache(KVCache):
    """KV cache where update is a no-op (for CFG / multiple forward passes)."""

    @override
    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        del k, v
        return self.k[..., : self.length, :], self.v[..., : self.length, :]
