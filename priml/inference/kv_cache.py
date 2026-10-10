"""Analytical cost of one decode step, read from a KV-cache geometry.

A decode step at context ``T`` reads every cached token, writes the new one's
keys and values, and runs two matmuls per query head over all ``T + 1`` keys,
its own included (``q @ K^T`` and ``p @ V``). Over ``L`` layers, ``B``
sequences, ``H`` query heads, ``Hkv`` key/value heads, and head width ``D``:

    flops = 4 * L * B * (T + 1) * H * D
    bytes = 2 * L * B * (T + 1) * Hkv * D * itemsize

The ratio is ``2 * (H / Hkv) / itemsize`` at every context: 1 FLOP/byte for
multi-head attention at bfloat16, about 295x below an H100's ridge.
Grouped-query attention raises it by exactly the grouping ratio and a narrower
cache by the itemsize it saves; nothing else does.

Reading ``P`` parameters once per step adds ``2 * B * P`` FLOPs over
``P * itemsize`` bytes at the compute dtype, intensity ``2 * B / itemsize``.
That term does not grow with context but does grow with batch: weight-dominated
decode turns compute-bound once ``B`` passes ``ridge * itemsize / 2`` (about
295 on an H100 at bfloat16), while the cache term stays memory-bound short of a
grouping ratio near the ridge.

``priml.cost`` owns the device table; this reads its ridge rather than
restating it. Activation traffic and prefill are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass

from configgle import Fig

import torch

from priml.cost import Device, peak, resolve_dtype


__all__ = ["DecodeCost", "KVCacheGeometry"]


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodeCost:
    """One decode step's work and traffic, and the ridge it is measured against."""

    flops: int
    """Attention and weight arithmetic for the whole batch."""

    kv_bytes: int
    """Cache traffic for the whole batch: every cached token read, one written."""

    weight_bytes: int
    """Parameter bytes, read once per step whatever the batch or context."""

    ridge: float
    """The device's intensity ceiling at the compute dtype, from ``priml.cost``."""

    @property
    def intensity(self) -> float:
        """FLOP per byte over the whole step, cache and weights together."""
        return self.flops / (self.kv_bytes + self.weight_bytes)

    @property
    def memory_bound(self) -> bool:
        """Whether the step sits below the ridge, waiting on memory."""
        return self.intensity < self.ridge

    @property
    def fraction_of_ridge(self) -> float:
        """``intensity / ridge``; 1.0 is exactly at the crossover."""
        return self.intensity / self.ridge


class KVCacheGeometry(Fig["KVCacheGeometry"]):
    """The geometry of a KV cache, and what one decode step costs against it.

    Every count defaults to -1, unset; pricing an unset or inconsistent
    geometry raises rather than multiplying sentinels into a plausible number.
    """

    num_layers: int = -1
    """Attention layers. Each keeps its own keys and values."""

    num_heads: int = -1
    """Query heads."""

    num_heads_kv: int = -1
    """Key/value heads; fewer than ``num_heads`` is grouped-query attention."""

    channels_head: int = -1
    """Per-head width of each key and value."""

    dtype: torch.dtype | None = None
    """Storage dtype of the cache entries (``None`` is torch's default).

    Not the compute dtype: an int8 cache dequantizes into a bfloat16 matmul,
    so ``decode_cost`` takes the compute dtype separately."""

    @property
    def grouping_ratio(self) -> float:
        """Query heads per key/value head; 1.0 for multi-head attention."""
        self._validate()
        return self.num_heads / self.num_heads_kv

    def bytes_per_token(self) -> int:
        """Bytes one token of context costs in the cache, across all layers."""
        self._validate()
        return (
            2
            * self.num_layers
            * self.num_heads_kv
            * self.channels_head
            * resolve_dtype(self.dtype).itemsize
        )

    def tokens_in(self, num_bytes: int) -> int:
        """How many tokens of context fit in ``num_bytes`` of cache."""
        return max(0, num_bytes // self.bytes_per_token())

    def decode_cost(
        self,
        *,
        batch_size: int,
        context_len: int,
        device: Device | str,
        dtype: torch.dtype | None,
        num_params: int = 0,
    ) -> DecodeCost:
        """Price one decode step against a device's ridge.

        Args:
          batch_size: Sequences decoded in lockstep.
          context_len: Tokens each sequence has cached before this step.
          device: Device name or a form-qualified name such as ``"h100-pcie"``.
          dtype: Compute dtype of the matmuls and the stored weights (``None``
            is torch's default); the ridge is read here, not at the cache's.
          num_params: Parameters read once per step; 0 isolates the cache.

        Returns:
          cost: The step's FLOPs, cache and weight bytes, and ridge.

        Raises:
          KeyError: ``device`` is not in ``priml.cost``'s table.
          ValueError: The geometry or an argument is invalid, or ``device``
            has no matmul rate at ``dtype``.

        """
        self._validate()
        if batch_size < 1 or context_len < 0 or num_params < 0:
            raise ValueError(
                "batch_size must be positive and context_len and num_params "
                f"nonnegative; got {batch_size}, {context_len}, {num_params}.",
            )
        compute = resolve_dtype(dtype)
        ridge = peak()[device, compute, "intensity", "matmul"]
        if ridge == 0:
            raise ValueError(f"{device} has no {compute} matmul rate.")
        attention_flops = (
            4
            * self.num_layers
            * batch_size
            * (context_len + 1)
            * self.num_heads
            * self.channels_head
        )
        return DecodeCost(
            flops=attention_flops + 2 * batch_size * num_params,
            kv_bytes=self.bytes_per_token() * batch_size * (context_len + 1),
            weight_bytes=num_params * compute.itemsize,
            ridge=ridge,
        )

    def _validate(self) -> None:
        for name, value in (
            ("num_layers", self.num_layers),
            ("num_heads", self.num_heads),
            ("num_heads_kv", self.num_heads_kv),
            ("channels_head", self.channels_head),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive; got {value}.")
        if self.num_heads % self.num_heads_kv:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be a multiple of "
                f"num_heads_kv ({self.num_heads_kv}).",
            )
