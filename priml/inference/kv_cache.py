"""FP4 Key-Value Cache for speculative decoding.

This module provides FP4 quantized KV cache implementations
for efficient speculative decoding with quantized models.
"""

from dataclasses import dataclass

import torch


@dataclass
class FP4KVCache:
    """FP4 quantized Key-Value cache.

    Stores quantized key and value tensors for attention layers.
    Used in speculative decoding to reduce memory bandwidth.
    """

    # Quantized keys: [num_layers, batch, num_heads, seq_len, head_dim]
    keys: torch.Tensor | None = None
    # Quantized values: [num_layers, batch, num_heads, seq_len, head_dim]
    values: torch.Tensor | None = None
    # Scale factors for dequantization
    key_scales: torch.Tensor | None = None
    value_scales: torch.Tensor | None = None
    # Block size for NVFP4 format (16 for block quantization, 1 for per-tensor)
    block_size: int = 1
    # FP4 format: "e2m1" or "nvfp4_block16"
    fp4_format: str = "e2m1"

    def __len__(self) -> int:
        """Return sequence length."""
        if self.keys is not None:
            return self.keys.shape[-2]
        return 0

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Append new keys and values to the cache."""
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = torch.cat([self.keys, keys], dim=-2)
            self.values = torch.cat([self.values, values], dim=-2)

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get keys and values for a specific layer."""
        if self.keys is None or self.values is None:
            raise ValueError("Cache is empty")
        return self.keys[layer_idx], self.values[layer_idx]

    def clear(self) -> None:
        """Clear the cache."""
        self.keys = None
        self.values = None
        self.key_scales = None
        self.value_scales = None