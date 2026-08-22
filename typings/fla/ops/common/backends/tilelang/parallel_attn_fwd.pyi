from torch import Tensor

import torch

"""TileLang forward kernel for parallel (causal) attention.

Supports: GQA, gating (g_cumsum), sliding-window attention, sink_bias,
variable-length (cu_seqlens). Output format matches the Triton reference
at `fla.ops.attn.parallel.parallel_attn_fwd` so LSE is directly consumable
by either backend during backward.
"""

def parallel_attn_fwd_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    sink_bias: torch.Tensor | None,
    scale: float,
    window_size: int | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor]: ...
