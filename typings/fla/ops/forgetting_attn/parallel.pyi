import torch

def parallel_forgetting_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = ...,
    window_size: int | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    **kwargs,
) -> torch.Tensor: ...
