import torch

def naive_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int = ...,
    scale: float | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    **kwargs,
) -> torch.Tensor: ...
