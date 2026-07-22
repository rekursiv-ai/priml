import torch

def naive_path_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    chunk_size: int = ...,
) -> Tensor: ...
