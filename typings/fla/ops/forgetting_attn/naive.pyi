from torch import Tensor

import torch

def naive_forgetting_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = ...,
    window_size: int | None = ...,
) -> Tensor: ...
