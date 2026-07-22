import torch

def fused_recurrent_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = ...,
    initial_state: torch.Tensor | tuple | None = ...,
    output_final_state: bool = ...,
    reverse: bool = ...,
    normalize: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
