import torch

def fused_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = ...,
    gv: torch.Tensor | None = ...,
    scale: int | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    reverse: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
