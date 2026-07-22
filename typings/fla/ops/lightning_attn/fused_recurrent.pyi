import torch

def fused_recurrent_lightning_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer_idx: int,
    num_layers: int,
    scale: float | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    reverse: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]: ...
