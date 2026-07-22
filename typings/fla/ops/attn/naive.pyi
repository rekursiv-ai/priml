import torch

def naive_parallel_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = ...,
    window_size: int | None = ...,
    causal: bool = ...,
    *,
    g: torch.Tensor | None = ...,
    sink_bias: torch.Tensor | None = ...,
) -> tuple[Tensor, Tensor | Any]: ...
def naive_attn_decoding(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = ...,
    scale: float | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    do_gate_scale: bool = ...,
    *,
    sink_bias: torch.Tensor | None = ...,
) -> Tensor: ...
