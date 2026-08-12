from torch import Tensor

def chunk_bwd_dqkwg_tilelang(
    q,
    k,
    v,
    do,
    h,
    dh,
    w=...,
    g=...,
    g_gamma=...,
    dv=...,
    scale=...,
    cu_seqlens=...,
    chunk_size=...,
    chunk_indices=...,
    use_exp2=...,
    transpose_state_layout=...,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]: ...
