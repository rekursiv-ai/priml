import torch
import triton
import triton.language as tl

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_mesa_net_h_kk_bwd_intra_kernel(
    k,
    beta,
    h,
    dh,
    g,
    q_star,
    dq,
    dk,
    dg,
    dbeta,
    dk_beta,
    dlamb,
    cu_seqlens,
    chunk_indices,
    B: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def chunk_mesa_net_h_kk_bwd_intra_fn(
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    q_star: torch.Tensor,
    dq: torch.Tensor,
    dk_beta: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
