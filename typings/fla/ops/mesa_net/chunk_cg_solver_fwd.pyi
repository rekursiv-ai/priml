import torch
import triton
import triton.language as tl

@triton.jit()
def chunk_update_once(b_p, b_k, b_v, b_m, b_g_exp_q, b_h, b_lamb): ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_mesa_cg_dim64_kernel(
    q,
    q_final,
    k,
    h,
    o,
    v,
    h_kv,
    g,
    beta,
    lamb,
    cu_seqlens,
    chunk_indices,
    T,
    max_CG_iteration: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def chunk_mesa_cg_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    h_kv: torch.Tensor,
    g_local_cumsum: torch.Tensor,
    beta: torch.Tensor,
    lamb: torch.Tensor,
    cu_seqlens: torch.Tensor | None = ...,
    chunk_size: int = ...,
    max_CG_iteration: int = ...,
    output_dtype: torch.dtype | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> torch.Tensor: ...
