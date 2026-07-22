import torch
import triton
import triton.language as tl

BLOCK_SIZE_C = ...

def parallel_deltaformer_chunk_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
    qk_scale: float,
    beta: torch.Tensor,
) -> tuple[Tensor, Tensor]: ...
def parallel_deltaformer_bwd_u_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    grad_v: torch.Tensor,
    fa_scale: float,
    beta: torch.Tensor,
) -> Tensor: ...
def parallel_deltaformer_bwd_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    u: torch.Tensor,
    lse: torch.Tensor,
    grad_v: torch.Tensor,
    qk_scale: float,
    fa_scale: float,
    beta: torch.Tensor,
) -> tuple[Tensor, Tensor, Tensor]: ...
def parallel_deltaformer_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    lse: torch.Tensor,
    qk_scale: float,
    beta: torch.Tensor,
) -> None: ...
@triton.autotune(configs=_config_deltaformer(), key=["C", "D"])
@triton.jit
def parallel_deltaformer_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    u_ptr,
    w_ptr,
    lse_ptr,
    beta_ptr,
    H,
    T,
    C,
    D: tl.constexpr,
    qk_scale: float,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
): ...
@triton.autotune(configs=_config_deltaformer(), key=["C", "D"])
@triton.jit
def parallel_deltaformer_bwd_kernel_u(
    o_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    lse_ptr,
    beta_ptr,
    H,
    T,
    C,
    D: tl.constexpr,
    fa_scale,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
): ...
@triton.autotune(configs=_config_deltaformer(), key=["T", "D"])
@triton.jit
def parallel_deltaformer_bwd_kernel_row_sum(
    row_dot_ptr,
    q_ptr,
    k_ptr,
    grad_v_ptr,
    u_ptr,
    lse_ptr,
    H,
    T,
    D: tl.constexpr,
    fa_scale,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
): ...
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": BC}, num_stages=ns, num_warps=nw)
        for BC in [64, 32]
        for ns in [4, 3]
        for nw in [4]
    ],
    key=["T", "D"],
)
@triton.jit
def parallel_deltaformer_bwd_kernel_qk(
    grad_q_ptr,
    grad_k_ptr,
    q_ptr,
    k_ptr,
    grad_v_ptr,
    u_ptr,
    lse_ptr,
    beta_ptr,
    row_dot_ptr,
    H,
    T,
    D: tl.constexpr,
    fa_scale: tl.constexpr,
    qk_scale: tl.constexpr,
    BLOCK_C: tl.constexpr,
): ...

class ParallelDeltaformerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qo: torch.Tensor,
        ko: torch.Tensor,
        vo: torch.Tensor,
        betao: torch.Tensor | None = ...,
        C: int = ...,
        cu_seqlens: torch.LongTensor | None = ...,
    ) -> Tensor: ...
    @staticmethod
    def backward(
        ctx, grad_u: torch.Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, None, None]: ...

def deltaformer_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor | None = ...,
    attention_mask: torch.LongTensor | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    C: int = ...,
) -> torch.Tensor: ...

__all__ = ["deltaformer_attn"]
