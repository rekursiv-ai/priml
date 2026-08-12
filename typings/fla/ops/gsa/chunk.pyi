from torch import Tensor
from fla.utils import autotune_cache_kwargs, input_guard

import torch
import triton
import triton.language as tl

@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gsa_fwd_k_kernel_inter(
    q,
    k,
    h,
    g,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_gsa_fwd_k_kernel_intra(
    v,
    g,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
) -> None: ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [2, 4, 8]],
    key=["BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gsa_bwd_k_kernel_dA(
    v,
    g,
    do,
    dA,
    chunk_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
) -> None: ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=["BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gsa_bwd_k_kernel_dqkvg(
    q,
    k,
    v,
    h,
    g,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dg,
    dgv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def chunk_gsa_bwd_k_kernel_intra_dvg(
    v,
    g,
    o,
    A,
    do,
    dv,
    dg,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
    NG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
) -> None: ...
def chunk_gsa_fwd_v(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
def chunk_gsa_fwd_k(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    scale: float = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
def chunk_gsa_bwd_v(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    dg: torch.Tensor,
    scale: float = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ...
def chunk_gsa_bwd_k(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    h0: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    dg: torch.Tensor,
    scale: float = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ...
def chunk_gsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = ...,
    output_final_state: bool = ...,
    scale: float = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
def chunk_gsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor,
    ok: torch.Tensor,
    p: torch.Tensor,
    A: tuple[torch.Tensor, torch.Tensor],
    h: tuple[torch.Tensor, torch.Tensor],
    initial_state: tuple[torch.Tensor, torch.Tensor] | None,
    scale: float,
    do: torch.Tensor,
    dht: tuple[torch.Tensor, torch.Tensor],
    cu_seqlens: torch.LongTensor | None = ...,
    chunk_size: int = ...,
    chunk_indices: torch.LongTensor | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: ...

class ChunkGSAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        s: torch.Tensor,
        g: torch.Tensor,
        scale: float,
        hk0: torch.Tensor | None,
        hv0: torch.Tensor | None,
        output_final_state: bool,
        checkpoint_level: int,
        cu_seqlens: torch.LongTensor | None,
        cu_seqlens_cpu: torch.LongTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @staticmethod
    @input_guard
    def backward(
        ctx, dov, dhkt=..., dhvt=...
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        None,
        Tensor,
        Tensor,
        None,
        None,
        None,
        None,
    ]: ...

@torch.compiler.disable
def chunk_gsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    g: torch.Tensor | None = ...,
    scale: float | None = ...,
    initial_state: tuple[torch.Tensor] | None = ...,
    output_final_state: bool | None = ...,
    checkpoint_level: int | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]: ...
