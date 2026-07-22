from fla.utils import input_guard

import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["T"])
def chunk_abc_fwd_kernel_h(
    k,
    v,
    z,
    h,
    h0,
    ht,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    NORMK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
): ...
@triton.jit(do_not_specialize=["T"])
def chunk_abc_fwd_kernel_intra_K(
    v,
    z,
    o,
    A,
    T,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
): ...
@triton.jit(do_not_specialize=["T"])
def chunk_abc_fwd_kernel_K(
    q,
    k,
    z,
    h,
    o,
    A,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
): ...
@triton.jit(do_not_specialize=["T"])
def chunk_abc_fwd_kernel_intra_V(
    q,
    k,
    z,
    A,
    scale,
    T,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
) -> None: ...
@triton.jit(do_not_specialize=["T"])
def chunk_abc_fwd_kernel_V(
    q,
    v,
    z,
    h,
    o,
    A,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
): ...
@triton.jit(do_not_specialize=["T"])
def chunk_abc_bwd_kernel_dh(
    q,
    z,
    do,
    dh,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    NORMK: tl.constexpr,
): ...
@triton.jit(do_not_specialize=["T"])
def chunk_abc_bwd_kernel_V(
    k,
    v,
    z,
    h,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dA,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
): ...
@triton.jit(do_not_specialize=["T"])
def chunk_abc_bwd_kernel_intra_V(
    q,
    k,
    z,
    dA,
    dq,
    dk,
    T,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
): ...
@triton.jit(do_not_specialize=["T"])
def chunk_abc_bwd_kernel_intra_K(
    v,
    z,
    do,
    dA,
    scale,
    T,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
) -> None: ...
@triton.jit(do_not_specialize=["T"])
def chunk_abc_bwd_kernel_K(
    q,
    k,
    v,
    z,
    h,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dA,
    scale,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
): ...
@triton.jit(do_not_specialize=["T"])
def chunk_abc_bwd_kernel_intra_KV(
    v,
    z,
    A,
    do,
    dv,
    T,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
    NC: tl.constexpr,
): ...
@triton.jit(do_not_specialize=["T"])
def chunk_abc_bwd_kernel_rcum_inter(
    s,
    z,
    ss,
    doo,
    T,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    NT: tl.constexpr,
): ...
@triton.jit(do_not_specialize=["T"])
def chunk_abc_bwd_kernel_rcum_intra(
    s,
    z,
    ss,
    doo,
    T,
    S: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    NC: tl.constexpr,
): ...

class ChunkABCFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx, q, k, v, s, initial_state, output_final_state
    ) -> tuple[Tensor, tuple[Any, Any] | None]: ...
    @staticmethod
    @input_guard
    def backward(
        ctx, dov, dht=...
    ) -> tuple[Tensor, Tensor, Any, Tensor, None, None]: ...

@torch.compiler.disable
def chunk_abc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: torch.Tensor,
    initial_state: tuple[torch.Tensor] | None = ...,
    output_final_state: bool = ...,
    head_first: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
