from fla.utils import autotune_cache_kwargs, input_guard

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BD": 32}, num_warps=1),
        triton.Config({"BD": 32}, num_warps=2),
        triton.Config({"BD": 32}, num_warps=4),
        triton.Config({"BD": 32}, num_warps=8),
        triton.Config({"BD": 64}, num_warps=1),
        triton.Config({"BD": 64}, num_warps=2),
        triton.Config({"BD": 64}, num_warps=4),
        triton.Config({"BD": 64}, num_warps=8),
        triton.Config({"BD": 128}, num_warps=1),
        triton.Config({"BD": 128}, num_warps=2),
        triton.Config({"BD": 128}, num_warps=4),
        triton.Config({"BD": 128}, num_warps=8),
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_hgrn_fwd_kernel_h(
    x,
    g,
    gc,
    o,
    h0,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
): ...
@triton.jit(do_not_specialize=["T"])
def chunk_hgrn_fwd_kernel_o(
    gc, o, s_b, s_t, s_d, T, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr
): ...
@triton.autotune(
    configs=[
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in [32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_hgrn_bwd_kernel_h(
    g, gc, dx, do, T, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr
): ...
@triton.jit(do_not_specialize=["T"])
def chunk_hgrn_bwd_kernel_o(
    g,
    gc,
    o,
    dx,
    dg,
    s_b,
    s_t,
    s_d,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
): ...

class ChunkHGRNFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx, x, g, initial_state=..., output_final_state=...
    ) -> tuple[Tensor, Tensor | None]: ...
    @staticmethod
    @input_guard
    def backward(ctx, do, dht=...) -> tuple[Tensor, Tensor, None, None]: ...

@torch.compiler.disable
def chunk_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = ...,
    output_final_state: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
