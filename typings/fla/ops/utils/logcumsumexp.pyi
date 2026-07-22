from fla.utils import autotune_cache_kwargs

import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=["S"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def logcumsumexp_fwd_kernel(s, z, T, S: tl.constexpr, BT: tl.constexpr): ...
