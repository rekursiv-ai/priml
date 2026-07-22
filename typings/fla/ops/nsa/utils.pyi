import triton
import triton.language as tl

@triton.jit
def argsort(x, ids, dim: tl.constexpr = ..., descending: tl.constexpr = ...): ...
