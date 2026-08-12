from torch import Tensor
from fla.utils import input_guard

import torch
import triton
import triton.language as tl

@triton.jit
def mesa_net_decoding_one_step_kernel(
    q,
    k,
    v,
    g,
    o,
    lamb,
    beta,
    prev_h_kk,
    prev_h_kv,
    curr_h_kk,
    curr_h_kv,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAX_CG_STEP: tl.constexpr,
): ...
@input_guard
def mesa_net_decoding_one_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    lamb: torch.Tensor,
    beta: torch.Tensor,
    prev_h_kk: torch.Tensor,
    prev_h_kv: torch.Tensor,
    max_CG_iteration: int = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...
