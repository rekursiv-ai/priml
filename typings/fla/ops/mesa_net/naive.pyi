from torch import Tensor
from typing import Any

def naive_mesa_net_decoding_one_step(
    q, k, v, g, lamb, beta, prev_h_kk, prev_h_kv, max_CG_iteration=...
) -> tuple[Any, Any, Any]: ...
def naive_mesa_net_exact(
    q, k, v, g, lamb, beta, h_kk_init=..., h_kv_init=...
) -> tuple[Any, Any | Tensor, Any | Tensor]: ...
def naive_mesa_net_CG(
    q,
    k,
    v,
    g,
    lamb,
    beta,
    chunk_size,
    max_CG_iteration=...,
    h_kk_init=...,
    h_kv_init=...,
) -> tuple[Any, Any | Tensor, Any | Tensor]: ...
