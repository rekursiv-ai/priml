from torch.utils._triton import has_triton

import torch

AUTOTUNE = ...

def tune_bsr_dense_addmm(
    input,
    bsr,
    dense,
    *,
    beta=...,
    alpha=...,
    left_alpha=...,
    right_alpha=...,
    out=...,
    store=...,
    verbose=...,
    force=...,
    opname=...,
):  # -> dict[Literal['GROUP_SIZE', 'SPLIT_N', 'TILE_M', 'TILE_N', 'num_stages', 'num_warps'], Any] | dict[Literal['GROUP_SIZE_ROW', 'SPLIT_N', 'num_stages', 'num_warps'], Any] | dict[Any, Any]:
    ...
def bsr_dense_addmm_meta(
    M,
    K,
    N,
    Ms,
    Ks,
    beta,
    alpha,
    SPLIT_N=...,
    GROUP_SIZE_ROW=...,
    num_warps=...,
    num_stages=...,
    sparsity=...,
    dtype=...,
    out_dtype=...,
    _version=...,
    **extra,
):  # -> dict[Any, Any] | dict[Literal['GROUP_SIZE', 'SPLIT_N', 'TILE_M', 'TILE_N', 'num_stages', 'num_warps'], Any] | dict[Literal['GROUP_SIZE_ROW', 'SPLIT_N', 'num_stages', 'num_warps'], Any] | dict[str, Any | int]:
    ...
def bsr_dense_addmm(
    input: torch.Tensor,
    bsr: torch.Tensor,
    dense: torch.Tensor,
    *,
    beta=...,
    alpha=...,
    left_alpha: torch.Tensor | None = ...,
    right_alpha: torch.Tensor | None = ...,
    out: torch.Tensor | None = ...,
    skip_checks: bool = ...,
    max_grid: tuple[int | None, int | None, int | None] | None = ...,
    meta: dict | None = ...,
):  # -> Tensor:
    ...

if has_triton(): ...
else:
    _bsr_strided_addmm_kernel = ...
