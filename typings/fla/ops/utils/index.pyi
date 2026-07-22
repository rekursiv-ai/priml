from fla.utils import autotune_cache_kwargs, tensor_cache

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [4, 8, 16, 32]],
    key=["B"],
    **autotune_cache_kwargs,
)
@triton.jit
def prepare_position_ids_kernel(y, cu_seqlens, B: tl.constexpr): ...
@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor: ...
@tensor_cache
def prepare_lens_from_mask(mask: torch.BoolTensor) -> torch.LongTensor: ...
@tensor_cache
def prepare_cu_seqlens_from_lens(
    lens: torch.LongTensor, dtype: torch.dtype | None = ...
) -> torch.LongTensor: ...
@tensor_cache
def prepare_cu_seqlens_from_mask(
    mask: torch.BoolTensor, dtype: torch.dtype | None = ...
) -> torch.LongTensor: ...
@tensor_cache
def prepare_split_cu_seqlens(
    batch_size: int | None = ...,
    seq_len: int | None = ...,
    split_size: int | None = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    dtype: torch.dtype | None = ...,
    device: torch.device | None = ...,
) -> torch.LongTensor: ...
@tensor_cache
def prepare_position_ids(
    cu_seqlens: torch.LongTensor, cu_seqlens_cpu: torch.LongTensor | None = ...
) -> torch.LongTensor: ...
@tensor_cache
def prepare_sequence_ids(
    cu_seqlens: torch.LongTensor, cu_seqlens_cpu: torch.LongTensor | None = ...
) -> torch.LongTensor: ...
@tensor_cache
def prepare_token_indices(
    cu_seqlens: torch.LongTensor, cu_seqlens_cpu: torch.LongTensor | None = ...
) -> torch.LongTensor: ...
@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
) -> torch.LongTensor: ...
@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor: ...
@tensor_cache
def get_max_num_splits(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
) -> int: ...
