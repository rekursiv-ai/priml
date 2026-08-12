from torch import Tensor
from collections import OrderedDict
from typing import NamedTuple

import weakref

import torch

"""Intra-Card Context Parallel for KDA inference (varlen mode only).

Optimized: all CPU-side index computation uses pure Python loops instead of
torch tensor operations (repeat_interleave, arange, cumsum, etc.) to eliminate
per-op overhead on tiny arrays. GPU tensors are created directly from Python
lists to minimize cudaStreamSynchronize calls.
"""
logger = ...
_intracard_cache: OrderedDict[tuple, _CacheEntry] = ...
_INTRACARD_CACHE_MAXSIZE = ...

class _CacheEntry(NamedTuple):
    cu_seqlens_ref: weakref.ReferenceType[torch.Tensor]
    cu_seqlens_subseq_values: list[int]
    split_info: SplitSeqInfo
    total_subseqs: int
    cu_seqlens_split_values: list[int]
    S_split_total: int
    non_first_indices: list[int]
    first_subseq_indices: list[int]
    last_subseq_indices: list[int]
    num_non_first: int
    merge_seq_offsets: list[int]
    merge_init_offsets: list[int]
    cu_seqlens_subseq_gpu: torch.Tensor
    cu_seqlens_split_flat: torch.Tensor

class SplitSeqInfo(NamedTuple):
    split_seq_ids: list[int]
    start_subseq_idx: list[int]
    num_subseqs: list[int]
    @property
    def num_split_seqs(self) -> int: ...
    def __bool__(self) -> bool: ...

def compute_subseq_len(
    seq_len: int, num_sms: int, num_heads: int, chunk_size: int = ...
) -> int: ...
def prepare_subseq_cu_seqlens(
    cu_seqlens_cpu: torch.Tensor,
    subseq_len: int,
    chunk_size: int = ...,
    max_splits: int = ...,
) -> tuple[list[int], SplitSeqInfo | bool, int]: ...
def intracard_pre_scan(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    cu_seqlens_subseq_split: torch.Tensor,
    S_split: int,
    chunk_size: int = ...,
    use_exp2: bool = ...,
) -> Tensor: ...
def intracard_merge(
    hm: torch.Tensor,
    split_info: SplitSeqInfo,
    num_non_first: int,
    merge_seq_offsets: list[int],
    merge_init_offsets: list[int],
    device: torch.device,
    initial_state: torch.Tensor | None = ...,
    transpose_state_layout: bool = ...,
) -> tuple[torch.Tensor | None, int]: ...
def intracard_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = ...,
    gk: torch.Tensor | None = ...,
    initial_state: torch.Tensor | None = ...,
    output_final_state: bool = ...,
    chunk_size: int = ...,
    save_new_value: bool = ...,
    cu_seqlens: torch.LongTensor | None = ...,
    cu_seqlens_cpu: torch.LongTensor | None = ...,
    chunk_indices: torch.LongTensor | None = ...,
    use_exp2: bool = ...,
    max_splits: int = ...,
    transpose_state_layout: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...
