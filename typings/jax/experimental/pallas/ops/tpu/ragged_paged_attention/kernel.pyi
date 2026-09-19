from _typeshed import Incomplete
from jax import lax as lax
from jax._src import dtypes as dtypes
from jax.experimental.pallas.ops.tpu.ragged_paged_attention.tuned_block_sizes import (
    get_tuned_block_sizes as get_tuned_block_sizes,
)

import jax

DEFAULT_MASK_VALUE: Incomplete

class MultiPageAsyncCopyDescriptor:
    def __init__(
        self,
        pages_hbm_ref,
        vmem_buf,
        sem,
        page_indices_ref,
        metadata,
    ) -> None: ...
    def start(self) -> None: ...
    def wait(self): ...

def ref_ragged_paged_attention(
    queries: jax.Array,
    kv_pages: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = ...,
    k_scale: float | None = None,
    v_scale: float | None = None,
): ...
def dynamic_validate_inputs(
    q: jax.Array,
    kv_pages: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    *,
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
): ...
def static_validate_inputs(
    q: jax.Array,
    kv_pages: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    *,
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
): ...
def ragged_paged_attention_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    seq_buf_idx_ref,
    num_seqs_ref,
    q_ref,
    kv_pages_hbm_ref,
    o_ref,
    kv_bufs,
    sems,
    l_ref,
    m_ref,
    acc_ref,
    *,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = ...,
    k_scale: float | None = None,
    v_scale: float | None = None,
): ...
def get_dtype_packing(dtype): ...
def get_min_heads_per_blk(num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype): ...
def ragged_paged_attention(
    q: jax.Array,
    kv_pages: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = ...,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
): ...
