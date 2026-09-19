from typing import Any

from _typeshed import Incomplete
from jax import lax as lax

import jax

DEFAULT_MASK_VALUE: Incomplete

def paged_attention_kernel(
    q_ref,
    k_pages_ref,
    k_scales_pages_ref,
    v_pages_ref,
    v_scales_pages_ref,
    block_tables_ref,
    lengths_ref,
    o_ref: Any,
    *residual_refs: Any,
    num_heads: int,
    pages_per_compute_block: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
): ...
def paged_attention_unbatched(
    q: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    block_tables: jax.Array,
    lengths: jax.Array | None,
    k_scales_pages: jax.Array | None = None,
    v_scales_pages: jax.Array | None = None,
    *,
    block_h: int,
    pages_per_compute_block: int,
    k_splits: int,
    num_warps: int,
    num_stages: int,
    interpret: bool,
    debug: bool,
    mask_value: float,
    attn_logits_soft_cap: float | None,
) -> jax.Array: ...
def paged_attention(
    q: jax.Array,
    k_pages: jax.Array,
    v_pages: jax.Array,
    block_tables: jax.Array,
    lengths: jax.Array | None,
    k_scales_pages: jax.Array | None = None,
    v_scales_pages: jax.Array | None = None,
    *,
    block_h: int = 16,
    pages_per_compute_block: int = 8,
    k_splits: int = 16,
    num_warps: int = 8,
    num_stages: int = 2,
    interpret: bool = False,
    debug: bool = False,
    mask_value: float = ...,
    attn_logits_soft_cap: float | None = None,
) -> jax.Array: ...
def paged_attention_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    *,
    mask_value: float = ...,
    attn_logits_soft_cap: float | None = None,
) -> jax.Array: ...
