from typing import Any

import dataclasses

from _typeshed import Incomplete
from jax import lax as lax

import jax
import jax.numpy as jnp

DEFAULT_MASK_VALUE: Incomplete

@dataclasses.dataclass(frozen=True, slots=True)
class BlockSizes:
    block_q: int
    block_k: int
    block_q_dkv: int | None = ...
    block_kv_dkv: int | None = ...
    block_q_dq: int | None = ...
    block_kv_dq: int | None = ...
    @classmethod
    def get_default(cls): ...
    @property
    def has_backward_blocks(self) -> bool: ...

def mha_forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    segment_ids_ref: jax.Array | None,
    o_ref: Any,
    *residual_refs: Any,
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_k: int,
    head_dim: int,
): ...
def segment_mask(q_segment_ids: jax.Array, kv_segment_ids: jax.Array): ...
def mha(
    q,
    k,
    v,
    segment_ids: jnp.ndarray | None,
    sm_scale: float = 1.0,
    causal: bool = False,
    block_sizes: BlockSizes = ...,
    backward_pass_impl: str = "triton",  # noqa: S107 -- default value mirrors upstream; stub is never executed
    num_warps: int | None = None,
    num_stages: int = 2,
    grid: tuple[int, ...] | None = None,
    interpret: bool = False,
    debug: bool = False,
    return_residuals: bool = False,
): ...
def mha_backward_kernel(
    q_ref,
    k_ref,
    v_ref,
    segment_ids_ref: jax.Array | None,
    out_ref,
    do_scaled_ref,
    lse_ref,
    delta_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    *,
    sm_scale: float,
    causal: bool,
    block_q_dkv: int,
    block_kv_dkv: int,
    block_q_dq: int,
    block_kv_dq: int,
    head_dim: int,
): ...
def mha_reference(
    q,
    k,
    v,
    segment_ids: jnp.ndarray | None,
    sm_scale: float = 1.0,
    causal: bool = False,
): ...
