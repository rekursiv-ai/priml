from typing import NamedTuple

import dataclasses

from _typeshed import Incomplete
from jax import lax as lax

import jax

DEFAULT_MASK_VALUE: Incomplete
NUM_LANES: int
NUM_SUBLANES: int

class SegmentIds(NamedTuple):
    q: jax.Array
    kv: jax.Array

@dataclasses.dataclass(frozen=True)
class BlockSizes:
    block_q: int
    block_k_major: int
    block_k: int
    block_b: int
    block_q_major_dkv: int | None = ...
    block_k_major_dkv: int | None = ...
    block_k_dkv: int | None = ...
    block_q_dkv: int | None = ...
    block_k_major_dq: int | None = ...
    block_k_dq: int | None = ...
    block_q_dq: int | None = ...
    def __post_init__(self) -> None: ...
    @property
    def has_backward_blocks(self) -> bool: ...
    @classmethod
    def get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model): ...

def flash_attention(
    q,
    k,
    v,
    ab=None,
    segment_ids=None,
    *,
    causal: bool = False,
    sm_scale: float = 1.0,
    block_sizes: BlockSizes | None = None,
    debug: bool = False,
): ...

MIN_BLOCK_SIZE: int
TRANS_B_DIM_NUMBERS: Incomplete

def below_or_on_diag(r, r_blk_size, c, c_blk_size): ...
def mha_reference_no_custom_vjp(
    q,
    k,
    v,
    ab: jax.Array | None = None,
    segment_ids: SegmentIds | None = None,
    *,
    causal: bool = False,
    mask_value: float = ...,
    sm_scale: float = 1.0,
    save_residuals: bool = False,
): ...
def mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None = None,
    causal: bool = False,
    mask_value: float = ...,
    sm_scale: float = 1.0,
): ...
def mha_reference_bwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    o,
    l,
    m,
    do,
    causal: bool = False,
    mask_value: float = ...,
    sm_scale: float = 1.0,
): ...
