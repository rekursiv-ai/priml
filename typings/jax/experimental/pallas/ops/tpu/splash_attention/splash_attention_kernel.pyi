from collections.abc import Callable
from typing import Any, NamedTuple

import dataclasses
import enum
import functools

from _typeshed import Incomplete
from jax import (
    ad_checkpoint as ad_checkpoint,
    lax as lax,
    tree_util as tree_util,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
    splash_attention_mask_info as mask_info_lib,
)

import jax
import numpy as np

partial = functools.partial
DEFAULT_MASK_VALUE: Incomplete
NUM_LANES: int
NUM_SUBLANES: int
NN_DIM_NUMBERS: Incomplete
NT_DIM_NUMBERS: Incomplete

class SegmentIds(NamedTuple):
    q: jax.Array
    kv: jax.Array

SplashCustomReturnType: Incomplete
SplashResidualsType: Incomplete
MaskFunctionType: Incomplete

def get_kernel_name(
    is_mqa: bool,
    save_residuals: bool,
    is_segmented: bool,
    phase: str,
) -> str: ...
def attention_reference(
    mask: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None = None,
    *,
    mask_value: float = ...,
    save_residuals: bool = False,
    custom_type: str = "flash",
    attn_logits_soft_cap: float | None = None,
) -> SplashCustomReturnType: ...
def attention_reference_custom(
    mask: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None = None,
    *,
    mask_value: float = ...,
    save_residuals: bool = False,
    custom_type: str = "flash",
    attn_logits_soft_cap: float | None = None,
): ...
def make_attention_reference(
    mask: mask_lib.Mask | np.ndarray,
    is_mqa: bool,
    backward_impl: str = "vanilla",
    **params: Any,
) -> Callable: ...

make_masked_mha_reference: Incomplete
make_masked_mqa_reference: Incomplete

class QKVLayout(enum.IntEnum):
    HEAD_DIM_MINOR = ...
    SEQ_MINOR = ...

def from_head_minor(vals: tuple[Any, ...], layout: QKVLayout): ...

@dataclasses.dataclass(frozen=True, slots=True)
class BlockSizes:
    block_q: int
    block_kv: int
    block_kv_compute: int | None = ...
    block_q_dkv: int | None = ...
    block_kv_dkv: int | None = ...
    block_kv_dkv_compute: int | None = ...
    block_q_dq: int | None = ...
    block_kv_dq: int | None = ...
    use_fused_bwd_kernel: bool = ...
    q_layout: QKVLayout = ...
    k_layout: QKVLayout = ...
    v_layout: QKVLayout = ...
    def __post_init__(self) -> None: ...
    @property
    def has_backward_blocks(self) -> bool: ...
    @classmethod
    def get_default(cls): ...

def flash_attention_kernel(
    data_next_ref,
    block_mask_ref,
    mask_next_ref,
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    sinks_ref,
    mask_ref,
    q_sequence_ref,
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    o_ref,
    logsumexp_ref=None,
    *,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    q_layout: QKVLayout,
    k_layout: QKVLayout,
    v_layout: QKVLayout,
    attn_logits_soft_cap: float | None,
    mask_function: MaskFunctionType | None,
): ...

class SplashAttentionKernel:
    kwargs: Incomplete
    fwd_mask_info: Incomplete
    dq_mask_info: Incomplete
    dkv_mask_info: Incomplete
    def __init__(
        self,
        fwd_mask_info: mask_info_lib.MaskInfo,
        dq_mask_info: mask_info_lib.MaskInfo | None,
        dkv_mask_info: mask_info_lib.MaskInfo | None,
        **kwargs,
    ) -> None: ...
    def __call__(self, *args, **kwargs) -> SplashCustomReturnType: ...
    def manual_sharding_spec(self, sharding: jax.sharding.NamedSharding): ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, kwargs, values): ...

make_splash_mha: Incomplete
make_splash_mqa: Incomplete
make_splash_mha_single_device: Incomplete
make_splash_mqa_single_device: Incomplete
