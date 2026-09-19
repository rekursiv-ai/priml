from typing import TypedDict

import enum

from _typeshed import Incomplete
from jax._src import (
    core as core,
    custom_derivatives as custom_derivatives,
    dispatch as dispatch,
    dtypes as dtypes,
    xla_bridge as xla_bridge,
)
from jax._src.custom_partitioning import custom_partitioning as custom_partitioning
from jax._src.custom_partitioning_sharding_rule import (
    BATCHING as BATCHING,
    ArrayMapping as ArrayMapping,
    CompoundFactor as CompoundFactor,
    SdyShardingRule as SdyShardingRule,
)
from jax._src.interpreters import (
    batching as batching,
    mlir as mlir,
)
from jax._src.lib import cuda_versions as cuda_versions
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import hlo as hlo
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    PartitionSpec as PartitionSpec,
)
from jax._src.typing import Array as Array

class FP8Params(TypedDict):
    amax_dQ: float
    amax_dK: float
    amax_dV: float
    amax_dP: float
    descale_q: float
    descale_k: float
    descale_v: float
    descale_s: float
    scale_s: float
    scale_o: float
    descale_o: float
    descale_dO: float
    descale_dP: float
    scale_dQ: float
    scale_dK: float
    scale_dV: float
    scale_dP: float

class AttentionLayout(enum.Enum):
    BTNH = 0
    BNTH = 1

class MaskType(enum.Enum):
    NO_MASK = 0
    PADDING = 1
    CAUSAL = 2
    PADDING_CAUSAL = 3
    ALIBI = 4

def convert_mask_type_to_string(mask_type: MaskType) -> str: ...
def has_padding(mask_type: MaskType) -> bool: ...
def should_export_dbias(bias_shape, query_shape, layout) -> bool: ...
def get_large_negative_number(dtype): ...
def element_type_to_backend_config_type_mapping(dtype): ...
def default_layouts(*shapes): ...
def get_max_seg_per_batch(q_offsets): ...
def check_is_paged_attention(page_table_k): ...
def create_dot_product_attention_backend_config_base(
    batch,
    num_heads,
    seq_q,
    seq_kv,
    dtype,
    fmha_scale,
    mask_type,
    layout,
    is_bwd,
): ...
def create_dot_product_attention_backend_config(
    batch,
    num_heads,
    seq_q,
    seq_kv,
    dtype,
    fmha_scale,
    seed,
    dropout_rate,
    mask_type,
    layout,
    sliding_window_length,
    max_seg_per_batch,
    is_paged_attention,
    is_bwd,
): ...
def create_dot_product_attention_fp8_backend_config(
    batch,
    num_heads,
    seq_q,
    seq_kv,
    dtype,
    fmha_scale,
    mask_type,
    layout,
    is_bwd,
): ...
def get_custom_call_name(has_bias, has_dropout, is_bwd, is_fp8: bool = False): ...

get_fp8_custom_call_name: Incomplete

def check_layout(
    query,
    key,
    value,
    bias,
    q_seqlen,
    kv_seqlen,
    q_offsets,
    kv_offsets,
    page_table_k,
    page_table_v,
    layout,
) -> None: ...
def check_is_flash_attention(
    query,
    key,
    value,
    layout: int,
    cudnn_version,
    has_bias,
    is_training,
    is_packed: bool = False,
    is_paged_attention: bool = False,
    is_fp8: bool = False,
): ...
def check_cudnn_version(): ...
def check_compute_capability(capability): ...
def is_cuda_compute_capability_equal(capability): ...

fp8_params_keys: Incomplete
fp8_params_keys_fwd: Incomplete
fp8_params_keys_bwd: Incomplete
params_from_keys: Incomplete

def check_fp8_params(params) -> None: ...

check_is_flash_attention_fp8: Incomplete

def combine_bias_and_mask(bias, mask, dtype): ...
def paged_attention(
    query: Array,
    key: Array,
    value: Array,
    q_seqlen: Array,
    kv_seqlen: Array,
    page_table_k: Array,
    page_table_v: Array,
    bias: Array | None = None,
    mask: Array | None = None,
    fp8_params: FP8Params | None = None,
    *,
    scale: float = 1.0,
    mask_type: MaskType = ...,
    seed: int = 42,
    dropout_rate: float = 0.0,
    qkv_layout: str = "BTNH",
    sliding_window_length: int | None = None,
    use_fp8: bool = False,
    return_residual: bool = False,
): ...
def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Array | None = None,
    mask: Array | None = None,
    q_seqlen: Array | None = None,
    kv_seqlen: Array | None = None,
    q_offsets: Array | None = None,
    kv_offsets: Array | None = None,
    fp8_params: FP8Params | None = None,
    *,
    scale: float = 1.0,
    mask_type: MaskType = ...,
    seed: int = 42,
    dropout_rate: float = 0.0,
    qkv_layout: str = "BTNH",
    sliding_window_length: int | None = None,
    use_fp8: bool = False,
    return_residual: bool = False,
): ...
