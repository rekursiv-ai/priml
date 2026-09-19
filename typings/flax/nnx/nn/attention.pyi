from collections.abc import (
    Callable as Callable,
    Mapping,
)
from typing import Any

from _typeshed import Incomplete
from flax import nnx as nnx
from flax.nnx import rnglib as rnglib
from flax.nnx.module import (
    Module as Module,
    first_from as first_from,
)
from flax.nnx.nn import (
    dtypes as dtypes,
    initializers as initializers,
)
from flax.nnx.nn.linear import (
    LinearGeneral as LinearGeneral,
    default_kernel_init as default_kernel_init,
)
from flax.nnx.nn.normalization import LayerNorm as LayerNorm
from flax.typing import (
    DotGeneralT as DotGeneralT,
    Dtype as Dtype,
    Initializer as Initializer,
    PrecisionLike as PrecisionLike,
    PromoteDtypeFn as PromoteDtypeFn,
    Shape as Shape,
)

Array: Incomplete

def dot_product_attention_weights(
    query: Array,
    key: Array,
    bias: Array | None = None,
    mask: Array | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: Array | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    module: Module | None = None,
    promote_dtype: PromoteDtypeFn = ...,
    is_causal: bool = False,
): ...
def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Array | None = None,
    mask: Array | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: Array | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    module: Module | None = None,
    promote_dtype: PromoteDtypeFn = ...,
    is_causal: bool = False,
): ...

class MultiHeadAttention(Module):
    num_heads: Incomplete
    in_features: Incomplete
    qkv_features: Incomplete
    out_features: Incomplete
    in_kv_features: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    broadcast_dropout: Incomplete
    dropout_rate: Incomplete
    deterministic: Incomplete
    precision: Incomplete
    use_bias: Incomplete
    attention_fn: Incomplete
    decode: Incomplete
    normalize_qk: Incomplete
    qkv_promote_dtype: Incomplete
    out_promote_dtype: Incomplete
    ln_promote_dtype: Incomplete
    qkv_dot_general: Incomplete
    out_dot_general: Incomplete
    qkv_dot_general_cls: Incomplete
    out_dot_general_cls: Incomplete
    head_dim: Incomplete
    query: Incomplete
    key: Incomplete
    value: Incomplete
    query_ln: LayerNorm | None
    key_ln: LayerNorm | None
    out: Incomplete
    rngs: Incomplete
    cached_key: nnx.Cache[Array] | None
    cached_value: nnx.Cache[Array] | None
    cache_index: nnx.Cache[Array] | None
    def __init__(
        self,
        num_heads: int,
        in_features: int,
        qkv_features: int | None = None,
        out_features: int | None = None,
        in_kv_features: int | None = None,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        broadcast_dropout: bool = True,
        dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        precision: PrecisionLike = None,
        kernel_init: Initializer = ...,
        out_kernel_init: Initializer | None = None,
        bias_init: Initializer = ...,
        out_bias_init: Initializer | None = None,
        use_bias: bool = True,
        attention_fn: Callable[..., Array] = ...,
        decode: bool | None = None,
        normalize_qk: bool = False,
        qkv_promote_dtype: PromoteDtypeFn = ...,
        out_promote_dtype: PromoteDtypeFn = ...,
        ln_promote_dtype: PromoteDtypeFn = ...,
        qkv_dot_general: DotGeneralT | None = None,
        out_dot_general: DotGeneralT | None = None,
        qkv_dot_general_cls: Any = None,
        out_dot_general_cls: Any = None,
        rngs: rnglib.Rngs,
        keep_rngs: bool = True,
        kernel_metadata: Mapping[str, Any] = ...,
        out_kernel_metadata: Mapping[str, Any] = ...,
        bias_metadata: Mapping[str, Any] = ...,
        out_bias_metadata: Mapping[str, Any] = ...,
        query_ln_scale_metadata: Mapping[str, Any] = ...,
        key_ln_scale_metadata: Mapping[str, Any] = ...,
    ) -> None: ...
    def __call__(
        self,
        inputs_q: Array,
        inputs_k: Array | None = None,
        inputs_v: Array | None = None,
        *,
        mask: Array | None = None,
        deterministic: bool | None = None,
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
        sow_weights: bool = False,
        decode: bool | None = None,
    ): ...
    def init_cache(self, input_shape: Shape, dtype: Dtype = ...): ...
    def set_view(
        self,
        deterministic: bool | None = None,
        decode: bool | None = None,
        batch_size: int | Shape | None = None,
        max_length: int | None = None,
    ): ...

def make_attention_mask(
    query_input: Array,
    key_input: Array,
    pairwise_fn: Callable[..., Any] = ...,
    extra_batch_dims: int = 0,
    dtype: Dtype = ...,
): ...
def make_causal_mask(
    x: Array,
    extra_batch_dims: int = 0,
    dtype: Dtype = ...,
) -> Array: ...
def combine_masks(*masks: Array | None, dtype: Dtype = ...) -> Array | None: ...
