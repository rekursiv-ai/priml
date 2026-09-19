from collections.abc import Callable as Callable
from typing import Any, overload

from flax.linen import initializers as initializers
from flax.linen.dtypes import promote_dtype as promote_dtype
from flax.linen.linear import (
    DenseGeneral as DenseGeneral,
    default_kernel_init as default_kernel_init,
)
from flax.linen.module import (
    Module as Module,
    compact as compact,
    merge_param as merge_param,
)
from flax.linen.normalization import LayerNorm as LayerNorm
from flax.typing import (
    Array as Array,
    DotGeneralT as DotGeneralT,
    Dtype as Dtype,
    Initializer as Initializer,
    PrecisionLike as PrecisionLike,
    PRNGKey as PRNGKey,
)

def dot_product_attention_weights(
    query: Array,
    key: Array,
    bias: Array | None = None,
    mask: Array | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: PRNGKey | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    module: Module | None = None,
    force_fp32_for_softmax: bool = False,
    einsum_dot_general: Callable[..., Array] | None = None,
    einsum: Callable[..., Array] | None = None,
): ...
def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Array | None = None,
    mask: Array | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: PRNGKey | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    module: Module | None = None,
    force_fp32_for_softmax: bool = False,
    einsum_dot_general: Callable[..., Array] | None = None,
    qk_attn_weights_einsum: Callable[..., Array] | None = None,
    attn_weights_value_einsum: Callable[..., Array] | None = None,
): ...

class MultiHeadDotProductAttention(Module):
    num_heads: int
    dtype: Dtype | None = ...
    param_dtype: Dtype = ...
    qkv_features: int | None = ...
    out_features: int | None = ...
    broadcast_dropout: bool = ...
    dropout_rate: float = ...
    deterministic: bool | None = ...
    precision: PrecisionLike = ...
    kernel_init: Initializer = ...
    out_kernel_init: Initializer | None = ...
    bias_init: Initializer = ...
    out_bias_init: Initializer | None = ...
    use_bias: bool = ...
    attention_fn: Callable[..., Array] = ...
    decode: bool = ...
    normalize_qk: bool = ...
    force_fp32_for_softmax: bool = ...
    qkv_dot_general: DotGeneralT | None = ...
    out_dot_general: DotGeneralT | None = ...
    qkv_dot_general_cls: Any = ...
    out_dot_general_cls: Any = ...
    qk_attn_weights_einsum_cls: Callable[..., Callable[..., Array]] | None = ...
    attn_weights_value_einsum_cls: Callable[..., Callable[..., Array]] | None = ...
    @overload
    def __call__(
        self,
        inputs_q: Array,
        inputs_k: Array | None = None,
        inputs_v: Array | None = None,
        *,
        mask: Array | None = None,
        deterministic: bool | None = None,
        dropout_rng: PRNGKey | None = None,
        sow_weights: bool = False,
    ): ...
    @overload
    def __call__(
        self,
        inputs_q: Array,
        *,
        inputs_kv: Array | None = None,
        mask: Array | None = None,
        deterministic: bool | None = None,
        dropout_rng: PRNGKey | None = None,
        sow_weights: bool = False,
    ): ...

class MultiHeadAttention(MultiHeadDotProductAttention): ...

class SelfAttention(MultiHeadDotProductAttention):
    @compact
    def __call__(
        self,
        inputs_q: Array,
        mask: Array | None = None,
        deterministic: bool | None = None,
        dropout_rng: PRNGKey | None = None,
        sow_weights: bool = False,
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
