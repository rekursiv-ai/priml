from typing import Any, Literal, overload

from _typeshed import Incomplete
from jax._src import (
    api as api,
    config as config,
    core as core,
    custom_derivatives as custom_derivatives,
    deprecations as deprecations,
    dtypes as dtypes,
    lax as lax,
    util as util,
)
from jax._src.core import AxisName as AxisName
from jax._src.cudnn.fused_attention_stablehlo import MaskType as MaskType
from jax._src.cudnn.scaled_matmul_stablehlo import BlockScaleConfig as BlockScaleConfig
from jax._src.numpy.reductions import Axis as Axis
from jax._src.sharding_impls import NamedSharding as NamedSharding
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DType as DType,
    DTypeLike as DTypeLike,
)

@api.jit
def identity(x: ArrayLike) -> Array: ...
@custom_derivatives.custom_jvp
@api.jit
def relu(x: ArrayLike) -> Array: ...
@api.jit
def squareplus(x: ArrayLike, b: ArrayLike = 4) -> Array: ...
@api.jit
def softplus(x: ArrayLike) -> Array: ...
@api.jit
def sparse_plus(x: ArrayLike) -> Array: ...
@api.jit
def soft_sign(x: ArrayLike) -> Array: ...
def sigmoid(x: ArrayLike) -> Array: ...
@api.jit
def sparse_sigmoid(x: ArrayLike) -> Array: ...
@api.jit
def silu(x: ArrayLike) -> Array: ...

swish = silu

@api.jit
def mish(x: ArrayLike) -> Array: ...
@api.jit
def log_sigmoid(x: ArrayLike) -> Array: ...
@api.jit
def elu(x: ArrayLike, alpha: ArrayLike = 1.0) -> Array: ...
@api.jit
def leaky_relu(x: ArrayLike, negative_slope: ArrayLike = 0.01) -> Array: ...
@api.jit
def hard_tanh(x: ArrayLike) -> Array: ...
@api.jit
def celu(x: ArrayLike, alpha: ArrayLike = 1.0) -> Array: ...
@api.jit
def selu(x: ArrayLike) -> Array: ...
def gelu(x: ArrayLike, approximate: bool = True) -> Array: ...
def glu(x: ArrayLike, axis: int = -1) -> Array: ...

logsumexp: Incomplete

def logmeanexp(
    x: ArrayLike,
    axis: Axis = None,
    where: ArrayLike | None = None,
    keepdims: bool = False,
) -> Array: ...
def log_softmax(
    x: ArrayLike,
    axis: Axis = -1,
    where: ArrayLike | None = None,
) -> Array: ...
def softmax(x: ArrayLike, axis: Axis = -1, where: ArrayLike | None = None) -> Array: ...
def standardize(
    x: ArrayLike,
    axis: Axis = -1,
    mean: ArrayLike | None = None,
    variance: ArrayLike | None = None,
    epsilon: ArrayLike = 1e-05,
    where: ArrayLike | None = None,
) -> Array: ...
def one_hot(
    x: Any,
    num_classes: int,
    *,
    dtype: Any | None = None,
    axis: int | AxisName = -1,
) -> Array: ...
@custom_derivatives.custom_jvp
@api.jit
def relu6(x: ArrayLike) -> Array: ...
@api.jit
def hard_sigmoid(x: ArrayLike) -> Array: ...
@api.jit
def hard_silu(x: ArrayLike) -> Array: ...

hard_swish = hard_silu

@overload
def dot_product_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    bias: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    *,
    scale: float | None = None,
    is_causal: bool = False,
    query_seq_lengths: ArrayLike | None = None,
    key_value_seq_lengths: ArrayLike | None = None,
    local_window_size: int | tuple[int, int] | None = None,
    implementation: Literal["xla", "cudnn"] | None = None,
    return_residual: Literal[False] = ...,
) -> Array: ...
@overload
def dot_product_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    bias: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    *,
    scale: float | None = None,
    is_causal: bool = False,
    query_seq_lengths: ArrayLike | None = None,
    key_value_seq_lengths: ArrayLike | None = None,
    local_window_size: int | tuple[int, int] | None = None,
    implementation: Literal["xla", "cudnn"] | None = None,
    return_residual: Literal[True] = ...,
) -> tuple[Array, Array]: ...
def scaled_matmul(
    lhs: Array,
    rhs: Array,
    lhs_scales: Array,
    rhs_scales: Array,
    preferred_element_type: DTypeLike = ...,
) -> Array: ...
def get_scaled_dot_general_config(
    mode: Literal["nvfp4", "mxfp8"],
    global_scale: Array | None = None,
): ...
def scaled_dot_general(
    lhs,
    rhs,
    dimension_numbers,
    preferred_element_type=...,
    configs: list[BlockScaleConfig] | None = None,
    implementation: Literal["cudnn"] | None = None,
): ...
@custom_derivatives.custom_jvp
@api.jit
def log1mexp(x: ArrayLike) -> Array: ...
