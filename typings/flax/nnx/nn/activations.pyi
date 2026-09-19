import typing as tp

from _typeshed import Incomplete
from flax import nnx
from flax.typing import Array, Dtype, PromoteDtypeFn
from jax.nn import (
    celu as celu,
    elu as elu,
    gelu as gelu,
    glu as glu,
    hard_sigmoid as hard_sigmoid,
    hard_silu as hard_silu,
    hard_swish as hard_swish,
    hard_tanh as hard_tanh,
    identity as identity,
    leaky_relu as leaky_relu,
    log_sigmoid as log_sigmoid,
    log_softmax as log_softmax,
    logsumexp as logsumexp,
    one_hot as one_hot,
    relu as relu,
    relu6 as relu6,
    selu as selu,
    sigmoid as sigmoid,
    silu as silu,
    soft_sign as soft_sign,
    softmax as softmax,
    softplus as softplus,
    standardize as standardize,
    swish as swish,
)
from jax.numpy import tanh as tanh

__all__ = [
    "PReLU",
    "celu",
    "elu",
    "gelu",
    "glu",
    "hard_sigmoid",
    "hard_silu",
    "hard_swish",
    "hard_tanh",
    "identity",
    "leaky_relu",
    "log_sigmoid",
    "log_softmax",
    "logsumexp",
    "one_hot",
    "relu",
    "relu6",
    "selu",
    "sigmoid",
    "silu",
    "soft_sign",
    "softmax",
    "softplus",
    "standardize",
    "swish",
    "tanh",
]

class PReLU(nnx.Module):
    negative_slope: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    promote_dtype: Incomplete
    def __init__(
        self,
        negative_slope_init: float = 0.01,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        promote_dtype: PromoteDtypeFn = ...,
        negative_slope_metadata: tp.Mapping[str, tp.Any] = ...,
    ) -> None: ...
    def __call__(self, inputs: Array) -> Array: ...
