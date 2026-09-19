from flax.linen.linear import Dense as Dense
from flax.linen.module import (
    Module as Module,
    compact as compact,
)
from flax.typing import (
    Array as Array,
    Dtype as Dtype,
)
from jax.nn import (
    celu as celu,
    elu as elu,
    gelu as gelu,
    glu as glu,
    hard_sigmoid as hard_sigmoid,
    hard_silu as hard_silu,
    hard_swish as hard_swish,
    hard_tanh as hard_tanh,
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
    standardize,
    swish as swish,
)
from jax.numpy import tanh as tanh

normalize = standardize

class PReLU(Module):
    param_dtype: Dtype = ...
    negative_slope_init: float = ...
    @compact
    def __call__(self, inputs: Array) -> Array: ...
