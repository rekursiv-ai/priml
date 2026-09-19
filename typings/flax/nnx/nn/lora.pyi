import typing as tp

from _typeshed import Incomplete
from flax.nnx import (
    rnglib as rnglib,
    variablelib as variablelib,
)
from flax.nnx.module import Module as Module
from flax.nnx.nn import (
    dtypes as dtypes,
    initializers as initializers,
)
from flax.nnx.nn.linear import Linear as Linear
from flax.typing import (
    Dtype as Dtype,
    Initializer as Initializer,
    PromoteDtypeFn as PromoteDtypeFn,
)

import jax

Array: Incomplete
Axis = int
Size = int
A = tp.TypeVar("A")
default_a_initializer: Incomplete
default_b_initializer: Incomplete

class LoRAParam(variablelib.Param[A]): ...

class LoRA(Module):
    in_features: Incomplete
    out_features: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    lora_param_type: Incomplete
    base_module: Incomplete
    promote_dtype: Incomplete
    lora_a: Incomplete
    lora_b: Incomplete
    def __init__(
        self,
        in_features: int,
        lora_rank: int,
        out_features: int,
        *,
        base_module: Module | None = None,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        a_initializer: Initializer = ...,
        b_initializer: Initializer = ...,
        lora_param_type: type[variablelib.Variable] = ...,
        promote_dtype: PromoteDtypeFn = ...,
        rngs: rnglib.Rngs,
        a_metadata: tp.Mapping[str, tp.Any] = ...,
        b_metadata: tp.Mapping[str, tp.Any] = ...,
    ) -> None: ...
    def __call__(self, x: jax.Array): ...

class LoRALinear(Linear):
    lora: Incomplete
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        lora_rank: int,
        lora_dtype: Dtype | None = None,
        lora_param_dtype: Dtype = ...,
        a_initializer: Initializer = ...,
        b_initializer: Initializer = ...,
        lora_param_type: type[variablelib.Variable] = ...,
        lora_promote_dtype: PromoteDtypeFn = ...,
        rngs: rnglib.Rngs,
        a_metadata: tp.Mapping[str, tp.Any] = ...,
        b_metadata: tp.Mapping[str, tp.Any] = ...,
        **kwargs,
    ) -> None: ...
    def __call__(self, x: jax.Array, out_sharding=None): ...
