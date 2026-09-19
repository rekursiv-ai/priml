from collections.abc import (
    Callable as Callable,
    Mapping,
)
from typing import Any, Protocol, TypeVar

from _typeshed import Incomplete
from flax import nnx as nnx
from flax.nnx import (
    filterlib as filterlib,
    rnglib as rnglib,
)
from flax.nnx.module import Module as Module
from flax.nnx.nn import (
    dtypes as dtypes,
    initializers as initializers,
)
from flax.nnx.nn.activations import (
    sigmoid as sigmoid,
    tanh as tanh,
)
from flax.nnx.nn.linear import Linear as Linear
from flax.nnx.transforms import iteration as iteration
from flax.typing import (
    Dtype as Dtype,
    Initializer as Initializer,
    PromoteDtypeFn as PromoteDtypeFn,
    Shape as Shape,
)

default_kernel_init: Incomplete
default_bias_init: Incomplete
A = TypeVar("A")
Array: Incomplete
type Output = Any
type Carry = Any

class RNNCellBase(Module):
    def initialize_carry(
        self,
        input_shape: tuple[int, ...],
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
        carry_init: Initializer | None = None,
    ) -> Carry: ...
    def __call__(self, carry: Carry, inputs: Array) -> tuple[Carry, Array]: ...
    @property
    def num_feature_axes(self) -> int: ...

def modified_orthogonal(key: Array, shape: Shape, dtype: Dtype = ...) -> Array: ...

class LSTMCell(RNNCellBase):
    in_features: Incomplete
    hidden_features: Incomplete
    gate_fn: Incomplete
    activation_fn: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    promote_dtype: Incomplete
    rngs: rnglib.RngStream | None
    ii: Incomplete
    if_: Incomplete
    ig: Incomplete
    io: Incomplete
    hi: Incomplete
    hf: Incomplete
    hg: Incomplete
    ho: Incomplete
    carry_init: Incomplete
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        gate_fn: Callable[..., Any] = ...,
        activation_fn: Callable[..., Any] = ...,
        kernel_init: Initializer = ...,
        recurrent_kernel_init: Initializer = ...,
        bias_init: Initializer = ...,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        carry_init: Initializer | None = None,
        promote_dtype: PromoteDtypeFn = ...,
        keep_rngs: bool = False,
        rngs: rnglib.Rngs,
        kernel_metadata: Mapping[str, Any] = ...,
        recurrent_kernel_metadata: Mapping[str, Any] = ...,
        bias_metadata: Mapping[str, Any] = ...,
    ) -> None: ...
    def __call__(
        self,
        carry: tuple[Array, Array],
        inputs: Array,
    ) -> tuple[tuple[Array, Array], Array]: ...
    def initialize_carry(
        self,
        input_shape: tuple[int, ...],
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
        carry_init: Initializer | None = None,
    ) -> tuple[Array, Array]: ...
    @property
    def num_feature_axes(self) -> int: ...

class OptimizedLSTMCell(RNNCellBase):
    in_features: Incomplete
    hidden_features: Incomplete
    gate_fn: Incomplete
    activation_fn: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    promote_dtype: Incomplete
    rngs: rnglib.RngStream | None
    dense_i: Incomplete
    dense_h: Incomplete
    carry_init: Incomplete
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        gate_fn: Callable[..., Any] = ...,
        activation_fn: Callable[..., Any] = ...,
        kernel_init: Initializer = ...,
        recurrent_kernel_init: Initializer = ...,
        bias_init: Initializer = ...,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        carry_init: Initializer | None = None,
        promote_dtype: PromoteDtypeFn = ...,
        keep_rngs: bool = False,
        rngs: rnglib.Rngs,
        kernel_metadata: Mapping[str, Any] = ...,
        recurrent_kernel_metadata: Mapping[str, Any] = ...,
        bias_metadata: Mapping[str, Any] = ...,
    ) -> None: ...
    def __call__(
        self,
        carry: tuple[Array, Array],
        inputs: Array,
    ) -> tuple[tuple[Array, Array], Array]: ...
    def initialize_carry(
        self,
        input_shape: tuple[int, ...],
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
        carry_init: Initializer | None = None,
    ) -> tuple[Array, Array]: ...
    @property
    def num_feature_axes(self) -> int: ...

class SimpleCell(RNNCellBase):
    in_features: Incomplete
    hidden_features: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    residual: Incomplete
    activation_fn: Incomplete
    promote_dtype: Incomplete
    rngs: rnglib.RngStream | None
    dense_h: Incomplete
    dense_i: Incomplete
    carry_init: Incomplete
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        dtype: Dtype = ...,
        param_dtype: Dtype = ...,
        carry_init: Initializer | None = None,
        residual: bool = False,
        activation_fn: Callable[..., Any] = ...,
        kernel_init: Initializer = ...,
        recurrent_kernel_init: Initializer = ...,
        bias_init: Initializer = ...,
        promote_dtype: PromoteDtypeFn = ...,
        keep_rngs: bool = False,
        rngs: rnglib.Rngs,
        kernel_metadata: Mapping[str, Any] = ...,
        recurrent_kernel_metadata: Mapping[str, Any] = ...,
        bias_metadata: Mapping[str, Any] = ...,
    ) -> None: ...
    def __call__(self, carry: Array, inputs: Array) -> tuple[Array, Array]: ...
    def initialize_carry(
        self,
        input_shape: tuple[int, ...],
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
        carry_init: Initializer | None = None,
    ) -> Array: ...
    @property
    def num_feature_axes(self) -> int: ...

class GRUCell(RNNCellBase):
    in_features: Incomplete
    hidden_features: Incomplete
    gate_fn: Incomplete
    activation_fn: Incomplete
    dtype: Incomplete
    param_dtype: Incomplete
    promote_dtype: Incomplete
    rngs: rnglib.RngStream | None
    dense_i: Incomplete
    dense_h: Incomplete
    carry_init: Incomplete
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        gate_fn: Callable[..., Any] = ...,
        activation_fn: Callable[..., Any] = ...,
        kernel_init: Initializer = ...,
        recurrent_kernel_init: Initializer = ...,
        bias_init: Initializer = ...,
        dtype: Dtype | None = None,
        param_dtype: Dtype = ...,
        carry_init: Initializer | None = None,
        promote_dtype: PromoteDtypeFn = ...,
        keep_rngs: bool = False,
        rngs: rnglib.Rngs,
        kernel_metadata: Mapping[str, Any] = ...,
        recurrent_kernel_metadata: Mapping[str, Any] = ...,
        bias_metadata: Mapping[str, Any] = ...,
    ) -> None: ...
    def __call__(self, carry: Array, inputs: Array) -> tuple[Array, Array]: ...
    def initialize_carry(
        self,
        input_shape: tuple[int, ...],
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
        carry_init: Initializer | None = None,
    ) -> Array: ...
    @property
    def num_feature_axes(self) -> int: ...

class RNN(Module):
    state_axes: dict[str, int | type[iteration.Carry] | None]
    cell: Incomplete
    time_major: Incomplete
    return_carry: Incomplete
    reverse: Incomplete
    keep_order: Incomplete
    unroll: Incomplete
    rngs: rnglib.RngStream | None
    broadcast_rngs: Incomplete
    def __init__(
        self,
        cell: RNNCellBase,
        *,
        time_major: bool = False,
        return_carry: bool = False,
        reverse: bool = False,
        keep_order: bool = False,
        unroll: int = 1,
        state_axes: Mapping[str, int | type[iteration.Carry] | None] | None = None,
        broadcast_rngs: filterlib.Filter = None,
        rngs: rnglib.Rngs | rnglib.RngStream | bool = True,
    ) -> None: ...
    def __call__(
        self,
        inputs: Array,
        *,
        initial_carry: Carry | None = None,
        seq_lengths: Array | None = None,
        return_carry: bool | None = None,
        time_major: bool | None = None,
        reverse: bool | None = None,
        keep_order: bool | None = None,
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
    ): ...

def flip_sequences(
    inputs: Array,
    seq_lengths: Array | None,
    num_batch_dims: int,
    time_major: bool,
) -> Array: ...

class RNNBase(Protocol):
    def __call__(
        self,
        inputs: Array,
        *,
        initial_carry: Carry | None = None,
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
        seq_lengths: Array | None = None,
        return_carry: bool | None = None,
        time_major: bool | None = None,
        reverse: bool | None = None,
        keep_order: bool | None = None,
    ) -> Output | tuple[Carry, Output]: ...

class Bidirectional(Module):
    forward_rnn: RNNBase
    backward_rnn: RNNBase
    merge_fn: Callable[[Array, Array], Array]
    time_major: bool
    return_carry: bool
    rngs: rnglib.RngStream | None
    def __init__(
        self,
        forward_rnn: RNNBase,
        backward_rnn: RNNBase,
        *,
        merge_fn: Callable[[Array, Array], Array] = ...,
        time_major: bool = False,
        return_carry: bool = False,
        rngs: rnglib.Rngs | rnglib.RngStream | bool = True,
    ) -> None: ...
    def __call__(
        self,
        inputs: Array,
        *,
        initial_carry: tuple[Carry, Carry] | None = None,
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
        seq_lengths: Array | None = None,
        return_carry: bool | None = None,
        time_major: bool | None = None,
        reverse: bool | None = None,
        keep_order: bool | None = None,
    ) -> Output | tuple[tuple[Carry, Carry], Output]: ...
