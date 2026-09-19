from typing import NamedTuple

from _typeshed import Incomplete
from jax import numpy as jnp

P: Incomplete
MAX_INT8: float

class QuantizedTensor(NamedTuple):
    weight: jnp.ndarray
    scales: jnp.ndarray

def to_int8(x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray: ...
def from_int8(
    x: jnp.ndarray,
    h: jnp.ndarray,
    dtype: jnp.dtype = ...,
) -> jnp.ndarray: ...
def get_quantization_scales(x: jnp.ndarray) -> jnp.ndarray: ...
def quantize_to_int8(x: jnp.ndarray) -> QuantizedTensor: ...
def unquantize_from_int8(x: QuantizedTensor, dtype: jnp.dtype = ...) -> jnp.ndarray: ...
