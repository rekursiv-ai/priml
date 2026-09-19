from typing import Any

from _typeshed import Incomplete
from jax._src import (
    core as core,
    dtypes as dtypes,
)
from jax._src.numpy.array_constructors import asarray as asarray
from jax._src.typing import Array as Array

import numpy as np

class _ScalarMeta(type):
    dtype: np.dtype
    @property
    def __numpy_dtype__(cls) -> np.dtype: ...
    def __hash__(cls) -> int: ...
    def __eq__(cls, other: object) -> bool: ...
    def __ne__(cls, other: object) -> bool: ...
    def __call__(cls, x: Any) -> Array: ...
    def __instancecheck__(cls, instance: Any) -> bool: ...

bool_: Incomplete
uint1: Incomplete
uint2: Incomplete
uint4: Incomplete
uint8: Incomplete
uint16: Incomplete
uint32: Incomplete
uint64: Incomplete
int1: Incomplete
int2: Incomplete
int4: Incomplete
int8: Incomplete
int16: Incomplete
int32: Incomplete
int64: Incomplete
float4_e2m1fn: Incomplete
float8_e3m4: Incomplete
float8_e4m3: Incomplete
float8_e8m0fnu: Incomplete
float8_e4m3fn: Incomplete
float8_e4m3fnuz: Incomplete
float8_e5m2: Incomplete
float8_e5m2fnuz: Incomplete
float8_e4m3b11fnuz: Incomplete
bfloat16: Incomplete
float16: Incomplete
float32: Incomplete
single: Incomplete
float64: Incomplete
double: Incomplete
complex64: Incomplete
csingle: Incomplete
complex128: Incomplete
cdouble: Incomplete
int_ = int64
uint = uint64
float_ = float64
complex_ = complex128
