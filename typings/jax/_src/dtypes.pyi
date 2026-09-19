from collections.abc import Callable
from typing import Any, Literal, overload

import abc
import functools

from _typeshed import Incomplete
from jax._src import (
    config as config,
    literals as literals,
    traceback_util as traceback_util,
)
from jax._src.typing import (
    Array as Array,
    DType as DType,
    DTypeLike as DTypeLike,
)
from jax._src.util import (
    StrictABC as StrictABC,
    set_module as set_module,
)

import numpy as np

export: Incomplete

class extended(np.generic, metaclass=abc.ABCMeta): ...
class prng_key(extended, metaclass=abc.ABCMeta): ...

class ExtendedDType(StrictABC, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def type(self) -> type: ...

float8_e3m4: type[np.generic]
float8_e4m3: type[np.generic]
float8_e8m0fnu: type[np.generic]
float8_e4m3b11fnuz: type[np.generic]
float8_e4m3fn: type[np.generic]
float8_e4m3fnuz: type[np.generic]
float8_e5m2: type[np.generic]
float8_e5m2fnuz: type[np.generic]
float4_e2m1fn: type[np.generic]

def supports_inf(dtype: DTypeLike) -> bool: ...

bfloat16: type[np.generic]
int1: type[np.generic] | None
uint1: type[np.generic] | None
int2: type[np.generic]
uint2: type[np.generic]
int4: type[np.generic]
uint4: type[np.generic]
bool_: Incomplete
int_: type[Any]
uint: type[Any]
float_: type[Any]
complex_: type[Any]

def default_int_dtype() -> DType: ...
def default_uint_dtype() -> DType: ...
def default_float_dtype() -> DType: ...
def default_complex_dtype() -> DType: ...

default_types: dict[str, Callable[[], DType]]

def jax_dtype(
    obj: DTypeLike | None,
    *,
    align: bool = False,
    copy: bool = False,
) -> DType: ...
def itemsize_bits(dtype: DTypeLike) -> int: ...

float0: np.dtype

def to_numeric_dtype(dtype: DTypeLike) -> DType: ...
def to_inexact_dtype(dtype: DTypeLike) -> DType: ...
def to_floating_dtype(dtype: DTypeLike) -> DType: ...
def to_complex_dtype(dtype: DTypeLike) -> DType: ...
@overload
def canonicalize_dtype(
    dtype: Any,
    allow_extended_dtype: Literal[False] = False,
) -> DType: ...
@overload
def canonicalize_dtype(
    dtype: Any,
    allow_extended_dtype: bool = False,
) -> DType | ExtendedDType: ...

class InvalidInputException(Exception): ...

canonicalize_value_handlers: dict[Any, Callable]

def canonicalize_value(x): ...

python_scalar_types: set[type]
python_scalar_types_to_dtypes: dict[type, DType]

@export
def scalar_type_of(x: Any) -> type: ...
def scalar_type_to_dtype(typ: type, value: Any = None) -> DType: ...
def coerce_to_array(x: Any, dtype: DTypeLike | None = None) -> np.ndarray: ...

iinfo: Incomplete
finfo: Incomplete

def issubdtype(
    a: DTypeLike | ExtendedDType | None,
    b: DTypeLike | ExtendedDType | None,
) -> bool: ...

can_cast: Incomplete
type JAXType = type | DType

def isdtype(
    dtype: DTypeLike,
    kind: str | DTypeLike | tuple[str | DTypeLike, ...],
) -> bool: ...

class TypePromotionError(ValueError): ...

def promote_types(a: DTypeLike, b: DTypeLike) -> DType: ...
def register_weak_scalar_type(typ: type): ...
def is_weakly_typed(x: Any) -> bool: ...
def is_python_scalar(x: Any) -> bool: ...
def check_valid_dtype(dtype: DType) -> None: ...
def dtype(x: Any) -> DType: ...
def lattice_result_type(*args: Any) -> tuple[DType, bool]: ...
@overload
def result_type(
    *args: Any,
    return_weak_type_flag: Literal[True],
) -> tuple[DType, bool]: ...
@overload
def result_type(*args: Any, return_weak_type_flag: Literal[False] = False) -> DType: ...
@overload
def result_type(
    *args: Any,
    return_weak_type_flag: bool = False,
) -> DType | tuple[DType, bool]: ...
def check_and_canonicalize_user_dtype(dtype, fun_name=None) -> DType: ...
def safe_to_cast(input_dtype_or_value: Any, output_dtype_or_value: Any) -> bool: ...
def primal_tangent_dtype(
    primal_dtype,
    tangent_dtype,
    name: str | None = None,
) -> ExtendedDType: ...
@functools.cache
def short_dtype_name(dtype) -> str: ...
def is_string_dtype(dtype: DTypeLike | None) -> bool: ...
