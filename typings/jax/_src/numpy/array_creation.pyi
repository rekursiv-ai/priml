from typing import Any, Literal, overload

from _typeshed import Incomplete
from jax._src import (
    api as api,
    core as core,
    dtypes as dtypes,
)
from jax._src.lax import lax as lax
from jax._src.lib import xla_client as xc
from jax._src.numpy import (
    ufuncs as ufuncs,
    util as util,
)
from jax._src.numpy.array_constructors import asarray as asarray
from jax._src.sharding import Sharding as Sharding
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    PartitionSpec as P,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DTypeLike as DTypeLike,
    DuckTypedArray as DuckTypedArray,
)
from jax._src.util import (
    canonicalize_axis as canonicalize_axis,
    set_module as set_module,
)

export: Incomplete

def canonicalize_shape(shape: Any, context: str = "") -> core.Shape: ...
@export
def zeros(
    shape: Any,
    dtype: DTypeLike | None = None,
    *,
    device: xc.Device | Sharding | None = None,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
@export
def ones(
    shape: Any,
    dtype: DTypeLike | None = None,
    *,
    device: xc.Device | Sharding | None = None,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
@export
def empty(
    shape: Any,
    dtype: DTypeLike | None = None,
    *,
    device: xc.Device | Sharding | None = None,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
@export
def full(
    shape: Any,
    fill_value: ArrayLike,
    dtype: DTypeLike | None = None,
    *,
    device: xc.Device | Sharding | None = None,
) -> Array: ...
@export
def zeros_like(
    a: ArrayLike | DuckTypedArray,
    dtype: DTypeLike | None = None,
    shape: Any = None,
    *,
    device: xc.Device | Sharding | None = None,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
@export
def ones_like(
    a: ArrayLike | DuckTypedArray,
    dtype: DTypeLike | None = None,
    shape: Any = None,
    *,
    device: xc.Device | Sharding | None = None,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
@export
def empty_like(
    prototype: ArrayLike | DuckTypedArray,
    dtype: DTypeLike | None = None,
    shape: Any = None,
    *,
    device: xc.Device | Sharding | None = None,
) -> Array: ...
@export
def full_like(
    a: ArrayLike | DuckTypedArray,
    fill_value: ArrayLike,
    dtype: DTypeLike | None = None,
    shape: Any = None,
    *,
    device: xc.Device | Sharding | None = None,
) -> Array: ...
@overload
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    endpoint: bool = True,
    retstep: Literal[False] = False,
    dtype: DTypeLike | None = None,
    axis: int = 0,
    *,
    device: xc.Device | Sharding | None = None,
) -> Array: ...
@overload
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int,
    endpoint: bool,
    retstep: Literal[True],
    dtype: DTypeLike | None = None,
    axis: int = 0,
    *,
    device: xc.Device | Sharding | None = None,
) -> tuple[Array, Array]: ...
@overload
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    endpoint: bool = True,
    *,
    retstep: Literal[True],
    dtype: DTypeLike | None = None,
    axis: int = 0,
    device: xc.Device | Sharding | None = None,
) -> tuple[Array, Array]: ...
@overload
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: DTypeLike | None = None,
    axis: int = 0,
    *,
    device: xc.Device | Sharding | None = None,
) -> Array | tuple[Array, Array]: ...
@export
def logspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    endpoint: bool = True,
    base: ArrayLike = 10.0,
    dtype: DTypeLike | None = None,
    axis: int = 0,
) -> Array: ...
@export
def geomspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    endpoint: bool = True,
    dtype: DTypeLike | None = None,
    axis: int = 0,
) -> Array: ...
