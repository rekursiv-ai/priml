from collections.abc import Sequence
from typing import Any, overload

from _typeshed import Incomplete
from jax._src import (
    api as api,
    config as config,
    core as core,
    dtypes as dtypes,
    literals as literals,
)
from jax._src.lax import lax as lax
from jax._src.lib import xla_client as xc
from jax._src.sharding import Sharding as Sharding
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    PartitionSpec as P,
    SingleDeviceSharding as SingleDeviceSharding,
    canonicalize_sharding as canonicalize_sharding,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DimSize as DimSize,
    Shape as Shape,
    SupportsNdim as SupportsNdim,
    SupportsShape as SupportsShape,
    SupportsSize as SupportsSize,
)
from jax._src.util import (
    canonicalize_axis_tuple as canonicalize_axis_tuple,
    safe_map as safe_map,
    safe_zip as safe_zip,
    set_module as set_module,
)

zip: Incomplete
unsafe_zip: Incomplete
map: Incomplete
unsafe_map: Incomplete
export: Incomplete

def promote_shapes(fun_name: str, *args: ArrayLike) -> list[Array]: ...
def promote_dtypes(*args: ArrayLike) -> list[Array]: ...
def promote_dtypes_inexact(*args: ArrayLike) -> list[Array]: ...
def promote_dtypes_numeric(*args: ArrayLike) -> list[Array]: ...
def promote_dtypes_complex(*args: ArrayLike) -> list[Array]: ...
@overload
def ensure_arraylike(fun_name: str, /) -> tuple[()]: ...
@overload
def ensure_arraylike(fun_name: str, a1: Any, /) -> Array: ...
@overload
def ensure_arraylike(fun_name: str, a1: Any, a2: Any, /) -> tuple[Array, Array]: ...
@overload
def ensure_arraylike(
    fun_name: str,
    a1: Any,
    a2: Any,
    a3: Any,
    /,
) -> tuple[Array, Array, Array]: ...
@overload
def ensure_arraylike(
    fun_name: str,
    a1: Any,
    a2: Any,
    a3: Any,
    a4: Any,
    /,
    *args: Any,
) -> tuple[Array, ...]: ...
def ensure_arraylike_tuple(fun_name: str, tup: Sequence[Any]) -> tuple[Array, ...]: ...
def check_arraylike(
    fun_name: str,
    *args: Any,
    emit_warning: bool = False,
    stacklevel: int = 3,
): ...
def check_arraylike_or_none(fun_name: str, *args: Any): ...
def check_no_float0s(fun_name: str, *args: Any): ...
def check_for_prngkeys(fun_name: str, *args: Any): ...
def promote_args(fun_name: str, *args: ArrayLike) -> list[Array]: ...
def promote_args_numeric(fun_name: str, *args: ArrayLike) -> list[Array]: ...
def promote_args_inexact(fun_name: str, *args: ArrayLike) -> list[Array]: ...
def canonicalize_device_to_sharding(
    device: xc.Device | Sharding | None,
) -> Sharding | None: ...
def choose_device_or_out_sharding(
    device: xc.Device | Sharding | None,
    out_sharding: NamedSharding | P | None,
    name: str,
) -> Sharding | NamedSharding | None: ...
@export
def ndim(a: ArrayLike | SupportsNdim) -> int: ...
@export
def shape(a: ArrayLike | SupportsShape) -> tuple[int, ...]: ...
@export
def size(
    a: ArrayLike | SupportsSize | SupportsShape,
    axis: int | Sequence[int] | None = None,
) -> int: ...
