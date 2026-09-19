from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any, overload

from _typeshed import Incomplete
from jax._src import (
    api as api,
    core as core,
    dtypes as dtypes,
)
from jax._src.export import shape_poly as shape_poly
from jax._src.lax import lax as lax
from jax._src.numpy import util as util
from jax._src.pjit import auto_axes as auto_axes
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    canonicalize_sharding as canonicalize_sharding,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DTypeLike as DTypeLike,
)
from jax._src.util import (
    partition_list as partition_list,
    set_module as set_module,
    unzip2 as unzip2,
)

import opt_einsum

export: Incomplete

class Unoptimized(opt_einsum.paths.PathOptimizer):
    def __call__(self, inputs, *args, **kwargs): ...

@overload
def einsum(
    subscript: str,
    /,
    *operands: ArrayLike,
    out: None = None,
    optimize: str | bool | list[tuple[int, ...]] = "auto",
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    _dot_general: Callable[..., Array] = ...,
    out_sharding=None,
) -> Array: ...
@overload
def einsum(
    arr: ArrayLike,
    axes: Sequence[Any],
    /,
    *operands: ArrayLike | Sequence[Any],
    out: None = None,
    optimize: str | bool | list[tuple[int, ...]] = "auto",
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    _dot_general: Callable[..., Array] = ...,
    out_sharding=None,
) -> Array: ...
@overload
def einsum_path(
    subscripts: str,
    /,
    *operands: ArrayLike,
    optimize: bool | str | list[tuple[int, ...]] = ...,
) -> tuple[list[tuple[int, ...]], Any]: ...
@overload
def einsum_path(
    arr: ArrayLike,
    axes: Sequence[Any],
    /,
    *operands: ArrayLike | Sequence[Any],
    optimize: bool | str | list[tuple[int, ...]] = ...,
) -> tuple[list[tuple[int, ...]], Any]: ...
