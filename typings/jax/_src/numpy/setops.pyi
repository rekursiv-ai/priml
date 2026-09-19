from typing import NamedTuple

from _typeshed import Incomplete
from jax._src import (
    api as api,
    core as core,
    dtypes as dtypes,
)
from jax._src.lax import lax as lax
from jax._src.numpy.array_creation import (
    empty as empty,
    full as full,
    full_like as full_like,
    ones as ones,
    zeros as zeros,
)
from jax._src.numpy.lax_numpy import (
    append as append,
    arange as arange,
    concatenate as concatenate,
    diff as diff,
    moveaxis as moveaxis,
    nonzero as nonzero,
    ravel as ravel,
    sort as sort,
    where as where,
)
from jax._src.numpy.reductions import (
    any as any,
    cumsum as cumsum,
)
from jax._src.numpy.sorting import lexsort as lexsort
from jax._src.numpy.ufuncs import isnan as isnan
from jax._src.numpy.util import (
    ensure_arraylike as ensure_arraylike,
    promote_dtypes as promote_dtypes,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)
from jax._src.util import (
    canonicalize_axis as canonicalize_axis,
    set_module as set_module,
)

export: Incomplete

@export
def setdiff1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool = False,
    *,
    size: int | None = None,
    fill_value: ArrayLike | None = None,
) -> Array: ...
@export
def union1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    *,
    size: int | None = None,
    fill_value: ArrayLike | None = None,
) -> Array: ...
@export
def setxor1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool = False,
    *,
    size: int | None = None,
    fill_value: ArrayLike | None = None,
) -> Array: ...
@export
def intersect1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: bool = False,
    return_indices: bool = False,
    *,
    size: int | None = None,
    fill_value: ArrayLike | None = None,
) -> Array | tuple[Array, Array, Array]: ...
@export
def isin(
    element: ArrayLike,
    test_elements: ArrayLike,
    assume_unique: bool = False,
    invert: bool = False,
    *,
    method: str = "auto",
) -> Array: ...

UNIQUE_SIZE_HINT: str

@export
def unique(
    ar: ArrayLike,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: int | None = None,
    *,
    equal_nan: bool = True,
    size: int | None = None,
    fill_value: ArrayLike | None = None,
    sorted: bool = True,
): ...

class _UniqueAllResult(NamedTuple):
    values: Array
    indices: Array
    inverse_indices: Array
    counts: Array

class _UniqueCountsResult(NamedTuple):
    values: Array
    counts: Array

class _UniqueInverseResult(NamedTuple):
    values: Array
    inverse_indices: Array

@export
def unique_all(
    x: ArrayLike,
    /,
    *,
    size: int | None = None,
    fill_value: ArrayLike | None = None,
) -> _UniqueAllResult: ...
@export
def unique_counts(
    x: ArrayLike,
    /,
    *,
    size: int | None = None,
    fill_value: ArrayLike | None = None,
) -> _UniqueCountsResult: ...
@export
def unique_inverse(
    x: ArrayLike,
    /,
    *,
    size: int | None = None,
    fill_value: ArrayLike | None = None,
) -> _UniqueInverseResult: ...
@export
def unique_values(
    x: ArrayLike,
    /,
    *,
    size: int | None = None,
    fill_value: ArrayLike | None = None,
) -> Array: ...
