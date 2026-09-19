from collections.abc import Sequence

from _typeshed import Incomplete
from jax._src import (
    api as api,
    core as core,
    dtypes as dtypes,
)
from jax._src.lax import lax as lax
from jax._src.numpy import (
    ufuncs as ufuncs,
    util as util,
)
from jax._src.numpy.vectorize import vectorize as vectorize
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    PartitionSpec as P,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DTypeLike as DTypeLike,
)
from jax._src.util import (
    canonicalize_axis as canonicalize_axis,
    set_module as set_module,
)

export: Incomplete

@export
def dot(
    a: ArrayLike,
    b: ArrayLike,
    *,
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding=None,
) -> Array: ...
@export
def matmul(
    a: ArrayLike,
    b: ArrayLike,
    *,
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
@export
@api.jit
def matvec(x1: ArrayLike, x2: ArrayLike, /) -> Array: ...
@export
@api.jit
def vecmat(x1: ArrayLike, x2: ArrayLike, /) -> Array: ...
@export
def vdot(
    a: ArrayLike,
    b: ArrayLike,
    *,
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array: ...
@export
def vecdot(
    x1: ArrayLike,
    x2: ArrayLike,
    /,
    *,
    axis: int = -1,
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array: ...
@export
def tensordot(
    a: ArrayLike,
    b: ArrayLike,
    axes: int | Sequence[int] | Sequence[Sequence[int]] = 2,
    *,
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
@export
def inner(
    a: ArrayLike,
    b: ArrayLike,
    *,
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array: ...
@export
def outer(a: ArrayLike, b: ArrayLike, out: None = None) -> Array: ...
