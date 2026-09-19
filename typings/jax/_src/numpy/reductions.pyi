from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, overload

from _typeshed import Incomplete
from jax._src import (
    api as api,
    config as config,
    core as core,
    dtypes as dtypes,
)
from jax._src.lax import control_flow as control_flow
from jax._src.numpy.util import (
    ensure_arraylike as ensure_arraylike,
    promote_dtypes_inexact as promote_dtypes_inexact,
    promote_dtypes_numeric as promote_dtypes_numeric,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DType as DType,
    DTypeLike as DTypeLike,
)
from jax._src.util import (
    canonicalize_axis as canonicalize_axis,
    canonicalize_axis_tuple as canonicalize_axis_tuple,
    maybe_named_axis as maybe_named_axis,
    set_module as set_module,
)

export: Incomplete
type Axis = int | Sequence[int] | None

def check_where(name: str, where: ArrayLike | None) -> Array | None: ...

type ReductionOp = Callable[[Any, Any], Any]

@export
def sum(
    a: ArrayLike,
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool = False,
    initial: ArrayLike | None = None,
    where: ArrayLike | None = None,
    promote_integers: bool = True,
) -> Array: ...
@export
def prod(
    a: ArrayLike,
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool = False,
    initial: ArrayLike | None = None,
    where: ArrayLike | None = None,
    promote_integers: bool = True,
) -> Array: ...
@export
def max(
    a: ArrayLike,
    axis: Axis = None,
    out: None = None,
    keepdims: bool = False,
    initial: ArrayLike | None = None,
    where: ArrayLike | None = None,
) -> Array: ...
@export
def min(
    a: ArrayLike,
    axis: Axis = None,
    out: None = None,
    keepdims: bool = False,
    initial: ArrayLike | None = None,
    where: ArrayLike | None = None,
) -> Array: ...
@export
def all(
    a: ArrayLike,
    axis: Axis = None,
    out: None = None,
    keepdims: bool = False,
    *,
    where: ArrayLike | None = None,
) -> Array: ...
@export
def any(
    a: ArrayLike,
    axis: Axis = None,
    out: None = None,
    keepdims: bool = False,
    *,
    where: ArrayLike | None = None,
) -> Array: ...
@export
def amin(
    a: ArrayLike,
    axis: Axis = None,
    out: None = None,
    keepdims: bool = False,
    initial: ArrayLike | None = None,
    where: ArrayLike | None = None,
) -> Array: ...
@export
def amax(
    a: ArrayLike,
    axis: Axis = None,
    out: None = None,
    keepdims: bool = False,
    initial: ArrayLike | None = None,
    where: ArrayLike | None = None,
) -> Array: ...
@export
def mean(
    a: ArrayLike,
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool = False,
    *,
    where: ArrayLike | None = None,
) -> Array: ...
@overload
def average(
    a: ArrayLike,
    axis: Axis = None,
    weights: ArrayLike | None = None,
    returned: Literal[False] = False,
    keepdims: bool = False,
) -> Array: ...
@overload
def average(
    a: ArrayLike,
    axis: Axis = None,
    weights: ArrayLike | None = None,
    *,
    returned: Literal[True],
    keepdims: bool = False,
) -> Array: ...
@overload
def average(
    a: ArrayLike,
    axis: Axis = None,
    weights: ArrayLike | None = None,
    returned: bool = False,
    keepdims: bool = False,
) -> Array | tuple[Array, Array]: ...
@export
def var(
    a: ArrayLike,
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: ArrayLike | None = None,
    mean: ArrayLike | None = None,
    correction: float | None = None,
) -> Array: ...
@export
def std(
    a: ArrayLike,
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: ArrayLike | None = None,
    mean: ArrayLike | None = None,
    correction: float | None = None,
) -> Array: ...
@export
def ptp(
    a: ArrayLike,
    axis: Axis = None,
    out: None = None,
    keepdims: bool = False,
) -> Array: ...
@export
def count_nonzero(a: ArrayLike, axis: Axis = None, keepdims: bool = False) -> Array: ...
@export
def nanmin(
    a: ArrayLike,
    axis: Axis = None,
    out: None = None,
    keepdims: bool = False,
    initial: ArrayLike | None = None,
    where: ArrayLike | None = None,
) -> Array: ...
@export
def nanmax(
    a: ArrayLike,
    axis: Axis = None,
    out: None = None,
    keepdims: bool = False,
    initial: ArrayLike | None = None,
    where: ArrayLike | None = None,
) -> Array: ...
@export
def nansum(
    a: ArrayLike,
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool = False,
    initial: ArrayLike | None = None,
    where: ArrayLike | None = None,
) -> Array: ...
@export
def nanprod(
    a: ArrayLike,
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool = False,
    initial: ArrayLike | None = None,
    where: ArrayLike | None = None,
) -> Array: ...
@export
def nanmean(
    a: ArrayLike,
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool = False,
    where: ArrayLike | None = None,
) -> Array: ...
@export
def nanvar(
    a: ArrayLike,
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: int = 0,
    keepdims: bool = False,
    where: ArrayLike | None = None,
    mean: ArrayLike | None = None,
) -> Array: ...
@export
def nanstd(
    a: ArrayLike,
    axis: Axis = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: int = 0,
    keepdims: bool = False,
    where: ArrayLike | None = None,
    mean: ArrayLike | None = None,
) -> Array: ...

class CumulativeReduction(Protocol):
    def __call__(
        self,
        a: ArrayLike,
        axis: Axis = None,
        dtype: DTypeLike | None = None,
        out: None = None,
    ) -> Array: ...

@export
def cumsum(
    a: ArrayLike,
    axis: int | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> Array: ...
@export
def cumprod(
    a: ArrayLike,
    axis: int | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> Array: ...
@export
def nancumsum(
    a: ArrayLike,
    axis: int | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> Array: ...
@export
def nancumprod(
    a: ArrayLike,
    axis: int | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> Array: ...
@export
def cumulative_sum(
    x: ArrayLike,
    /,
    *,
    axis: int | None = None,
    dtype: DTypeLike | None = None,
    include_initial: bool = False,
) -> Array: ...
@export
def cumulative_prod(
    x: ArrayLike,
    /,
    *,
    axis: int | None = None,
    dtype: DTypeLike | None = None,
    include_initial: bool = False,
) -> Array: ...
@export
def quantile(
    a: ArrayLike,
    q: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> Array: ...
@export
def nanquantile(
    a: ArrayLike,
    q: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> Array: ...
@export
def percentile(
    a: ArrayLike,
    q: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> Array: ...
@export
def nanpercentile(
    a: ArrayLike,
    q: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> Array: ...
@export
def median(
    a: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> Array: ...
@export
def nanmedian(
    a: ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    out: None = None,
    overwrite_input: bool = False,
    keepdims: bool = False,
) -> Array: ...
