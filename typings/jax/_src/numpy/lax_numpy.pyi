from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import IO, Any, Literal, Protocol, TypeVar, overload

import os

from _typeshed import Incomplete
from jax._src import (
    api as api,
    core as core,
    deprecations as deprecations,
    dtypes as dtypes,
)
from jax._src.custom_derivatives import custom_jvp as custom_jvp
from jax._src.lax import (
    control_flow as control_flow,
    lax as lax,
)
from jax._src.lib import xla_client as xc
from jax._src.mesh import get_abstract_mesh as get_abstract_mesh
from jax._src.numpy import (
    array_creation as array_creation,
    indexing as indexing,
    reductions as reductions,
    tensor_contractions as tensor_contractions,
    ufuncs as ufuncs,
    util as util,
)
from jax._src.numpy.array_constructors import (
    array as array,
    asarray as asarray,
)
from jax._src.numpy.sorting import (
    argsort as argsort,
    sort as sort,
)
from jax._src.numpy.vectorize import vectorize as vectorize
from jax._src.pjit import auto_axes as auto_axes
from jax._src.sharding import Sharding as Sharding
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    PartitionSpec as P,
    canonicalize_sharding as canonicalize_sharding,
)
from jax._src.tree_util import tree_map as tree_map
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DeprecatedArg as DeprecatedArg,
    DimSize as DimSize,
    DType as DType,
    DTypeLike as DTypeLike,
    Shape as Shape,
    SupportsShape as SupportsShape,
)
from jax._src.util import (
    ceil_of_ratio as ceil_of_ratio,
    safe_zip as safe_zip,
    set_module as set_module,
    unzip2 as unzip2,
)

import numpy as np

export: Incomplete
T = TypeVar("T")

def get_printoptions(): ...
def printoptions(*args, **kwargs): ...
def set_printoptions(*args, **kwargs): ...
@export
def iscomplexobj(x: Any) -> bool: ...

iinfo: Incomplete
finfo: Incomplete
can_cast: Incomplete
promote_types: Incomplete
ComplexWarning = np.exceptions.ComplexWarning

@export
def load(
    file: IO[bytes] | str | os.PathLike[Any],
    *args: Any,
    **kwargs: Any,
) -> Array: ...
@export
@api.jit
def fmin(x1: ArrayLike, x2: ArrayLike) -> Array: ...
@export
@api.jit
def fmax(x1: ArrayLike, x2: ArrayLike) -> Array: ...
@export
def issubdtype(arg1: DTypeLike, arg2: DTypeLike) -> bool: ...
@export
def isscalar(element: Any) -> bool: ...
@export
def result_type(*args: Any) -> DType: ...
@export
@api.jit
def trunc(x: ArrayLike) -> Array: ...
@export
def convolve(
    a: ArrayLike,
    v: ArrayLike,
    mode: str = "full",
    *,
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array: ...
@export
def correlate(
    a: ArrayLike,
    v: ArrayLike,
    mode: str = "valid",
    *,
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array: ...
@export
def histogram_bin_edges(
    a: ArrayLike,
    bins: ArrayLike = 10,
    range: Array | Sequence[ArrayLike] | None = None,
    weights: ArrayLike | None = None,
) -> Array: ...
@export
def histogram(
    a: ArrayLike,
    bins: ArrayLike = 10,
    range: Sequence[ArrayLike] | None = None,
    weights: ArrayLike | None = None,
    density: bool | None = None,
) -> tuple[Array, Array]: ...
@export
def histogram2d(
    x: ArrayLike,
    y: ArrayLike,
    bins: ArrayLike | list[ArrayLike] = 10,
    range: Sequence[Array | Sequence[ArrayLike] | None] | None = None,
    weights: ArrayLike | None = None,
    density: bool | None = None,
) -> tuple[Array, Array, Array]: ...
@export
def histogramdd(
    sample: ArrayLike,
    bins: ArrayLike | list[ArrayLike] = 10,
    range: Sequence[Array | Sequence[ArrayLike] | None] | None = None,
    weights: ArrayLike | None = None,
    density: bool | None = None,
) -> tuple[Array, list[Array]]: ...
@export
def transpose(a: ArrayLike, axes: Sequence[int] | None = None) -> Array: ...
@export
def permute_dims(a: ArrayLike, /, axes: tuple[int, ...]) -> Array: ...
@export
def matrix_transpose(x: ArrayLike, /) -> Array: ...
@export
def rot90(m: ArrayLike, k: int = 1, axes: tuple[int, int] = (0, 1)) -> Array: ...
@export
def flip(m: ArrayLike, axis: int | Sequence[int] | None = None) -> Array: ...
@export
def fliplr(m: ArrayLike) -> Array: ...
@export
def flipud(m: ArrayLike) -> Array: ...
@export
@api.jit
def iscomplex(x: ArrayLike) -> Array: ...
@export
@api.jit
def isreal(x: ArrayLike) -> Array: ...
@export
def angle(z: ArrayLike, deg: bool = False) -> Array: ...
@export
def diff(
    a: ArrayLike,
    n: int = 1,
    axis: int = -1,
    prepend: ArrayLike | None = None,
    append: ArrayLike | None = None,
) -> Array: ...
@export
@api.jit
def ediff1d(
    ary: ArrayLike,
    to_end: ArrayLike | None = None,
    to_begin: ArrayLike | None = None,
) -> Array: ...
@export
def gradient(
    f: ArrayLike,
    *varargs: ArrayLike,
    axis: int | Sequence[int] | None = None,
    edge_order: int | None = None,
) -> Array | list[Array]: ...
@export
def isrealobj(x: Any) -> bool: ...
@export
def reshape(
    a: ArrayLike,
    shape: DimSize | Shape,
    order: str = "C",
    *,
    copy: bool | None = None,
    out_sharding=None,
) -> Array: ...
@export
def ravel(a: ArrayLike, order: str = "C", *, out_sharding=None) -> Array: ...
@export
def ravel_multi_index(
    multi_index: Sequence[ArrayLike],
    dims: Sequence[int],
    mode: str = "raise",
    order: str = "C",
    *,
    dtype: DTypeLike | None = None,
) -> Array: ...
@export
def unravel_index(indices: ArrayLike, shape: Shape) -> tuple[Array, ...]: ...
@export
def resize(a: ArrayLike, new_shape: Shape) -> Array: ...
@export
def squeeze(a: ArrayLike, axis: int | Sequence[int] | None = None) -> Array: ...
@export
def expand_dims(a: ArrayLike, axis: int | Sequence[int]) -> Array: ...
@export
def swapaxes(a: ArrayLike, axis1: int, axis2: int) -> Array: ...
@export
def moveaxis(
    a: ArrayLike,
    source: int | Sequence[int],
    destination: int | Sequence[int],
) -> Array: ...
@export
def isclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: ArrayLike = 1e-05,
    atol: ArrayLike = 1e-08,
    equal_nan: bool = False,
) -> Array: ...
@export
def interp(
    x: ArrayLike,
    xp: ArrayLike,
    fp: ArrayLike,
    left: ArrayLike | str | None = None,
    right: ArrayLike | str | None = None,
    period: ArrayLike | None = None,
) -> Array: ...
@overload
def where(
    condition: ArrayLike,
    x: None = None,
    y: None = None,
    /,
    *,
    size: int | None = None,
    fill_value: ArrayLike | tuple[ArrayLike, ...] | None = None,
) -> tuple[Array, ...]: ...
@overload
def where(
    condition: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    /,
    *,
    size: int | None = None,
    fill_value: ArrayLike | tuple[ArrayLike, ...] | None = None,
) -> Array: ...
@overload
def where(
    condition: ArrayLike,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    /,
    *,
    size: int | None = None,
    fill_value: ArrayLike | tuple[ArrayLike, ...] | None = None,
) -> Array | tuple[Array, ...]: ...
@export
def select(
    condlist: Sequence[ArrayLike],
    choicelist: Sequence[ArrayLike],
    default: ArrayLike = 0,
) -> Array: ...
@export
def bincount(
    x: ArrayLike,
    weights: ArrayLike | None = None,
    minlength: int = 0,
    *,
    length: int | None = None,
) -> Array: ...
@overload
def broadcast_shapes(*shapes: Sequence[int]) -> tuple[int, ...]: ...
@overload
def broadcast_shapes(
    *shapes: Sequence[int | core.Tracer],
) -> tuple[int | core.Tracer, ...]: ...
@export
def broadcast_arrays(*args: ArrayLike) -> list[Array]: ...
@export
def broadcast_to(
    array: ArrayLike,
    shape: DimSize | Shape,
    *,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
@export
def split(
    ary: ArrayLike,
    indices_or_sections: int | Sequence[int] | ArrayLike,
    axis: int = 0,
) -> list[Array]: ...
@export
def vsplit(
    ary: ArrayLike,
    indices_or_sections: int | Sequence[int] | ArrayLike,
) -> list[Array]: ...
@export
def hsplit(
    ary: ArrayLike,
    indices_or_sections: int | Sequence[int] | ArrayLike,
) -> list[Array]: ...
@export
def dsplit(
    ary: ArrayLike,
    indices_or_sections: int | Sequence[int] | ArrayLike,
) -> list[Array]: ...
@export
def array_split(
    ary: ArrayLike,
    indices_or_sections: int | Sequence[int] | ArrayLike,
    axis: int = 0,
) -> list[Array]: ...
@export
@api.jit
def clip(
    arr: ArrayLike | None = None,
    /,
    min: ArrayLike | None = None,
    max: ArrayLike | None = None,
    *,
    a: ArrayLike | DeprecatedArg = ...,
    a_min: ArrayLike | DeprecatedArg | None = ...,
    a_max: ArrayLike | DeprecatedArg | None = ...,
) -> Array: ...
@export
def round(a: ArrayLike, decimals: int = 0, out: None = None) -> Array: ...
@export
def around(a: ArrayLike, decimals: int = 0, out: None = None) -> Array: ...
@export
@api.jit
def fix(x: ArrayLike, out: None = None) -> Array: ...
@export
@api.jit
def nan_to_num(
    x: ArrayLike,
    copy: bool = True,
    nan: ArrayLike = 0.0,
    posinf: ArrayLike | None = None,
    neginf: ArrayLike | None = None,
) -> Array: ...
@export
def allclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: ArrayLike = 1e-05,
    atol: ArrayLike = 1e-08,
    equal_nan: bool = False,
) -> Array: ...
@export
def nonzero(
    a: ArrayLike,
    *,
    size: int | None = None,
    fill_value: ArrayLike | tuple[ArrayLike, ...] | None = None,
) -> tuple[Array, ...]: ...
@export
def flatnonzero(
    a: ArrayLike,
    *,
    size: int | None = None,
    fill_value: ArrayLike | tuple[ArrayLike, ...] | None = None,
) -> Array: ...
@export
def unwrap(
    p: ArrayLike,
    discont: ArrayLike | None = None,
    axis: int = -1,
    period: ArrayLike = ...,
) -> Array: ...

type PadValueLike[T] = T | Sequence[T] | Sequence[Sequence[T]]
type PadValue[T] = tuple[tuple[T, T], ...]

class PadStatFunc(Protocol):
    def __call__(
        self,
        array: ArrayLike,
        /,
        *,
        axis: int | None = None,
        keepdims: bool = False,
    ) -> Array: ...

@export
def pad(
    array: ArrayLike,
    pad_width: PadValueLike[int | Array | np.ndarray],
    mode: str | Callable[..., Any] = "constant",
    **kwargs,
) -> Array: ...
@export
def stack(
    arrays: np.ndarray | Array | Sequence[ArrayLike],
    axis: int = 0,
    out: None = None,
    dtype: DTypeLike | None = None,
) -> Array: ...
@export
def unstack(x: ArrayLike, /, *, axis: int = 0) -> tuple[Array, ...]: ...
@export
def tile(A: ArrayLike, reps: DimSize | Sequence[DimSize]) -> Array: ...
@export
def concatenate(
    arrays: np.ndarray | Array | Sequence[ArrayLike],
    axis: int | None = 0,
    dtype: DTypeLike | None = None,
) -> Array: ...
@export
def concat(arrays: Sequence[ArrayLike], /, *, axis: int | None = 0) -> Array: ...
@export
def vstack(
    tup: np.ndarray | Array | Sequence[ArrayLike],
    dtype: DTypeLike | None = None,
) -> Array: ...
@export
def hstack(
    tup: np.ndarray | Array | Sequence[ArrayLike],
    dtype: DTypeLike | None = None,
) -> Array: ...
@export
def dstack(
    tup: np.ndarray | Array | Sequence[ArrayLike],
    dtype: DTypeLike | None = None,
) -> Array: ...
@export
def column_stack(tup: np.ndarray | Array | Sequence[ArrayLike]) -> Array: ...
@export
def choose(
    a: ArrayLike,
    choices: Array | np.ndarray | Sequence[ArrayLike],
    out: None = None,
    mode: str = "raise",
) -> Array: ...
@export
@api.jit
def block(arrays: ArrayLike | list[ArrayLike]) -> Array: ...
@overload
def atleast_1d() -> list[Array]: ...
@overload
def atleast_1d(x: ArrayLike, /) -> Array: ...
@overload
def atleast_1d(x: ArrayLike, y: ArrayLike, /, *arys: ArrayLike) -> list[Array]: ...
@overload
def atleast_2d() -> list[Array]: ...
@overload
def atleast_2d(x: ArrayLike, /) -> Array: ...
@overload
def atleast_2d(x: ArrayLike, y: ArrayLike, /, *arys: ArrayLike) -> list[Array]: ...
@overload
def atleast_3d() -> list[Array]: ...
@overload
def atleast_3d(x: ArrayLike, /) -> Array: ...
@overload
def atleast_3d(x: ArrayLike, y: ArrayLike, /, *arys: ArrayLike) -> list[Array]: ...
@export
def astype(
    x: ArrayLike,
    dtype: DTypeLike | None,
    /,
    *,
    copy: bool = False,
    device: xc.Device | Sharding | None = None,
) -> Array: ...
@export
def copy(a: ArrayLike, order: str | None = None) -> Array: ...
@export
def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan: bool = False) -> Array: ...
@export
def array_equiv(a1: ArrayLike, a2: ArrayLike) -> Array: ...
@export
def frombuffer(
    buffer: bytes | Any,
    dtype: DTypeLike = ...,
    count: int = -1,
    offset: int = 0,
) -> Array: ...
@export
def fromfile(*args, **kwargs) -> None: ...
@export
def fromiter(*args, **kwargs) -> None: ...
@export
def from_dlpack(
    x: Any,
    /,
    *,
    device: xc.Device | Sharding | None = None,
    copy: bool | None = None,
) -> Array: ...
@export
def fromfunction(
    function: Callable[..., Array],
    shape: Any,
    *,
    dtype: DTypeLike = ...,
    **kwargs,
) -> Array: ...
@export
def fromstring(
    string: str,
    dtype: DTypeLike = ...,
    count: int = -1,
    *,
    sep: str,
) -> Array: ...
@export
def eye(
    N: DimSize,
    M: DimSize | None = None,
    k: int | ArrayLike = 0,
    dtype: DTypeLike | None = None,
    *,
    device: xc.Device | Sharding | None = None,
) -> Array: ...
@export
def identity(n: DimSize, dtype: DTypeLike | None = None) -> Array: ...
@export
def arange(
    start: ArrayLike | DimSize,
    stop: ArrayLike | DimSize | None = None,
    step: ArrayLike | None = None,
    dtype: DTypeLike | None = None,
    *,
    device: xc.Device | Sharding | None = None,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
@export
def meshgrid(
    *xi: ArrayLike,
    copy: bool = True,
    sparse: bool = False,
    indexing: str = "xy",
) -> list[Array]: ...
@export
@api.jit
def i0(x: ArrayLike) -> Array: ...
@export
def ix_(*args: ArrayLike) -> tuple[Array, ...]: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: DTypeLike | None = None,
    sparse: Literal[False] = False,
) -> Array: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: DTypeLike | None = None,
    *,
    sparse: Literal[True],
) -> tuple[Array, ...]: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: DTypeLike | None = None,
    sparse: bool = False,
) -> Array | tuple[Array, ...]: ...
@export
def repeat(
    a: ArrayLike,
    repeats: ArrayLike,
    axis: int | None = None,
    *,
    total_repeat_length: int | None = None,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
@export
def trapezoid(
    y: ArrayLike,
    x: ArrayLike | None = None,
    dx: ArrayLike = 1.0,
    axis: int = -1,
) -> Array: ...
@export
def tri(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: DTypeLike | None = None,
) -> Array: ...
@export
def tril(m: ArrayLike, k: int = 0) -> Array: ...
@export
def triu(m: ArrayLike, k: int = 0) -> Array: ...
@export
def trace(
    a: ArrayLike,
    offset: int | ArrayLike = 0,
    axis1: int = 0,
    axis2: int = 1,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> Array: ...
@export
def mask_indices(
    n: int,
    mask_func: Callable[[ArrayLike, int], Array],
    k: int = 0,
    *,
    size: int | None = None,
) -> tuple[Array, Array]: ...
@export
def triu_indices(
    n: DimSize,
    k: DimSize = 0,
    m: DimSize | None = None,
) -> tuple[Array, Array]: ...
@export
def tril_indices(
    n: DimSize,
    k: DimSize = 0,
    m: DimSize | None = None,
) -> tuple[Array, Array]: ...
@export
def triu_indices_from(
    arr: ArrayLike | SupportsShape,
    k: int = 0,
) -> tuple[Array, Array]: ...
@export
def tril_indices_from(
    arr: ArrayLike | SupportsShape,
    k: int = 0,
) -> tuple[Array, Array]: ...
@export
def fill_diagonal(
    a: ArrayLike,
    val: ArrayLike,
    wrap: bool = False,
    *,
    inplace: bool = True,
) -> Array: ...
@export
def diag_indices(n: int, ndim: int = 2) -> tuple[Array, ...]: ...
@export
def diag_indices_from(arr: ArrayLike) -> tuple[Array, ...]: ...
@export
def diagonal(
    a: ArrayLike,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
) -> Array: ...
@export
def diag(v: ArrayLike, k: int = 0) -> Array: ...
@export
def diagflat(v: ArrayLike, k: int = 0) -> Array: ...
@export
def trim_zeros(
    filt: ArrayLike,
    trim: str = "fb",
    axis: int | Sequence[int] | None = None,
) -> Array: ...
def trim_zeros_tol(filt, tol, trim: str = "fb"): ...
@export
def append(arr: ArrayLike, values: ArrayLike, axis: int | None = None) -> Array: ...
@export
def delete(
    arr: ArrayLike,
    obj: ArrayLike | slice,
    axis: int | None = None,
    *,
    assume_unique_indices: bool = False,
) -> Array: ...
@export
def insert(
    arr: ArrayLike,
    obj: ArrayLike | slice,
    values: ArrayLike,
    axis: int | None = None,
) -> Array: ...
@export
def apply_along_axis(
    func1d: Callable,
    axis: int,
    arr: ArrayLike,
    *args,
    **kwargs,
) -> Array: ...
@export
def apply_over_axes(
    func: Callable[[ArrayLike, int], Array],
    a: ArrayLike,
    axes: Sequence[int],
) -> Array: ...
@export
def cross(
    a,
    b,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
): ...
@export
@api.jit
def kron(a: ArrayLike, b: ArrayLike) -> Array: ...
@export
def vander(x: ArrayLike, N: int | None = None, increasing: bool = False) -> Array: ...
@export
def argwhere(
    a: ArrayLike,
    *,
    size: int | None = None,
    fill_value: ArrayLike | None = None,
) -> Array: ...
@export
def argmax(
    a: ArrayLike,
    axis: int | None = None,
    out: None = None,
    keepdims: bool | None = None,
) -> Array: ...
@export
def argmin(
    a: ArrayLike,
    axis: int | None = None,
    out: None = None,
    keepdims: bool | None = None,
) -> Array: ...
@export
def nanargmax(
    a: ArrayLike,
    axis: int | None = None,
    out: None = None,
    keepdims: bool | None = None,
) -> Array: ...
@export
def nanargmin(
    a: ArrayLike,
    axis: int | None = None,
    out: None = None,
    keepdims: bool | None = None,
) -> Array: ...
@export
def roll(
    a: ArrayLike,
    shift: ArrayLike | Sequence[int],
    axis: int | Sequence[int] | None = None,
) -> Array: ...
@export
def rollaxis(a: ArrayLike, axis: int, start: int = 0) -> Array: ...
@export
def packbits(a: ArrayLike, axis: int | None = None, bitorder: str = "big") -> Array: ...
@export
def unpackbits(
    a: ArrayLike,
    axis: int | None = None,
    count: int | None = None,
    bitorder: str = "big",
) -> Array: ...
@export
@api.jit
def gcd(x1: ArrayLike, x2: ArrayLike) -> Array: ...
@export
@api.jit
def lcm(x1: ArrayLike, x2: ArrayLike) -> Array: ...
@export
def extract(
    condition: ArrayLike,
    arr: ArrayLike,
    *,
    size: int | None = None,
    fill_value: ArrayLike = 0,
) -> Array: ...
@export
def compress(
    condition: ArrayLike,
    a: ArrayLike,
    axis: int | None = None,
    *,
    size: int | None = None,
    fill_value: ArrayLike = 0,
    out: None = None,
) -> Array: ...
@export
def cov(
    m: ArrayLike,
    y: ArrayLike | None = None,
    rowvar: bool = True,
    bias: bool = False,
    ddof: int | None = None,
    fweights: ArrayLike | None = None,
    aweights: ArrayLike | None = None,
    dtype: DTypeLike | None = None,
) -> Array: ...
@export
def corrcoef(
    x: ArrayLike,
    y: ArrayLike | None = None,
    rowvar: bool = True,
    dtype: DTypeLike | None = None,
) -> Array: ...
@export
def searchsorted(
    a: ArrayLike,
    v: ArrayLike,
    side: str = "left",
    sorter: ArrayLike | None = None,
    *,
    method: str = "scan",
) -> Array: ...
@export
def digitize(
    x: ArrayLike,
    bins: ArrayLike,
    right: bool = False,
    *,
    method: str | None = None,
) -> Array: ...
@export
def piecewise(
    x: ArrayLike,
    condlist: Array | Sequence[ArrayLike],
    funclist: list[ArrayLike | Callable[..., Array]],
    *args,
    **kw,
) -> Array: ...
