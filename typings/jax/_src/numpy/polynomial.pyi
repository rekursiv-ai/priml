from _typeshed import Incomplete
from jax._src import (
    api as api,
    core as core,
    dtypes as dtypes,
)
from jax._src.lax import (
    control_flow as control_flow,
    lax as lax,
)
from jax._src.numpy import linalg as linalg
from jax._src.numpy.array_creation import (
    full as full,
    ones as ones,
    zeros as zeros,
)
from jax._src.numpy.lax_numpy import (
    arange as arange,
    argmin as argmin,
    array as array,
    atleast_1d as atleast_1d,
    concatenate as concatenate,
    convolve as convolve,
    diag as diag,
    finfo as finfo,
    roll as roll,
    trim_zeros as trim_zeros,
    trim_zeros_tol as trim_zeros_tol,
    vander as vander,
)
from jax._src.numpy.reductions import all as all
from jax._src.numpy.tensor_contractions import (
    dot as dot,
    outer as outer,
)
from jax._src.numpy.ufuncs import (
    maximum as maximum,
    sqrt as sqrt,
    true_divide as true_divide,
)
from jax._src.numpy.util import (
    ensure_arraylike as ensure_arraylike,
    promote_dtypes as promote_dtypes,
    promote_dtypes_inexact as promote_dtypes_inexact,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)
from jax._src.util import set_module as set_module

export: Incomplete

@export
def roots(p: ArrayLike, *, strip_zeros: bool = True) -> Array: ...
@export
def polyfit(
    x: ArrayLike,
    y: ArrayLike,
    deg: int,
    rcond: float | None = None,
    full: bool = False,
    w: ArrayLike | None = None,
    cov: bool = False,
) -> Array | tuple[Array, ...]: ...
@export
@api.jit
def poly(seq_of_zeros: ArrayLike) -> Array: ...
@export
def polyval(p: ArrayLike, x: ArrayLike, *, unroll: int = 16) -> Array: ...
@export
@api.jit
def polyadd(a1: ArrayLike, a2: ArrayLike) -> Array: ...
@export
def polyint(p: ArrayLike, m: int = 1, k: int | ArrayLike | None = None) -> Array: ...
@export
def polyder(p: ArrayLike, m: int = 1) -> Array: ...
@export
def polymul(
    a1: ArrayLike,
    a2: ArrayLike,
    *,
    trim_leading_zeros: bool = False,
) -> Array: ...
@export
def polydiv(
    u: ArrayLike,
    v: ArrayLike,
    *,
    trim_leading_zeros: bool = False,
) -> tuple[Array, Array]: ...
@export
@api.jit
def polysub(a1: ArrayLike, a2: ArrayLike) -> Array: ...
