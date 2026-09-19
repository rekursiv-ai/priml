from collections.abc import Sequence
from typing import Literal, NamedTuple, overload

from _typeshed import Incomplete
from jax._src import (
    api as api,
    config as config,
    core as core,
)
from jax._src.custom_derivatives import custom_jvp as custom_jvp
from jax._src.lax import lax as lax
from jax._src.numpy import (
    array_creation as array_creation,
    einsum as einsum,
    indexing as indexing,
    reductions as reductions,
    tensor_contractions as tensor_contractions,
    ufuncs as ufuncs,
)
from jax._src.numpy.util import (
    ensure_arraylike as ensure_arraylike,
    promote_dtypes_inexact as promote_dtypes_inexact,
)
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

class EighResult(NamedTuple):
    eigenvalues: Array
    eigenvectors: Array

class EigResult(NamedTuple):
    eigenvalues: Array
    eigenvectors: Array

class QRResult(NamedTuple):
    Q: Array
    R: Array

class SlogdetResult(NamedTuple):
    sign: Array
    logabsdet: Array

class SVDResult(NamedTuple):
    U: Array
    S: Array
    Vh: Array

@export
def cholesky(
    a: ArrayLike,
    *,
    upper: bool = False,
    symmetrize_input: bool = True,
) -> Array: ...
@overload
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    *,
    compute_uv: Literal[True],
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> SVDResult: ...
@overload
def svd(
    a: ArrayLike,
    full_matrices: bool,
    compute_uv: Literal[True],
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> SVDResult: ...
@overload
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    *,
    compute_uv: Literal[False],
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> Array: ...
@overload
def svd(
    a: ArrayLike,
    full_matrices: bool,
    compute_uv: Literal[False],
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> Array: ...
@overload
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian: bool = False,
    subset_by_index: tuple[int, int] | None = None,
) -> Array | SVDResult: ...
@export
def matrix_power(a: ArrayLike, n: int) -> Array: ...
@export
@api.jit
def matrix_rank(
    M: ArrayLike,
    rtol: ArrayLike | None = None,
    *,
    tol: ArrayLike | None = None,
) -> Array: ...
@export
def slogdet(a: ArrayLike, *, method: str | None = None) -> SlogdetResult: ...
@export
@api.jit
def det(a: ArrayLike) -> Array: ...
@export
def eig(a: ArrayLike) -> EigResult: ...
@export
@api.jit
def eigvals(a: ArrayLike) -> Array: ...
@export
def eigh(
    a: ArrayLike,
    UPLO: str | None = None,
    symmetrize_input: bool = True,
) -> EighResult: ...
@export
def eigvalsh(
    a: ArrayLike,
    UPLO: str | None = "L",
    *,
    symmetrize_input: bool = True,
) -> Array: ...
@export
def pinv(
    a: ArrayLike,
    rtol: ArrayLike | None = None,
    hermitian: bool = False,
    *,
    rcond: ArrayLike | None = None,
) -> Array: ...
@export
@api.jit
def inv(a: ArrayLike) -> Array: ...
@export
def norm(
    x: ArrayLike,
    ord: int | str | None = None,
    axis: tuple[int, ...] | int | None = None,
    keepdims: bool = False,
) -> Array: ...
@overload
def qr(
    a: ArrayLike,
    mode: Literal["reduced", "complete", "raw", "full"] = "reduced",
) -> QRResult: ...
@overload
def qr(a: ArrayLike, mode: Literal["r"]) -> Array: ...
@overload
def qr(a: ArrayLike, mode: str) -> Array | QRResult: ...
@export
@api.jit
def solve(a: ArrayLike, b: ArrayLike) -> Array: ...
@export
def lstsq(
    a: ArrayLike,
    b: ArrayLike,
    rcond: float | None = None,
    *,
    numpy_resid: bool = False,
) -> tuple[Array, Array, Array, Array]: ...
@export
def cross(x1: ArrayLike, x2: ArrayLike, /, *, axis: int = -1): ...
@export
def outer(x1: ArrayLike, x2: ArrayLike, /) -> Array: ...
@export
def matrix_norm(
    x: ArrayLike,
    /,
    *,
    keepdims: bool = False,
    ord: str | int = "fro",
) -> Array: ...
@export
def matrix_transpose(x: ArrayLike, /) -> Array: ...
@export
def vector_norm(
    x: ArrayLike,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ord: int | str = 2,
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
def matmul(
    x1: ArrayLike,
    x2: ArrayLike,
    /,
    *,
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> Array: ...
@export
def tensordot(
    x1: ArrayLike,
    x2: ArrayLike,
    /,
    *,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
    precision: lax.PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
@export
def svdvals(x: ArrayLike, /) -> Array: ...
@export
def diagonal(x: ArrayLike, /, *, offset: int = 0) -> Array: ...
@export
def tensorinv(a: ArrayLike, ind: int = 2) -> Array: ...
@export
def tensorsolve(
    a: ArrayLike,
    b: ArrayLike,
    axes: tuple[int, ...] | None = None,
) -> Array: ...
@export
def multi_dot(
    arrays: Sequence[ArrayLike],
    *,
    precision: lax.PrecisionLike = None,
) -> Array: ...
@export
def cond(x: ArrayLike, p=None): ...
@export
def trace(
    x: ArrayLike,
    /,
    *,
    offset: int = 0,
    dtype: DTypeLike | None = None,
) -> Array: ...
