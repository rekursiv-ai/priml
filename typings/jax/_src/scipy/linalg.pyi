from typing import Any, Literal, overload

from jax._src import (
    config as config,
    dtypes as dtypes,
    lax as lax,
)
from jax._src.api import (
    jit as jit,
    jvp as jvp,
    vmap as vmap,
)
from jax._src.numpy.util import (
    check_arraylike as check_arraylike,
    promote_args_inexact as promote_args_inexact,
    promote_dtypes as promote_dtypes,
    promote_dtypes_complex as promote_dtypes_complex,
    promote_dtypes_inexact as promote_dtypes_inexact,
)
from jax._src.tpu.linalg import qdwh as qdwh
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

def cholesky(
    a: ArrayLike,
    lower: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> Array: ...
def cho_factor(
    a: ArrayLike,
    lower: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[Array, bool]: ...
def cho_solve(
    c_and_lower: tuple[ArrayLike, bool],
    b: ArrayLike,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> Array: ...
@overload
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    compute_uv: Literal[True] = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: str = "gesdd",
) -> tuple[Array, Array, Array]: ...
@overload
def svd(
    a: ArrayLike,
    full_matrices: bool,
    compute_uv: Literal[False],
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: str = "gesdd",
) -> Array: ...
@overload
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    *,
    compute_uv: Literal[False],
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: str = "gesdd",
) -> Array: ...
@overload
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    compute_uv: bool = True,
    overwrite_a: bool = False,
    check_finite: bool = True,
    lapack_driver: str = "gesdd",
) -> Array | tuple[Array, Array, Array]: ...
def det(
    a: ArrayLike,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> Array: ...
@overload
def eigh(
    a: ArrayLike,
    b: ArrayLike | None = None,
    lower: bool = True,
    eigvals_only: Literal[False] = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    turbo: bool = True,
    eigvals: None = None,
    type: int = 1,
    check_finite: bool = True,
) -> tuple[Array, Array]: ...
@overload
def eigh(
    a: ArrayLike,
    b: ArrayLike | None = None,
    lower: bool = True,
    *,
    eigvals_only: Literal[True],
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    turbo: bool = True,
    eigvals: None = None,
    type: int = 1,
    check_finite: bool = True,
) -> Array: ...
@overload
def eigh(
    a: ArrayLike,
    b: ArrayLike | None,
    lower: bool,
    eigvals_only: Literal[True],
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    turbo: bool = True,
    eigvals: None = None,
    type: int = 1,
    check_finite: bool = True,
) -> Array: ...
@overload
def eigh(
    a: ArrayLike,
    b: ArrayLike | None = None,
    lower: bool = True,
    eigvals_only: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    turbo: bool = True,
    eigvals: None = None,
    type: int = 1,
    check_finite: bool = True,
) -> Array | tuple[Array, Array]: ...
def schur(a: ArrayLike, output: str = "real") -> tuple[Array, Array]: ...
def inv(
    a: ArrayLike,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> Array: ...
def lu_factor(
    a: ArrayLike,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[Array, Array]: ...
def lu_solve(
    lu_and_piv: tuple[Array, ArrayLike],
    b: ArrayLike,
    trans: int = 0,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> Array: ...
@overload
def lu(
    a: ArrayLike,
    permute_l: Literal[False] = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[Array, Array, Array]: ...
@overload
def lu(
    a: ArrayLike,
    permute_l: Literal[True],
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[Array, Array]: ...
@overload
def lu(
    a: ArrayLike,
    permute_l: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...
@overload
def qr(
    a: ArrayLike,
    overwrite_a: bool = False,
    lwork: Any = None,
    *,
    mode: Literal["full", "economic"],
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[Array, Array]: ...
@overload
def qr(
    a: ArrayLike,
    overwrite_a: bool = False,
    lwork: Any = None,
    *,
    mode: Literal["full", "economic"],
    pivoting: Literal[True] = True,
    check_finite: bool = True,
) -> tuple[Array, Array, Array]: ...
@overload
def qr(
    a: ArrayLike,
    overwrite_a: bool = False,
    lwork: Any = None,
    *,
    mode: Literal["full", "economic"],
    pivoting: bool = False,
    check_finite: bool = True,
) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...
@overload
def qr(
    a: ArrayLike,
    overwrite_a: bool = False,
    lwork: Any = None,
    *,
    mode: Literal["r"],
    pivoting: Literal[False] = False,
    check_finite: bool = True,
) -> tuple[Array]: ...
@overload
def qr(
    a: ArrayLike,
    overwrite_a: bool = False,
    lwork: Any = None,
    *,
    mode: Literal["r"],
    pivoting: Literal[True] = True,
    check_finite: bool = True,
) -> tuple[Array, Array]: ...
@overload
def qr(
    a: ArrayLike,
    overwrite_a: bool = False,
    lwork: Any = None,
    *,
    mode: Literal["r"],
    pivoting: bool = False,
    check_finite: bool = True,
) -> tuple[Array] | tuple[Array, Array]: ...
@overload
def qr(
    a: ArrayLike,
    overwrite_a: bool = False,
    lwork: Any = None,
    mode: str = "full",
    pivoting: bool = False,
    check_finite: bool = True,
) -> tuple[Array] | tuple[Array, Array] | tuple[Array, Array, Array]: ...
def solve(
    a: ArrayLike,
    b: ArrayLike,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    debug: bool = False,
    check_finite: bool = True,
    assume_a: str = "gen",
) -> Array: ...
def solve_triangular(
    a: ArrayLike,
    b: ArrayLike,
    trans: int | str = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    debug: Any = None,
    check_finite: bool = True,
) -> Array: ...
def expm(
    A: ArrayLike,
    *,
    upper_triangular: bool = False,
    max_squarings: int = 16,
) -> Array: ...
@overload
def expm_frechet(
    A: ArrayLike,
    E: ArrayLike,
    *,
    method: str | None = None,
    compute_expm: Literal[True] = True,
) -> tuple[Array, Array]: ...
@overload
def expm_frechet(
    A: ArrayLike,
    E: ArrayLike,
    *,
    method: str | None = None,
    compute_expm: Literal[False],
) -> Array: ...
@overload
def expm_frechet(
    A: ArrayLike,
    E: ArrayLike,
    *,
    method: str | None = None,
    compute_expm: bool = True,
) -> Array | tuple[Array, Array]: ...
@jit
def block_diag(*arrs: ArrayLike) -> Array: ...
def eigh_tridiagonal(
    d: ArrayLike,
    e: ArrayLike,
    *,
    eigvals_only: bool = False,
    select: str = "a",
    select_range: tuple[float, float] | None = None,
    tol: float | None = None,
) -> Array: ...
def polar(
    a: ArrayLike,
    side: str = "right",
    *,
    method: str = "qdwh",
    eps: float | None = None,
    max_iterations: int | None = None,
) -> tuple[Array, Array]: ...
def sqrtm(A: ArrayLike, blocksize: int = 1) -> Array: ...
def rsf2csf(
    T: ArrayLike,
    Z: ArrayLike,
    check_finite: bool = True,
) -> tuple[Array, Array]: ...
@overload
def hessenberg(
    a: ArrayLike,
    *,
    calc_q: Literal[False],
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> Array: ...
@overload
def hessenberg(
    a: ArrayLike,
    *,
    calc_q: Literal[True],
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> tuple[Array, Array]: ...
def toeplitz(c: ArrayLike, r: ArrayLike | None = None) -> Array: ...
def hilbert(n: int) -> Array: ...
def pascal(n: int, kind: str | None = None) -> Array: ...
def solve_sylvester(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    *,
    method: str = "schur",
    tol: float = 1e-08,
) -> Array: ...
