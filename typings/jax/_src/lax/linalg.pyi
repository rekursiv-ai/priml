from collections.abc import Callable as Callable
from typing import Any, Literal, overload

import enum

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    api as api,
    config as config,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    ffi as ffi,
)
from jax._src.core import (
    ShapedArray as ShapedArray,
    is_constant_dim as is_constant_dim,
    is_constant_shape as is_constant_shape,
)
from jax._src.custom_partitioning_sharding_rule import (
    sdy_sharding_rule_to_mlir as sdy_sharding_rule_to_mlir,
    str_to_sdy_sharding_rule as str_to_sdy_sharding_rule,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.lax import (
    control_flow as control_flow,
    lax as lax,
)
from jax._src.lib import (
    cuda_versions as cuda_versions,
    gpu_linalg as gpu_linalg,
    gpu_solver as gpu_solver,
    gpu_sparse as gpu_sparse,
    lapack as lapack,
)
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import (
    chlo as chlo,
    hlo as hlo,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

def register_module_custom_calls(module) -> None: ...
def cholesky(x: Array, *, symmetrize_input: bool = True) -> Array: ...
def cholesky_update(r_matrix: ArrayLike, w_vector: ArrayLike) -> Array: ...

class EigImplementation(enum.Enum):
    CUSOLVER = "cusolver"
    MAGMA = "magma"
    LAPACK = "lapack"

def eig(
    x: ArrayLike,
    *,
    compute_left_eigenvectors: bool = True,
    compute_right_eigenvectors: bool = True,
    implementation: EigImplementation | None = None,
    use_magma: bool | None = None,
) -> list[Array]: ...

class EighImplementation(enum.Enum):
    QR = "qr"
    JACOBI = "jacobi"
    QDWH = "qdwh"

def eigh(
    x: Array,
    *,
    lower: bool = True,
    symmetrize_input: bool = True,
    sort_eigenvalues: bool = True,
    subset_by_index: tuple[int, int] | None = None,
    implementation: EighImplementation | None = None,
) -> tuple[Array, Array]: ...
def hessenberg(a: ArrayLike) -> tuple[Array, Array]: ...
def householder_product(a: ArrayLike, taus: ArrayLike) -> Array: ...
def lu(x: ArrayLike) -> tuple[Array, Array, Array]: ...
def lu_pivots_to_permutation(pivots: ArrayLike, permutation_size: int) -> Array: ...
@overload
def qr(
    x: ArrayLike,
    *,
    pivoting: Literal[False] = False,
    full_matrices: bool = True,
    use_magma: bool | None = None,
) -> tuple[Array, Array]: ...
@overload
def qr(
    x: ArrayLike,
    *,
    pivoting: Literal[True],
    full_matrices: bool = True,
    use_magma: bool | None = None,
) -> tuple[Array, Array, Array]: ...
@overload
def qr(
    x: ArrayLike,
    *,
    pivoting: bool = False,
    full_matrices: bool = True,
    use_magma: bool | None = None,
) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...
def schur(
    x: ArrayLike,
    *,
    compute_schur_vectors: bool = True,
    sort_eig_vals: bool = False,
    select_callable: Callable[..., Any] | None = None,
) -> tuple[Array, Array]: ...

class SvdAlgorithm(enum.Enum):
    DEFAULT = "default"
    QR = "QR"
    JACOBI = "Jacobi"
    POLAR = "polar"

@overload
def svd(
    x: ArrayLike,
    *,
    full_matrices: bool = True,
    compute_uv: Literal[True],
    subset_by_index: tuple[int, int] | None = None,
    algorithm: SvdAlgorithm | None = None,
) -> tuple[Array, Array, Array]: ...
@overload
def svd(
    x: ArrayLike,
    *,
    full_matrices: bool = True,
    compute_uv: Literal[False],
    subset_by_index: tuple[int, int] | None = None,
    algorithm: SvdAlgorithm | None = None,
) -> Array: ...
@overload
def svd(
    x: ArrayLike,
    *,
    full_matrices: bool = True,
    compute_uv: bool = True,
    subset_by_index: tuple[int, int] | None = None,
    algorithm: SvdAlgorithm | None = None,
) -> Array | tuple[Array, Array, Array]: ...
def symmetric_product(
    a_matrix: ArrayLike,
    c_matrix: ArrayLike,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    symmetrize_output: bool = False,
): ...
def triangular_solve(
    a: ArrayLike,
    b: ArrayLike,
    *,
    left_side: bool = False,
    lower: bool = False,
    transpose_a: bool = False,
    conjugate_a: bool = False,
    unit_diagonal: bool = False,
) -> Array: ...
def tridiagonal(
    a: ArrayLike,
    *,
    lower: bool = True,
) -> tuple[Array, Array, Array, Array]: ...
def tridiagonal_solve(dl: Array, d: Array, du: Array, b: Array) -> Array: ...
def register_cpu_gpu_lowering(
    prim,
    lowering_rule,
    supported_platforms=("cpu", "cuda", "rocm"),
) -> None: ...
def linalg_shape_rule(
    multiple_results,
    supports_batching,
    ranks,
    result_shape,
    name,
    *avals,
    **kwargs,
): ...
def linalg_sharding_rule(
    multiple_results,
    shape_rule,
    ranks,
    name,
    *avals,
    **kwargs,
): ...
def linalg_vma_rule(multiple_results, shape_rule, name, *avals, **kwargs): ...
def linalg_primitive(
    result_dtype,
    accepted_dtypes,
    ranks,
    result_shape,
    name,
    multiple_results: bool = False,
    supports_batching: bool = True,
    require_same: bool = True,
): ...

standard_linalg_primitive: Incomplete
cholesky_p: Incomplete
cholesky_update_p: Incomplete

def eig_jvp_rule(
    primals,
    tangents,
    *,
    compute_left_eigenvectors,
    compute_right_eigenvectors,
    implementation,
): ...

eig_p: Incomplete
eigh_p: Incomplete
hessenberg_p: Incomplete
householder_product_p: Incomplete
lu_p: Incomplete

def lu_solve(
    lu: ArrayLike,
    permutation: ArrayLike,
    b: ArrayLike,
    trans: int = 0,
) -> Array: ...

lu_pivots_to_permutation_p: Incomplete

def geqrf(a: ArrayLike) -> tuple[Array, Array]: ...

geqrf_p: Incomplete

def geqp3(
    a: ArrayLike,
    jpvt: ArrayLike,
    *,
    use_magma: bool | None = None,
) -> tuple[Array, Array, Array]: ...

geqp3_p: Incomplete

def qr_jvp_rule(primals, tangents, *, pivoting, full_matrices, use_magma): ...

qr_p: Incomplete
schur_p: Incomplete
svd_p: Incomplete
symmetric_product_p: Incomplete
triangular_solve_p: Incomplete
tridiagonal_p: Incomplete
tridiagonal_solve_p: Incomplete

def symmetrize(x: Array) -> Array: ...
