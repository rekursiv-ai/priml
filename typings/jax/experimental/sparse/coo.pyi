from typing import Any, NamedTuple

from _typeshed import Incomplete
from jax import (
    lax as lax,
    tree_util as tree_util,
)
from jax._src import (
    core as core,
    dispatch as dispatch,
)
from jax._src.interpreters import ad as ad
from jax._src.lib.mlir.dialects import hlo as hlo
from jax._src.numpy.util import promote_dtypes as promote_dtypes
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DTypeLike as DTypeLike,
)
from jax.experimental.sparse._base import JAXSparse as JAXSparse
from jax.experimental.sparse.util import (
    CuSparseEfficiencyWarning as CuSparseEfficiencyWarning,
)
from jax.interpreters import mlir as mlir

import jax

type Dtype = Any
type Shape = tuple[int, ...]

class COOInfo(NamedTuple):
    shape: Shape
    rows_sorted: bool = ...
    cols_sorted: bool = ...

class COO(JAXSparse):
    data: jax.Array
    row: jax.Array
    col: jax.Array
    shape: tuple[int, int]
    nse: Incomplete
    dtype: Incomplete
    def __init__(
        self,
        args: tuple[Array, Array, Array],
        *,
        shape: Shape,
        rows_sorted: bool = False,
        cols_sorted: bool = False,
    ) -> None: ...
    @classmethod
    def fromdense(
        cls,
        mat: Array,
        *,
        nse: int | None = None,
        index_dtype: DTypeLike = ...,
    ) -> COO: ...
    def todense(self) -> Array: ...
    def transpose(self, axes: tuple[int, ...] | None = None) -> COO: ...
    def tree_flatten(self) -> tuple[tuple[Array, Array, Array], dict[str, Any]]: ...
    @classmethod
    def tree_unflatten(cls, aux_data, children): ...
    def __matmul__(self, other: ArrayLike) -> Array: ...

coo_todense_p: Incomplete

def coo_todense(mat: COO) -> Array: ...

coo_fromdense_p: Incomplete

def coo_fromdense(
    mat: Array,
    *,
    nse: int | None = None,
    index_dtype: DTypeLike = ...,
) -> COO: ...

coo_matvec_p: Incomplete

def coo_matvec(mat: COO, v: Array, transpose: bool = False) -> Array: ...

coo_matmat_p: Incomplete

def coo_matmat(mat: COO, B: Array, *, transpose: bool = False) -> Array: ...
