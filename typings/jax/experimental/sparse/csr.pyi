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
from jax._src.numpy.util import promote_dtypes as promote_dtypes
from jax._src.typing import (
    Array as Array,
    DTypeLike as DTypeLike,
)
from jax.experimental.sparse._base import JAXSparse as JAXSparse
from jax.experimental.sparse.coo import COOInfo as COOInfo
from jax.experimental.sparse.util import (
    CuSparseEfficiencyWarning as CuSparseEfficiencyWarning,
)
from jax.interpreters import mlir as mlir

import jax

type Shape = tuple[int, ...]

class CSR(JAXSparse):
    data: jax.Array
    indices: jax.Array
    indptr: jax.Array
    shape: tuple[int, int]
    nse: Incomplete
    dtype: Incomplete
    def __init__(self, args, *, shape) -> None: ...
    @classmethod
    def fromdense(cls, mat, *, nse=None, index_dtype=...): ...
    def todense(self): ...
    def transpose(self, axes=None): ...
    def __matmul__(self, other): ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, aux_data, children): ...

class CSC(JAXSparse):
    data: jax.Array
    indices: jax.Array
    indptr: jax.Array
    shape: tuple[int, int]
    nse: Incomplete
    dtype: Incomplete
    def __init__(self, args, *, shape) -> None: ...
    @classmethod
    def fromdense(cls, mat, *, nse=None, index_dtype=...): ...
    def todense(self): ...
    def transpose(self, axes=None): ...
    def __matmul__(self, other): ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, aux_data, children): ...

csr_todense_p: Incomplete

def csr_todense(mat: CSR) -> Array: ...

csr_fromdense_p: Incomplete

def csr_fromdense(
    mat: Array,
    *,
    nse: int | None = None,
    index_dtype: DTypeLike = ...,
) -> CSR: ...

csr_matvec_p: Incomplete

def csr_matvec(mat: CSR, v: Array, transpose: bool = False) -> Array: ...

csr_matmat_p: Incomplete

def csr_matmat(mat: CSR, B: Array, *, transpose: bool = False) -> Array: ...
