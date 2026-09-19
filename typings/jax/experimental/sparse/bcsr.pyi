from collections.abc import Sequence
from typing import NamedTuple

from _typeshed import Incomplete
from jax import (
    lax as lax,
    tree_util as tree_util,
)
from jax._src import (
    api_util as api_util,
    config as config,
    core as core,
    dispatch as dispatch,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.lax.lax import DotDimensionNumbers as DotDimensionNumbers
from jax._src.lib.mlir.dialects import hlo as hlo
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DTypeLike as DTypeLike,
)
from jax._src.util import (
    safe_zip as safe_zip,
    split_list as split_list,
)
from jax.experimental.sparse import bcoo as bcoo
from jax.experimental.sparse._base import JAXSparse as JAXSparse
from jax.experimental.sparse.util import (
    CuSparseEfficiencyWarning as CuSparseEfficiencyWarning,
    Shape as Shape,
    SparseEfficiencyWarning as SparseEfficiencyWarning,
    SparseInfo as SparseInfo,
    nfold_vmap as nfold_vmap,
)

import jax

def bcsr_eliminate_zeros(mat: BCSR, nse: int | None = None) -> BCSR: ...
def bcsr_sum_duplicates(mat: BCSR, nse: int | None = None) -> BCSR: ...

class BCSRProperties(NamedTuple):
    n_batch: int
    n_dense: int
    nse: int

bcsr_fromdense_p: Incomplete

def bcsr_fromdense(
    mat: ArrayLike,
    *,
    nse: int | None = None,
    n_batch: int = 0,
    n_dense: int = 0,
    index_dtype: DTypeLike = ...,
) -> BCSR: ...

bcsr_todense_p: Incomplete

def bcsr_todense(mat: BCSR) -> Array: ...

bcsr_extract_p: Incomplete

def bcsr_extract(indices: ArrayLike, indptr: ArrayLike, mat: ArrayLike) -> Array: ...

bcsr_dot_general_p: Incomplete

def bcsr_dot_general(
    lhs: BCSR | Array,
    rhs: Array,
    *,
    dimension_numbers: DotDimensionNumbers,
    precision: None = None,
    preferred_element_type: None = None,
    out_sharding=None,
) -> Array: ...
def bcsr_broadcast_in_dim(
    mat: BCSR,
    *,
    shape: Shape,
    broadcast_dimensions: Sequence[int],
    sharding=None,
) -> BCSR: ...
def bcsr_concatenate(operands: Sequence[BCSR], *, dimension: int) -> BCSR: ...

class BCSR(JAXSparse):
    data: jax.Array
    indices: jax.Array
    indptr: jax.Array
    shape: Shape
    nse: Incomplete
    dtype: Incomplete
    n_batch: Incomplete
    n_sparse: Incomplete
    n_dense: Incomplete
    indices_sorted: bool
    unique_indices: bool
    def __init__(
        self,
        args: tuple[Array, Array, Array],
        *,
        shape: Sequence[int],
        indices_sorted: bool = False,
        unique_indices: bool = False,
    ) -> None: ...
    def transpose(self, *args, **kwargs) -> None: ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, aux_data, children): ...
    def sum_duplicates(
        self,
        nse: int | None = None,
        remove_zeros: bool = True,
    ) -> BCSR: ...
    @classmethod
    def fromdense(
        cls,
        mat,
        *,
        nse=None,
        index_dtype=...,
        n_dense: int = 0,
        n_batch: int = 0,
    ): ...
    def todense(self): ...
    def to_bcoo(self) -> bcoo.BCOO: ...
    @classmethod
    def from_bcoo(cls, arr: bcoo.BCOO) -> BCSR: ...
    @classmethod
    def from_scipy_sparse(
        cls,
        mat,
        *,
        index_dtype=None,
        n_dense: int = 0,
        n_batch: int = 0,
    ): ...
