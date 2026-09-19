from collections.abc import Sequence
from typing import Any, NamedTuple, Protocol

from _typeshed import Incomplete
from jax import (
    lax as lax,
    tree_util as tree_util,
    vmap as vmap,
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
from jax._src.lax.lax import (
    DotDimensionNumbers as DotDimensionNumbers,
    ranges_like as ranges_like,
    remaining as remaining,
)
from jax._src.lax.slicing import (
    GatherDimensionNumbers as GatherDimensionNumbers,
    GatherScatterMode as GatherScatterMode,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DTypeLike as DTypeLike,
)
from jax._src.util import (
    canonicalize_axis as canonicalize_axis,
    safe_zip as safe_zip,
    split_list as split_list,
    unzip2 as unzip2,
)
from jax.experimental.sparse._base import JAXSparse as JAXSparse
from jax.experimental.sparse._lowerings import (
    coo_spmm_p as coo_spmm_p,
    coo_spmv_p as coo_spmv_p,
)
from jax.experimental.sparse.util import (
    CuSparseEfficiencyWarning as CuSparseEfficiencyWarning,
    Shape as Shape,
    SparseEfficiencyError as SparseEfficiencyError,
    SparseEfficiencyWarning as SparseEfficiencyWarning,
    SparseInfo as SparseInfo,
    nfold_vmap as nfold_vmap,
)

CUSPARSE_DATA_DTYPES: Incomplete
CUSPARSE_INDEX_DTYPES: Incomplete

def bcoo_eliminate_zeros(mat: BCOO, nse: int | None = None) -> BCOO: ...

class BCOOProperties(NamedTuple):
    n_batch: int
    n_sparse: int
    n_dense: int
    nse: int

class Buffer(Protocol):
    @property
    def shape(self) -> Shape: ...
    @property
    def dtype(self) -> Any: ...

bcoo_todense_p: Incomplete

def bcoo_todense(mat: BCOO) -> Array: ...

bcoo_fromdense_p: Incomplete

def bcoo_fromdense(
    mat: Array,
    *,
    nse: int | None = None,
    n_batch: int = 0,
    n_dense: int = 0,
    index_dtype: DTypeLike = ...,
) -> BCOO: ...

bcoo_extract_p: Incomplete

def bcoo_extract(
    sparr: BCOO,
    arr: ArrayLike,
    *,
    assume_unique: bool | None = None,
) -> BCOO: ...

bcoo_transpose_p: Incomplete

def bcoo_transpose(mat: BCOO, *, permutation: Sequence[int]) -> BCOO: ...

bcoo_dot_general_p: Incomplete

def bcoo_dot_general(
    lhs: BCOO | Array,
    rhs: BCOO | Array,
    *,
    dimension_numbers: DotDimensionNumbers,
    precision: None = None,
    preferred_element_type: None = None,
    out_sharding=None,
) -> BCOO | Array: ...

bcoo_dot_general_sampled_p: Incomplete

def bcoo_dot_general_sampled(
    A: Array,
    B: Array,
    indices: Array,
    *,
    dimension_numbers: DotDimensionNumbers,
) -> Array: ...

bcoo_spdot_general_p: Incomplete
bcoo_sort_indices_p: Incomplete

def bcoo_sort_indices(mat: BCOO) -> BCOO: ...

bcoo_sum_duplicates_p: Incomplete

def bcoo_sum_duplicates(mat: BCOO, nse: int | None = None) -> BCOO: ...
def bcoo_update_layout(
    mat: BCOO,
    *,
    n_batch: int | None = None,
    n_dense: int | None = None,
    on_inefficient: str | None = "error",
) -> BCOO: ...
def bcoo_broadcast_in_dim(
    mat: BCOO,
    *,
    shape: Shape,
    broadcast_dimensions: Sequence[int],
    sharding=None,
) -> BCOO: ...
def bcoo_concatenate(operands: Sequence[BCOO], *, dimension: int) -> BCOO: ...
def bcoo_reshape(
    mat: BCOO,
    *,
    new_sizes: Sequence[int],
    dimensions: Sequence[int] | None = None,
    sharding=None,
) -> BCOO: ...
def bcoo_rev(operand, dimensions): ...
def bcoo_squeeze(arr: BCOO, *, dimensions: Sequence[int]) -> BCOO: ...
def bcoo_slice(
    mat: BCOO,
    *,
    start_indices: Sequence[int],
    limit_indices: Sequence[int],
    strides: Sequence[int] | None = None,
) -> BCOO: ...
def bcoo_dynamic_slice(
    mat: BCOO,
    start_indices: Sequence[Any],
    slice_sizes: Sequence[int],
) -> BCOO: ...
def bcoo_reduce_sum(mat: BCOO, *, axes: Sequence[int]) -> BCOO: ...
def bcoo_multiply_sparse(lhs: BCOO, rhs: BCOO) -> BCOO: ...
def bcoo_multiply_dense(sp_mat: BCOO, v: Array) -> Array: ...
def bcoo_gather(
    operand: BCOO,
    start_indices: Array,
    dimension_numbers: GatherDimensionNumbers,
    slice_sizes: Shape,
    *,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    mode: str | GatherScatterMode | None = None,
    fill_value=None,
) -> BCOO: ...
def bcoo_conv_general_dilated(
    lhs,
    rhs,
    *,
    window_strides,
    padding,
    lhs_dilation=None,
    rhs_dilation=None,
    dimension_numbers=None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    precision=None,
    preferred_element_type=None,
    out_sharding=None,
) -> BCOO: ...

class BCOO(JAXSparse):
    data: Array
    indices: Array
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
        args: tuple[Array, Array],
        *,
        shape: Sequence[int],
        indices_sorted: bool = False,
        unique_indices: bool = False,
    ) -> None: ...
    def reshape(self, *args, **kwargs) -> BCOO: ...
    def astype(self, *args, **kwargs) -> BCOO: ...
    def sum(self) -> BCOO: ...
    @classmethod
    def fromdense(
        cls,
        mat: Array,
        *,
        nse: int | None = None,
        index_dtype: DTypeLike = ...,
        n_dense: int = 0,
        n_batch: int = 0,
    ) -> BCOO: ...
    @classmethod
    def from_scipy_sparse(
        cls,
        mat,
        *,
        index_dtype: DTypeLike | None = None,
        n_dense: int = 0,
        n_batch: int = 0,
    ) -> BCOO: ...
    def update_layout(
        self,
        *,
        n_batch: int | None = None,
        n_dense: int | None = None,
        on_inefficient: str = "error",
    ) -> BCOO: ...
    def sum_duplicates(
        self,
        nse: int | None = None,
        remove_zeros: bool = True,
    ) -> BCOO: ...
    def sort_indices(self) -> BCOO: ...
    def todense(self) -> Array: ...
    def transpose(self, axes: Sequence[int] | None = None) -> BCOO: ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, aux_data, children): ...
