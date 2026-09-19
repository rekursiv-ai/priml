from collections.abc import Sequence
from typing import Any, NamedTuple

import dataclasses
import enum

from _typeshed import Incomplete
from jax._src import (
    api as api,
    array as array,
    config as config,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    errors as errors,
    indexing as indexing,
    literals as literals,
)
from jax._src.lax import (
    lax as lax,
    slicing as slicing,
)
from jax._src.numpy import (
    array_constructors as array_constructors,
    einsum as einsum,
    lax_numpy as lax_numpy,
    ufuncs as ufuncs,
    util as util,
)
from jax._src.partition_spec import PartitionSpec as PartitionSpec
from jax._src.pjit import auto_axes as auto_axes
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    canonicalize_sharding as canonicalize_sharding,
)
from jax._src.tree_util import (
    register_pytree_node_class as register_pytree_node_class,
    tree_flatten as tree_flatten,
    tree_unflatten as tree_unflatten,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    Index as Index,
    StaticScalar as StaticScalar,
)
from jax._src.util import (
    canonicalize_axis as canonicalize_axis,
    safe_zip as safe_zip,
    set_module as set_module,
    tuple_update as tuple_update,
    unzip3 as unzip3,
)

export: Incomplete

class IndexType(enum.Enum):
    NONE = "none"
    SLICE = "slice"
    ELLIPSIS = "ellipsis"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    DYNAMIC_SLICE = "dynamic_slice"
    @classmethod
    def from_index(cls, idx: Index) -> IndexType: ...

class ParsedIndex(NamedTuple):
    index: Index
    typ: IndexType
    consumed_axes: tuple[int, ...]

@dataclasses.dataclass(frozen=True, kw_only=True)
class NDIndexer:
    shape: tuple[int, ...]
    indices: list[ParsedIndex]
    @classmethod
    def from_raw_indices(
        cls,
        indices: Index | tuple[Index, ...],
        shape: tuple[int, ...],
    ) -> NDIndexer: ...
    def validate_static_indices(self, normalize_indices: bool = True) -> None: ...
    def validate_slices(self) -> None: ...
    @staticmethod
    def is_sharded(arr) -> bool: ...
    def has_partial_slices(self) -> bool: ...
    def expand_bool_indices(self) -> NDIndexer: ...
    def expand_scalar_bool_indices(
        self,
        sharding_spec: Any = None,
    ) -> tuple[NDIndexer, Any]: ...
    def convert_sequences_to_arrays(self) -> NDIndexer: ...
    def expand_ellipses(self) -> NDIndexer: ...
    def normalize_indices(self) -> NDIndexer: ...
    def to_static_slice(
        self,
        *,
        arr_is_sharded: bool = False,
        normalize_indices: bool = True,
        mode: str | slicing.GatherScatterMode | None,
    ) -> _StaticSliceIndexer: ...
    def to_dynamic_slice(
        self,
        *,
        arr_is_sharded: bool = False,
        normalize_indices: bool = True,
        mode: str | slicing.GatherScatterMode | None,
    ) -> _DynamicSliceIndexer: ...
    def is_advanced_int_indexer(self): ...
    def to_gather(
        self,
        x_sharding: NamedSharding | Any,
        normalize_indices: bool = True,
    ) -> _GatherIndexer: ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, aux_data, children): ...

@export
def take(
    a: ArrayLike,
    indices: ArrayLike,
    axis: int | None = None,
    out: None = None,
    mode: str | None = None,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    fill_value: StaticScalar | None = None,
) -> Array: ...
@export
def take_along_axis(
    arr: ArrayLike,
    indices: ArrayLike,
    axis: int | None = -1,
    mode: str | slicing.GatherScatterMode | None = None,
    fill_value: StaticScalar | None = None,
) -> Array: ...
@export
def put_along_axis(
    arr: ArrayLike,
    indices: ArrayLike,
    values: ArrayLike,
    axis: int | None,
    inplace: bool = True,
    *,
    mode: str | None = None,
) -> Array: ...

class IndexingStrategy(enum.Enum):
    AUTO = "auto"
    GATHER = "gather"
    SCATTER = "scatter"
    STATIC_SLICE = "static_slice"
    DYNAMIC_SLICE = "dynamic_slice"

def rewriting_take(
    arr: Array,
    idx: Index | tuple[Index, ...],
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | slicing.GatherScatterMode | None = None,
    fill_value: ArrayLike | None = None,
    normalize_indices: bool = True,
    out_sharding: NamedSharding | PartitionSpec | None = None,
    strategy: IndexingStrategy = ...,
) -> Array: ...

class _StaticSliceIndexer(NamedTuple):
    start_indices: Sequence[int]
    limit_indices: Sequence[int]
    strides: Sequence[int] | None
    rev_axes: Sequence[int]
    squeeze_axes: Sequence[int]
    newaxis_dims: Sequence[int]
    def is_trivial_slice(self, arr_shape: Sequence[int]): ...

class _DynamicSliceIndexer(NamedTuple):
    start_indices: Sequence[ArrayLike]
    slice_sizes: Sequence[int]
    rev_axes: Sequence[int]
    squeeze_axes: Sequence[int]
    newaxis_dims: Sequence[int]
    trivial_slicing: bool
    normalize_indices: bool

class _GatherIndexer(NamedTuple):
    slice_shape: Sequence[int]
    gather_slice_shape: Sequence[int]
    gather_indices: ArrayLike
    dnums: slicing.GatherDimensionNumbers
    unique_indices: bool
    indices_are_sorted: bool
    reversed_y_dims: Sequence[int]
    newaxis_dims: Sequence[int]
    scalar_bool_dims: Sequence[int]
    slice_sharding: NamedSharding | None = ...

def eliminate_deprecated_list_indexing(idx): ...
@export
def place(
    arr: ArrayLike,
    mask: ArrayLike,
    vals: ArrayLike,
    *,
    inplace: bool = True,
) -> Array: ...
@export
def put(
    a: ArrayLike,
    ind: ArrayLike,
    v: ArrayLike,
    mode: str | None = None,
    *,
    inplace: bool = True,
) -> Array: ...
