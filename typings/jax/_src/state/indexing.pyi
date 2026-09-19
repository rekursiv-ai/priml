from typing import ClassVar

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    core as core,
    pretty_printer as pp,
    tree_util as tree_util,
)
from jax._src.indexing import (
    Slice as Slice,
    ds as ds,
    dslice as dslice,
)
from jax._src.state import types as state_types
from jax._src.typing import Array as Array
from jax._src.util import (
    merge_lists as merge_lists,
    partition_list as partition_list,
)

type IntIndexer = int | Array
type DimIndexer = IntIndexer | Slice

def unpack_ndindexer(
    indexer: NDIndexer,
) -> tuple[tuple[bool, ...], tuple[Slice, ...], tuple[IntIndexer, ...]]: ...

indexer_transform_type_registry: set[type]

@dataclasses.dataclass
class NDIndexer(state_types.Transform):
    indices: tuple[DimIndexer, ...]
    shape: tuple[int, ...]
    int_indexer_shape: tuple[int | Array, ...]
    validate: bool = ...
    def __post_init__(self) -> None: ...
    @property
    def is_dynamic_size(self): ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, data, flat_idx): ...
    @classmethod
    def from_indices_shape(cls, indices, shape) -> NDIndexer: ...
    @classmethod
    def make_trivial_indexer(cls, shape: tuple[int, ...]) -> NDIndexer: ...
    def get_indexer_shape(self) -> tuple[int | Array, ...]: ...
    def get_indexer_shape_static(self) -> tuple[int, ...]: ...
    def transform_type(self, x: core.AbstractValue): ...
    def undo(self, x: core.AbstractValue): ...
    def pretty_print(self, context: core.JaxprPpContext) -> pp.Doc: ...

class DShapedArray:
    shape: Incomplete
    dtype: Incomplete
    weak_type: Incomplete
    sharding: Incomplete
    vma: Incomplete
    memory_space: Incomplete
    def __init__(
        self,
        shape,
        dtype,
        weak_type: bool = False,
        *,
        sharding=None,
        vma: frozenset[core.AxisName] = ...,
        memory_space: core.MemorySpace = ...,
    ) -> None: ...
    def lower_val(self, val): ...
    def raise_val(self, val): ...
    def lo_ty(self): ...
    def update(self, shape=None, dtype=None, weak_type=None, **kwargs): ...
    ndim: Incomplete
    size: Incomplete
    broadcast: ClassVar[core.aval_method | None]
    transpose: ClassVar[core.aval_method | None]
    reshape: ClassVar[core.aval_method | None]
    def __eq__(self, other): ...
    def __hash__(self): ...
    def __ne__(self, other): ...
    def str_short(self, short_dtypes: bool = False, mesh_axis_types: bool = False): ...
    def update_vma(self, vma): ...
    def update_weak_type(self, weak_type): ...
