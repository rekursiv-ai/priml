from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import NamedTuple

import enum

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    api as api,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    source_info_util as source_info_util,
    util as util,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.lax import lax as lax
from jax._src.lax.utils import (
    input_dtype as input_dtype,
    standard_primitive as standard_primitive,
)
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import hlo as hlo
from jax._src.named_sharding import NamedSharding as NamedSharding
from jax._src.state.indexing import ds as ds
from jax._src.traceback_util import api_boundary as api_boundary
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    Shape as Shape,
)
from jax._src.util import (
    safe_map as safe_map,
    safe_zip as safe_zip,
)

import numpy as np

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

def slice(
    operand: ArrayLike,
    start_indices: Sequence[int],
    limit_indices: Sequence[int],
    strides: Sequence[int] | None = None,
) -> Array: ...
def dynamic_slice(
    operand: Array | np.ndarray,
    start_indices: Array | np.ndarray | Sequence[ArrayLike],
    slice_sizes: Shape,
    *,
    allow_negative_indices: bool | Sequence[bool] = True,
) -> Array: ...
def dynamic_update_slice(
    operand: Array | np.ndarray,
    update: ArrayLike,
    start_indices: Array | Sequence[ArrayLike],
    *,
    allow_negative_indices: bool | Sequence[bool] = True,
) -> Array: ...

class GatherDimensionNumbers(NamedTuple):
    offset_dims: tuple[int, ...]
    collapsed_slice_dims: tuple[int, ...]
    start_index_map: tuple[int, ...]
    operand_batching_dims: tuple[int, ...] = ...
    start_indices_batching_dims: tuple[int, ...] = ...

class GatherScatterMode(enum.Enum):
    CLIP = ...
    FILL_OR_DROP = ...
    PROMISE_IN_BOUNDS = ...
    ONE_HOT = ...
    @staticmethod
    def from_any(s: str | GatherScatterMode | None) -> GatherScatterMode: ...

def gather(
    operand: ArrayLike,
    start_indices: ArrayLike,
    dimension_numbers: GatherDimensionNumbers,
    slice_sizes: Shape,
    *,
    unique_indices: bool = False,
    indices_are_sorted: bool = False,
    mode: str | GatherScatterMode | None = None,
    fill_value=None,
) -> Array: ...

class ScatterDimensionNumbers(NamedTuple):
    update_window_dims: Sequence[int]
    inserted_window_dims: Sequence[int]
    scatter_dims_to_operand_dims: Sequence[int]
    operand_batching_dims: Sequence[int] = ...
    scatter_indices_batching_dims: Sequence[int] = ...

def scatter_add(
    operand: ArrayLike,
    scatter_indices: ArrayLike,
    updates: ArrayLike,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array: ...
def scatter_sub(
    operand: ArrayLike,
    scatter_indices: ArrayLike,
    updates: ArrayLike,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array: ...
def scatter_mul(
    operand: ArrayLike,
    scatter_indices: ArrayLike,
    updates: ArrayLike,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array: ...
def scatter_min(
    operand: ArrayLike,
    scatter_indices: ArrayLike,
    updates: ArrayLike,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array: ...
def scatter_max(
    operand: ArrayLike,
    scatter_indices: ArrayLike,
    updates: ArrayLike,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array: ...
def scatter_apply(
    operand: Array,
    scatter_indices: Array,
    func: Callable[[Array], Array],
    dimension_numbers: ScatterDimensionNumbers,
    *,
    update_shape: Shape = (),
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array: ...
def scatter(
    operand: ArrayLike,
    scatter_indices: ArrayLike,
    updates: ArrayLike,
    dimension_numbers: ScatterDimensionNumbers,
    *,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    mode: str | GatherScatterMode | None = None,
) -> Array: ...
def index_take(src: Array, idxs: Array, axes: Sequence[int]) -> Array: ...
def slice_in_dim(
    operand: Array | np.ndarray,
    start_index: int | None,
    limit_index: int | None,
    stride: int = 1,
    axis: int = 0,
) -> Array: ...
def index_in_dim(
    operand: Array | np.ndarray,
    index: int,
    axis: int = 0,
    keepdims: bool = True,
) -> Array: ...
def dynamic_slice_in_dim(
    operand: Array | np.ndarray,
    start_index: ArrayLike,
    slice_size: int,
    axis: int = 0,
    *,
    allow_negative_indices: bool = True,
) -> Array: ...
def dynamic_index_in_dim(
    operand: Array | np.ndarray,
    index: ArrayLike,
    axis: int = 0,
    keepdims: bool = True,
    *,
    allow_negative_indices: bool = True,
) -> Array: ...
def dynamic_update_slice_in_dim(
    operand: Array | np.ndarray,
    update: ArrayLike,
    start_index: ArrayLike,
    axis: int,
    *,
    allow_negative_indices: bool = True,
) -> Array: ...
def dynamic_update_index_in_dim(
    operand: Array | np.ndarray,
    update: ArrayLike,
    index: ArrayLike,
    axis: int,
    *,
    allow_negative_indices: bool = True,
) -> Array: ...

slice_p: Incomplete
dynamic_slice_p: Incomplete
dynamic_update_slice_p: Incomplete
gather_p: Incomplete
scatter_add_p: Incomplete
scatter_sub_p: Incomplete
scatter_mul_p: Incomplete
scatter_min_p: Incomplete
scatter_max_p: Incomplete
scatter_p: Incomplete
