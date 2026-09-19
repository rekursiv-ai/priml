from collections.abc import Callable as Callable

from jax._src import (
    config as config,
    core as core,
    dtypes as dtypes,
    tree_util as tree_util,
    util as util,
)
from jax._src.lax import (
    lax as lax,
    slicing as slicing,
)
from jax._src.numpy import (
    indexing as indexing,
    reductions as reductions,
)
from jax._src.numpy.util import (
    check_arraylike as check_arraylike,
    promote_dtypes as promote_dtypes,
)
from jax._src.pjit import auto_axes as auto_axes
from jax._src.sharding_impls import NamedSharding as NamedSharding
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    Index as Index,
)

def segment_sum(
    data: ArrayLike,
    segment_ids: ArrayLike,
    num_segments: int | None = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    bucket_size: int | None = None,
    mode: slicing.GatherScatterMode | str | None = None,
) -> Array: ...
def segment_prod(
    data: ArrayLike,
    segment_ids: ArrayLike,
    num_segments: int | None = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    bucket_size: int | None = None,
    mode: slicing.GatherScatterMode | str | None = None,
) -> Array: ...
def segment_max(
    data: ArrayLike,
    segment_ids: ArrayLike,
    num_segments: int | None = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    bucket_size: int | None = None,
    mode: slicing.GatherScatterMode | str | None = None,
) -> Array: ...
def segment_min(
    data: ArrayLike,
    segment_ids: ArrayLike,
    num_segments: int | None = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    bucket_size: int | None = None,
    mode: slicing.GatherScatterMode | str | None = None,
) -> Array: ...
