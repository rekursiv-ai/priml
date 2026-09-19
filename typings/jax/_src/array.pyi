from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

import enum
import functools

from _typeshed import Incomplete
from jax._src import (
    api as api,
    basearray as basearray,
    config as config,
    core as core,
    dispatch as dispatch,
    dtypes as dtypes,
    errors as errors,
    literals as literals,
    profiler as profiler,
    util as util,
    xla_bridge as xla_bridge,
)
from jax._src.interpreters import (
    mlir as mlir,
    pxla as pxla,
)
from jax._src.layout import (
    AutoLayout as AutoLayout,
    Format as Format,
    Layout as Layout,
)
from jax._src.mesh import empty_concrete_mesh as empty_concrete_mesh
from jax._src.op_shardings import are_hlo_shardings_equal as are_hlo_shardings_equal
from jax._src.sharding import Sharding as Sharding
from jax._src.sharding_impls import (
    NamedSharding as NamedSharding,
    PmapSharding as PmapSharding,
    SingleDeviceSharding as SingleDeviceSharding,
    device_replica_id_map as device_replica_id_map,
    hashed_index as hashed_index,
    local_to_global_shape as local_to_global_shape,
    num_addressable_indices as num_addressable_indices,
)
from jax._src.tree_util import (
    broadcast_prefix as broadcast_prefix,
    tree_flatten as tree_flatten,
    tree_unflatten as tree_unflatten,
)
from jax._src.typing import (
    ArrayLike as ArrayLike,
    DLDeviceType as DLDeviceType,
    DTypeLike as DTypeLike,
    ExtendedDType as ExtendedDType,
)
from jax._src.util import (
    cache as cache,
    safe_zip as safe_zip,
    unzip3 as unzip3,
    use_cpp_class as use_cpp_class,
    use_cpp_method as use_cpp_method,
)

zip: Incomplete
unsafe_zip: Incomplete
type Shape = tuple[int, ...]
Device: Incomplete
type Index = tuple[slice, ...]
type PRNGKeyArray = Any

class Shard:
    def __init__(
        self,
        device: Device,
        sharding: Sharding,
        global_shape: Shape,
        data: ArrayImpl | PRNGKeyArray | None = None,
    ) -> None: ...
    @functools.cached_property
    def index(self) -> Index: ...
    @functools.cached_property
    def replica_id(self) -> int: ...
    @property
    def device(self): ...
    @property
    def data(self): ...

class ArrayImpl(basearray.Array):
    aval: core.ShapedArray
    def __init__(
        self,
        aval: core.ShapedArray,
        sharding: Sharding,
        arrays: Sequence[ArrayImpl],
        committed: bool,
        _skip_checks: bool = False,
    ) -> None: ...
    @property
    def shape(self) -> Shape: ...
    @property
    def dtype(self): ...
    @property
    def ndim(self): ...
    @property
    def size(self): ...
    @property
    def sharding(self): ...
    @property
    def device(self): ...
    @property
    def weak_type(self): ...
    @property
    def committed(self) -> bool: ...
    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __float__(self) -> float: ...
    def __int__(self) -> int: ...
    def __complex__(self) -> complex: ...
    def __hex__(self): ...
    def __oct__(self): ...
    def __index__(self) -> int: ...
    def tobytes(self, order: str = "C"): ...
    def tolist(self): ...
    def __format__(self, format_spec) -> str: ...
    def __getitem__(self, idx): ...
    def __iter__(self): ...
    @property
    def is_fully_replicated(self) -> bool: ...
    @property
    def is_fully_addressable(self) -> bool: ...
    def __array__(self, dtype=None, context=None, copy=None): ...
    def __dlpack__(
        self,
        *,
        stream: int | Any | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[DLDeviceType, int] | None = None,
        copy: bool | None = None,
    ): ...
    def __dlpack_device__(self) -> tuple[enum.Enum, int]: ...
    def __reduce__(self): ...
    def unsafe_buffer_pointer(self): ...
    @property
    def __cuda_array_interface__(self): ...
    def on_device_size_in_bytes(self): ...
    def devices(self) -> set[Device]: ...
    @property
    def device_buffer(self) -> None: ...
    @property
    def device_buffers(self) -> None: ...
    def addressable_data(self, index: int) -> ArrayImpl: ...
    @functools.cached_property
    def addressable_shards(self) -> Sequence[Shard]: ...
    @property
    def format(self): ...
    @property
    def global_shards(self) -> Sequence[Shard]: ...
    def delete(self) -> None: ...
    def is_deleted(self): ...
    def block_until_ready(self): ...
    @profiler.annotate_function
    def copy_to_host_async(self) -> None: ...

def make_array_from_callback(
    shape: Shape,
    sharding: Sharding | Format,
    data_callback: Callable[[Index | None], ArrayLike],
    dtype: DTypeLike | None = None,
) -> ArrayImpl: ...
def make_array_from_process_local_data(sharding, local_data, global_shape=None): ...
def make_array_from_single_device_arrays(
    shape: Shape,
    sharding: Sharding,
    arrays: Sequence[basearray.Array],
    *,
    dtype: DTypeLike | None = None,
) -> ArrayImpl: ...
def as_slice_indices(
    arr: Any,
    idx: Index,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...
def shard_device_array(x, devices, indices, sharding): ...
def shard_sharded_device_array_slow_path(x, devices, indices, sharding): ...
