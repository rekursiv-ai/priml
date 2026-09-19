from _typeshed import Incomplete
from jax._src import (
    basearray as basearray,
    core as core,
    dtypes as dtypes,
    sharding_impls as sharding_impls,
    tree_util as tree_util,
)
from jax._src.interpreters import pxla as pxla
from jax._src.util import (
    safe_map as safe_map,
    safe_zip as safe_zip,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

class EArray(basearray.Array):
    __hash__: Incomplete
    __array_priority__: int
    aval: Incomplete
    def __init__(self, aval, data) -> None: ...
    def block_until_ready(self): ...
    def copy_to_host_async(self) -> None: ...
    def copy(self): ...
    def __iter__(self): ...
    shape: Incomplete
    dtype: Incomplete
    ndim: Incomplete
    size: Incomplete
    itemsize: Incomplete
    def __len__(self) -> int: ...
    devices: Incomplete
    is_fully_addressable: Incomplete
    is_fully_replicated: Incomplete
    delete: Incomplete
    is_deleted: Incomplete
    on_device_size_in_bytes: Incomplete
    unsafe_buffer_pointer: Incomplete
    @property
    def sharding(self): ...
    @property
    def committed(self): ...
    @property
    def device(self): ...
    def addressable_data(self, index: int) -> EArray: ...
    @property
    def addressable_shards(self) -> None: ...
    @property
    def global_shards(self) -> None: ...
