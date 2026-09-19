from collections.abc import Mapping, Sequence

import functools

from _typeshed import Incomplete
from jax._src.op_shardings import (
    are_hlo_shardings_equal as are_hlo_shardings_equal,
    get_num_ways_dim_sharded as get_num_ways_dim_sharded,
    is_hlo_sharding_replicated as is_hlo_sharding_replicated,
    op_sharding_to_indices as op_sharding_to_indices,
)
from jax._src.util import (
    cache as cache,
    safe_zip as safe_zip,
    use_cpp_class as use_cpp_class,
)

type Shape = tuple[int, ...]
Device: Incomplete
type Index = tuple[slice, ...]
type XLADeviceAssignment = Sequence[Device]

class IndivisibleError(ValueError): ...

def common_devices_indices_map(
    s: Sharding,
    global_shape: Shape,
) -> Mapping[Device, Index]: ...

class Sharding:
    @property
    def device_set(self) -> set[Device]: ...
    @property
    def is_fully_replicated(self) -> bool: ...
    @property
    def is_fully_addressable(self) -> bool: ...
    @property
    def num_devices(self) -> int: ...
    @property
    def memory_kind(self) -> str | None: ...
    def with_memory_kind(self, kind: str) -> Sharding: ...
    @functools.cached_property
    def addressable_devices(self) -> set[Device]: ...
    def addressable_devices_indices_map(
        self,
        global_shape: Shape,
    ) -> Mapping[Device, Index | None]: ...
    def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]: ...
    @property
    def has_addressable_devices(self) -> bool: ...
    def shard_shape(self, global_shape: Shape) -> Shape: ...
    def is_equivalent_to(self, other: Sharding, ndim: int) -> bool: ...
