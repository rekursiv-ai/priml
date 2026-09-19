from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import dataclasses
import functools
import types

from _typeshed import Incomplete
from jax._src import (
    config as config,
    core as core,
    mesh as mesh_lib,
    mesh_utils as mesh_utils,
    sharding as jsharding,
    sharding_specs as sharding_specs,
    source_info_util as source_info_util,
    tree_util as tree_util,
    util as util,
)
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir.dialects import sdy as sdy
from jax._src.named_sharding import (
    AUTO as AUTO,
    UNSPECIFIED as UNSPECIFIED,
    ArrayMapping as ArrayMapping,
    ArrayMappingOrAutoOrUnspecified as ArrayMappingOrAutoOrUnspecified,
    NamedSharding as NamedSharding,
    SdyArray as SdyArray,
    SdyDim as SdyDim,
    UnspecifiedValue as UnspecifiedValue,
    array_mapping_to_axis_resources as array_mapping_to_axis_resources,
    flatten_spec as flatten_spec,
    get_array_mapping as get_array_mapping,
    modify_sdy_sharding_wrt_axis_types as modify_sdy_sharding_wrt_axis_types,
    named_sharding_to_xla_hlo_sharding as named_sharding_to_xla_hlo_sharding,
)
from jax._src.op_shardings import (
    are_hlo_shardings_equal as are_hlo_shardings_equal,
    get_num_ways_dim_sharded as get_num_ways_dim_sharded,
    is_hlo_sharding_replicated as is_hlo_sharding_replicated,
)
from jax._src.partition_spec import PartitionSpec as PartitionSpec
from jax._src.util import (
    safe_zip as safe_zip,
    use_cpp_class as use_cpp_class,
    use_cpp_method as use_cpp_method,
)

import numpy as np

config_ext: Incomplete
type Shape = tuple[int, ...]
Device: Incomplete
type Index = tuple[slice, ...]
type XLADeviceAssignment = tuple[Device, ...]
XLACompatibleSharding: Incomplete

def hashed_index(x) -> int: ...
def device_replica_id_map(sharding, global_shape: Shape) -> Mapping[Device, int]: ...

@dataclasses.dataclass
class SdyArrayList:
    shardings: Sequence[SdyArray]
    def build(self) -> sdy.TensorShardingPerValueAttr: ...

replicated_hlo_sharding: Incomplete

class SingleDeviceSharding(jsharding.Sharding):
    def __init__(self, device: Device, *, memory_kind: str | None = None) -> None: ...
    def __reduce__(self): ...
    def __hash__(self): ...
    def __eq__(self, other): ...
    @property
    def num_devices(self) -> int: ...
    @property
    def device_set(self) -> set[Device]: ...
    @property
    def memory_kind(self) -> str | None: ...
    def with_memory_kind(self, kind: str) -> SingleDeviceSharding: ...
    def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]: ...
    @property
    def is_fully_replicated(self) -> bool: ...
    @property
    def is_fully_addressable(self) -> bool: ...
    def check_compatible_aval(self, aval_shape: Shape) -> None: ...

def pmap_sharding_devices_indices_map(
    self,
    global_shape: Shape,
) -> Mapping[Device, Index]: ...

class PmapSharding(jsharding.Sharding):
    devices: np.ndarray
    sharding_spec: sharding_specs.ShardingSpec
    def __init__(
        self,
        devices: Sequence[Device] | np.ndarray,
        sharding_spec: sharding_specs.ShardingSpec,
    ) -> None: ...
    def __reduce__(self): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def is_equivalent_to(self, other: PmapSharding, ndim: int) -> bool: ...
    @classmethod
    def default(
        cls,
        shape: Shape,
        sharded_dim: int | None = 0,
        devices: Sequence[xc.Device] | None = None,
    ) -> PmapSharding: ...
    @property
    def num_devices(self) -> int: ...
    @functools.cached_property
    def device_set(self) -> set[Device]: ...
    def devices_indices_map(self, global_shape: Shape) -> Mapping[Device, Index]: ...
    @property
    def memory_kind(self) -> str | None: ...
    def with_memory_kind(self, kind: str): ...
    @functools.cached_property
    def is_fully_replicated(self) -> bool: ...
    @functools.cached_property
    def is_fully_addressable(self) -> bool: ...
    def check_compatible_aval(self, aval_shape: Shape) -> None: ...
    def shard_shape(self, global_shape: Shape) -> Shape: ...

class GSPMDSharding(jsharding.Sharding):
    def __init__(
        self,
        devices: Sequence[Device] | xc.DeviceList,
        op_sharding: xc.OpSharding | xc.HloSharding,
        *,
        memory_kind: str | None = None,
    ) -> None: ...
    def __reduce__(self): ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def check_compatible_aval(self, aval_shape: Shape) -> None: ...
    @property
    def num_devices(self) -> int: ...
    @functools.cached_property
    def device_set(self) -> set[Device]: ...
    @property
    def memory_kind(self) -> str | None: ...
    def with_memory_kind(self, kind: str) -> GSPMDSharding: ...
    @functools.cached_property
    def is_fully_replicated(self) -> bool: ...
    @functools.cached_property
    def is_fully_addressable(self) -> bool: ...
    @classmethod
    def get_replicated(cls, device_assignment, *, memory_kind: str | None = None): ...

type MeshAxisName = Any

def prepare_axis_resources(
    axis_resources,
    arg_name,
    allow_unconstrained_dims: bool = False,
): ...

class AxisEnv(NamedTuple):
    nreps: int
    names: tuple[Any, ...]
    sizes: tuple[int, ...]

@dataclasses.dataclass(frozen=True)
class SPMDAxisContext:
    mesh: mesh_lib.Mesh
    manual_axes: frozenset[MeshAxisName] = ...
    @property
    def axis_env(self): ...
    @property
    def unsafe_axis_env(self): ...
    def extend_manual(self, axes: frozenset[MeshAxisName]) -> SPMDAxisContext: ...

@dataclasses.dataclass(frozen=True)
class ReplicaAxisContext:
    axis_env: AxisEnv

@dataclasses.dataclass(frozen=True)
class ShardingContext:
    num_devices: int
    device_assignment: tuple[xc.Device, ...] | None = ...
    abstract_mesh: mesh_lib.AbstractMesh | None = ...
    def __post_init__(self) -> None: ...
    @property
    def axis_env(self): ...

def strides_for_sizes(sizes): ...
def unflatten_array(named_sizes, assignment): ...
def unflatten_superdims(assignment): ...
def explode_superdims(sizes, dims): ...
def parse_flatten_op_sharding(
    hlo_sharding: xc.OpSharding | xc.HloSharding,
    mesh: mesh_lib.Mesh | mesh_lib.AbstractMesh,
) -> Sequence[PartitionSpec]: ...

class NonUniformShardingError(ValueError): ...

def get_process_index_and_count(
    tensor_sharding: jsharding.Sharding,
    dim: int,
    ndims: int,
) -> tuple[int, int]: ...
def local_to_global_shape(
    sharding: jsharding.Sharding,
    local_shape: Shape,
) -> tuple[int | None, ...]: ...
def num_addressable_indices(
    tensor_sharding: jsharding.Sharding,
    dim: int,
    global_shape: Shape,
) -> int: ...
def physical_hlo_sharding(aval, hlo_sharding: xc.HloSharding) -> xc.HloSharding: ...
def is_single_device_sharding(sharding: jsharding.Sharding) -> bool: ...
def make_key_array_phys_sharding(aval, sharding): ...
def physical_sharding(aval, sharding: jsharding.Sharding) -> jsharding.Sharding: ...
def get_logical_gspmd_sharding(logical_shape, dtype, phys_sharding): ...
def check_replicated_trailing_dims(
    sharding: jsharding.Sharding,
    logical_shape,
    dtype,
): ...
def logical_sharding(logical_shape, dtype, phys_sharding) -> jsharding.Sharding: ...
def cached_named_sharding(
    mesh: mesh_lib.Mesh | mesh_lib.AbstractMesh,
    pspec: PartitionSpec,
    memory_kind: str | None = None,
) -> NamedSharding: ...
def canonicalize_sharding(
    sharding: NamedSharding | PartitionSpec | None,
    api_name: str,
    check_mesh_consistency: bool = True,
) -> NamedSharding | None: ...
def make_mesh(
    axis_shapes: Sequence[int],
    axis_names: Sequence[str],
    axis_types: tuple[mesh_lib.AxisType, ...] | None = None,
    *,
    devices: Sequence[xc.Device] | None = None,
) -> mesh_lib.Mesh: ...

class set_mesh:
    prev_abstract_mesh: Incomplete
    prev_mesh: Incomplete
    def __init__(self, mesh: mesh_lib.Mesh) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

def get_mesh() -> mesh_lib.Mesh: ...
