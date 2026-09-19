from collections.abc import Sequence
from typing import Any

import dataclasses
import functools

from _typeshed import Incomplete
from jax._src import (
    mesh as mesh_lib,
    sharding as JSharding,
)
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir.dialects import sdy as sdy
from jax._src.mesh import AxisType as AxisType
from jax._src.partition_spec import PartitionSpec as PartitionSpec
from jax._src.util import (
    cache as cache,
    use_cpp_class as use_cpp_class,
    use_cpp_method as use_cpp_method,
)

type Shape = tuple[int, ...]
Device: Incomplete
type Index = tuple[slice, ...]
type XLADeviceAssignment = Sequence[Device]

class AUTO:
    mesh: Incomplete
    def __init__(self, mesh: mesh_lib.Mesh) -> None: ...

class UnspecifiedValue: ...

UNSPECIFIED: Incomplete
type MeshAxisName = Any
ArrayMapping: Incomplete
type ArrayMappingOrAutoOrUnspecified = ArrayMapping | AUTO | UnspecifiedValue

class NamedSharding(JSharding.Sharding):
    mesh: mesh_lib.Mesh | mesh_lib.AbstractMesh
    spec: PartitionSpec
    def __init__(
        self,
        mesh: mesh_lib.Mesh | mesh_lib.AbstractMesh,
        spec: PartitionSpec,
        *,
        memory_kind: str | None = None,
        _logical_device_ids=None,
    ) -> None: ...
    def __reduce__(self): ...
    @property
    def memory_kind(self) -> str | None: ...
    def __hash__(self): ...
    def __eq__(self, other): ...
    def check_compatible_aval(self, aval_shape: Shape) -> None: ...
    @property
    def num_devices(self) -> int: ...
    @property
    def device_set(self) -> set[Device]: ...
    @property
    def is_fully_addressable(self) -> bool: ...
    @property
    def addressable_devices(self) -> set[Device]: ...
    @functools.cached_property
    def is_fully_replicated(self) -> bool: ...
    @functools.cached_property
    def replicated_axes(self) -> frozenset[MeshAxisName]: ...
    def with_memory_kind(self, kind: str) -> NamedSharding: ...
    def update(self, **kwargs) -> NamedSharding: ...

def flatten_spec(spec): ...
def get_array_mapping(
    axis_resources: PartitionSpec | AUTO | UnspecifiedValue,
) -> ArrayMappingOrAutoOrUnspecified: ...

@dataclasses.dataclass
class SdyDim:
    axes: Sequence[str]
    is_open: bool
    def build(self) -> sdy.DimensionShardingAttr: ...

@dataclasses.dataclass(kw_only=True)
class SdyArray:
    mesh_shape: tuple[tuple[str, int], ...] | None
    dim_shardings: Sequence[SdyDim]
    logical_device_ids: tuple[int, ...] | None = ...
    replicated_axes: tuple[str, ...] = ...
    unreduced_axes: frozenset[str] = ...
    def build(self) -> sdy.TensorShardingAttr: ...

def modify_sdy_sharding_wrt_axis_types(sdy_sharding: SdyArray, mesh): ...
def named_sharding_to_xla_hlo_sharding(self, num_dimensions: int) -> xc.HloSharding: ...
def array_mapping_to_axis_resources(array_mapping: ArrayMapping): ...
def check_pspec(mesh, spec, _manual_axes=...) -> None: ...

class DuplicateSpecError(Exception):
    message: Incomplete
    mesh: Incomplete
    pspec: Incomplete
    def __init__(self, message, mesh, pspec) -> None: ...
