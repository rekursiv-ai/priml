from collections.abc import Sequence

from _typeshed import Incomplete
from jax._src import util as util
from jax._src.lib import pmap_lib as pmap_lib

unsafe_map: Incomplete
map: Incomplete
NoSharding: Incomplete
Chunked: Incomplete
Unstacked: Incomplete
ShardedAxis: Incomplete
Replicated: Incomplete
type MeshDimAssignment = ShardedAxis | Replicated
ShardingSpec: Incomplete
type Index = int | slice | tuple[int | slice, ...]

def spec_to_indices(shape: Sequence[int], spec: ShardingSpec) -> tuple[Index, ...]: ...
def pmap_sharding_spec(
    nrep,
    axis_size,
    sharded_shape: Sequence[int],
    map_axis: int | None,
) -> ShardingSpec: ...
def create_pmap_sharding_spec(
    shape: tuple[int, ...],
    sharded_dim: int = 0,
    sharded_dim_size: int | None = None,
): ...
