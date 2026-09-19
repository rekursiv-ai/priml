from collections.abc import Callable
from typing import Any, NamedTuple

from _typeshed import Incomplete
from jax._src import (
    api as api,
    array as array,
    config as config,
    core as core,
    dtypes as dtypes,
    prng as prng,
    sharding_impls as sharding_impls,
    stages as stages,
    traceback_util as traceback_util,
    util as util,
)
from jax._src.api_util import (
    argnums_partial as argnums_partial,
    donation_vector as donation_vector,
    fun_signature as fun_signature,
    fun_sourceinfo as fun_sourceinfo,
)
from jax._src.interpreters import pxla as pxla
from jax._src.lax import lax as lax
from jax._src.lib import (
    jaxlib_extension_version as jaxlib_extension_version,
    xla_client as xc,
)
from jax._src.mesh import Mesh as Mesh
from jax._src.shard_map import shard_map as shard_map
from jax._src.tree_util import (
    broadcast_flattened_prefix_with_treedef as broadcast_flattened_prefix_with_treedef,
    broadcast_prefix as broadcast_prefix,
    prefix_errors as prefix_errors,
    tree_flatten as tree_flatten,
    tree_map as tree_map,
    tree_unflatten as tree_unflatten,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

def pmap(
    f,
    axis_name=None,
    *,
    in_axes: int = 0,
    out_axes: int = 0,
    static_broadcasted_argnums=(),
    devices=None,
    backend=None,
    axis_size=None,
    donate_argnums=(),
): ...

class CachedShardMap(NamedTuple):
    pmapped: Callable[..., Any]
    in_specs_flat: tuple[sharding_impls.PartitionSpec, ...]
    local_devices: list[xc.Device]
    in_local_shardings: list[sharding_impls.NamedSharding]
    in_global_shardings: list[sharding_impls.NamedSharding]
    mesh: Mesh
    out_specs: Any
    out_local_shardings_thunk: Callable[
        [sharding_impls.PartitionSpec],
        tuple[sharding_impls.NamedSharding, sharding_impls.NamedSharding],
    ]
    donate_argnums: list[int]
    out_global_shardings: Any
    jitted_f: Any
    jitted_f_with_shardings: Any

def host_local_array_to_global_array(
    dyn_args_flat,
    cached,
    trace_state_clean,
    donated_invars,
): ...
def global_array_to_host_local_array(out, cached, trace_state_clean): ...
