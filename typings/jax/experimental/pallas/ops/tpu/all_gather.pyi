from collections.abc import Sequence

from _typeshed import Incomplete
from jax import lax as lax
from jax._src import shard_map as shard_map
from jax.experimental.pallas import tpu as pltpu

import jax

P: Incomplete

def get_neighbor(
    idx: jax.Array,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    *,
    direction: str,
) -> tuple[jax.Array, ...]: ...
def ag_kernel(
    x_ref,
    o_ref,
    send_sem,
    recv_sem,
    *,
    axis_name: str,
    mesh: jax.sharding.Mesh,
): ...
def all_gather(
    x,
    *,
    mesh: jax.sharding.Mesh,
    axis_name: str | Sequence[str],
    memory_space: pltpu.MemorySpace = ...,
): ...
