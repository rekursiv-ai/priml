from collections.abc import Hashable

from jax import lax as lax
from jax.experimental import multihost_utils as multihost_utils
from jax.experimental.mosaic.gpu import profiler as profiler
from jax.extend import backend as backend

import jax

def all_gather(
    x: jax.Array,
    *,
    axis_name: Hashable,
    gather_dimension: int = 0,
    num_blocks: int | None = None,
    tile_size: int | None = None,
    vec_size: int | None = None,
) -> jax.Array: ...
