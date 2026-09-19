from typing import Literal

from jax import lax as lax
from jax.experimental import multihost_utils as multihost_utils
from jax.experimental.mosaic.gpu import profiler as profiler
from jax.extend import backend as backend

import jax

def reduce_scatter(
    x: jax.Array,
    *,
    axis_name,
    scatter_dimension: int | None = 0,
    reduction: Literal["add", "min", "max", "and", "or", "xor"] = "add",
    num_blocks: int | None = None,
    tile_size: int | None = None,
    vec_size: int | None = None,
) -> jax.Array: ...
