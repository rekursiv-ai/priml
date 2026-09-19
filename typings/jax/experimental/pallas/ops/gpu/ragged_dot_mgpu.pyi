import dataclasses

from jax import (
    lax as lax,
    random as random,
)
from jax.experimental.mosaic.gpu import profiler as profiler

import jax

@dataclasses.dataclass(frozen=True)
class GroupInfo:
    group_id: jax.Array
    block: jax.Array
    block_start: jax.Array
    actual_start: jax.Array
    actual_end: jax.Array
    start_within_block: jax.Array
    actual_size: jax.Array
    @classmethod
    def create(cls, group_lengths, tile, tid): ...

def ragged_dot(
    lhs,
    rhs,
    *,
    group_sizes,
    block_m: int,
    block_n: int,
    block_k: int,
    max_concurrent_steps: int,
    grid_block_n: int,
    transpose_rhs: bool = False,
    load_group_sizes_to_register: bool = True,
) -> jax.Array: ...
def main(unused_argv): ...
