from jax import (
    lax as lax,
    random as random,
)
from jax.experimental.mosaic.gpu import profiler as profiler

import jax

def transposed_ragged_dot(
    lhs,
    rhs,
    *,
    group_sizes,
    block_m: int,
    block_n: int,
    block_k: int,
    max_concurrent_steps: int,
    grid_block_n: int,
) -> jax.Array: ...
def ref_transposed_ragged_dot(lhs, rhs, group_sizes): ...
def main(unused_argv) -> None: ...
