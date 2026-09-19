from collections.abc import Sequence

from _typeshed import Incomplete
from jax._src import prng as prng
from jax.experimental.pallas.ops.tpu.random import prng_utils as prng_utils

type Shape = Sequence[int]
BLOCK_SIZE: Incomplete

def threefry_2x32_count(
    key,
    shape: Shape,
    unpadded_shape: Shape,
    block_size: tuple[int, int],
): ...
def plthreefry_random_bits(key, bit_width: int, shape: Shape): ...

plthreefry_prng_impl: Incomplete
