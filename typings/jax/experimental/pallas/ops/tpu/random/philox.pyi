from collections.abc import Sequence

from _typeshed import Incomplete
from jax import typing as typing
from jax._src import prng as prng
from jax.experimental.pallas.ops.tpu.random import prng_utils as prng_utils

import jax

type Shape = Sequence[int]
BLOCK_SIZE: Incomplete
K_HI_32: int
K_LO_32: int
MUL_A: int
MUL_B: int

def mul32_hi_lo(x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]: ...
def philox_4x32(hi0, lo0, hi1, lo1, k_hi, k_lo, rounds: int = 10): ...
def philox_4x32_kernel(
    key,
    shape: Shape,
    unpadded_shape: Shape,
    block_size: tuple[int, int],
    offset: typing.ArrayLike = 0,
    fuse_output: bool = True,
): ...
def philox_4x32_count(
    key,
    shape: Shape,
    offset: typing.ArrayLike = 0,
    fuse_output: bool = True,
): ...
def philox_split(key, shape: Shape): ...
def philox_random_bits(key, bit_width: int, shape: Shape): ...
def philox_fold_in(key, data): ...

plphilox_prng_impl: Incomplete
