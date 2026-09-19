from _typeshed import Incomplete
from jax._src import (
    blocked_sampler as blocked_sampler,
    dtypes as dtypes,
    prng as jax_prng,
    typing as typing,
)
from jax._src.pallas import primitives as primitives
from jax._src.pallas.mosaic.primitives import (
    prng_random_bits as prng_random_bits,
    prng_seed as prng_seed,
)

import jax

Shape: Incomplete
SampleFnType: Incomplete
KeylessSampleFnType: Incomplete
set_seed = prng_seed
unwrap_pallas_seed: Incomplete
wrap_pallas_seed: Incomplete

def to_pallas_key(key: jax.Array) -> jax.Array: ...
def is_pallas_impl(impl: jax_prng.PRNGImpl) -> bool: ...

tpu_key_impl: Incomplete
tpu_internal_stateful_impl: Incomplete
stateful_bits: Incomplete
stateful_uniform: Incomplete
stateful_bernoulli: Incomplete
stateful_normal: Incomplete

def sample_block(
    sampler_fn: SampleFnType,
    global_key: jax.Array,
    block_size: Shape,
    tile_size: Shape,
    total_size: Shape,
    block_index: tuple[typing.ArrayLike, ...] | None = None,
    **kwargs,
) -> jax.Array: ...
