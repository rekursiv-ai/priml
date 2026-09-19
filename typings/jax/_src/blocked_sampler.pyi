from collections.abc import Sequence
from typing import Any, Protocol

from _typeshed import Incomplete
from jax._src import random as random
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

type NdKeyList = Any
Shape: Incomplete

class SampleFn(Protocol):
    def __call__(self, key: ArrayLike, *args, shape: Shape, **kwargs) -> Array: ...

def blocked_fold_in(
    global_key: ArrayLike,
    total_size: Shape,
    block_size: Shape,
    tile_size: Shape,
    block_index: Sequence[ArrayLike],
) -> NdKeyList: ...
def sample_block(
    sampler_fn: SampleFn,
    keys: NdKeyList,
    block_size: Shape,
    tile_size: Shape,
    *args,
    **kwargs,
) -> Array: ...
