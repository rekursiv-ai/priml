from collections.abc import (
    Callable as Callable,
    Hashable,
)

from jax._src import (
    prng as prng,
    random as random,
)
from jax._src.typing import Array as Array

type Shape = tuple[int, ...]

def define_prng_impl(
    *,
    key_shape: Shape,
    seed: Callable[[Array], Array],
    split: Callable[[Array, Shape], Array],
    random_bits: Callable[[Array, int, Shape], Array],
    fold_in: Callable[[Array, int], Array],
    name: str = "<unnamed>",
    tag: str = "?",
) -> Hashable: ...
