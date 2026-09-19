from collections.abc import Callable as Callable

import chex
import jax

def tree_split_key_like(
    rng_key: chex.PRNGKey,
    target_tree: chex.ArrayTree,
) -> chex.ArrayTree: ...
def tree_random_like(
    rng_key: chex.PRNGKey,
    target_tree: chex.ArrayTree,
    sampler: Callable[[chex.PRNGKey, chex.Shape, chex.ArrayDType], chex.Array]
    | Callable[
        [chex.PRNGKey, chex.Shape, chex.ArrayDType, jax.sharding.Sharding],
        chex.Array,
    ] = ...,
    dtype: chex.ArrayDType | None = None,
) -> chex.ArrayTree: ...
def tree_unwrap_random_key_data(input_tree: chex.ArrayTree) -> chex.ArrayTree: ...
