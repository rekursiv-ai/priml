from collections.abc import Sequence

from flax.linen.module import (
    Module as Module,
    compact as compact,
    merge_param as merge_param,
)
from flax.typing import PRNGKey as PRNGKey

import jax

class Dropout(Module):
    rate: float
    broadcast_dims: Sequence[int] = ...
    deterministic: bool | None = ...
    rng_collection: str = ...
    @compact
    def __call__(
        self,
        inputs,
        deterministic: bool | None = None,
        rng: PRNGKey | None = None,
    ) -> jax.Array: ...
