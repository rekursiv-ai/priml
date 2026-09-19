from collections.abc import Sequence

from _typeshed import Incomplete
from flax import nnx as nnx
from flax.nnx import rnglib as rnglib
from flax.nnx.module import (
    Module as Module,
    first_from as first_from,
)

import jax

class Dropout(Module):
    rate: Incomplete
    broadcast_dims: Incomplete
    deterministic: Incomplete
    rng_collection: Incomplete
    rngs: Incomplete
    def __init__(
        self,
        rate: float,
        *,
        broadcast_dims: Sequence[int] = (),
        deterministic: bool = False,
        rng_collection: str = "dropout",
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
    ) -> None: ...
    def __call__(
        self,
        inputs,
        *,
        deterministic: bool | None = None,
        rngs: rnglib.Rngs | rnglib.RngStream | jax.Array | None = None,
    ) -> jax.Array: ...
    def set_view(self, deterministic: bool | None = None): ...
