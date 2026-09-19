from collections.abc import Callable as Callable
from typing import Any, NamedTuple

from optax._src import (
    base as base,
    numerics as numerics,
    utils as utils,
    wrappers as wrappers,
)

import chex
import jax

class WeightDecaySchedule(NamedTuple):
    count: chex.Array

def add_decayed_weights(
    weight_decay: float | jax.Array | base.ScalarOrSchedule = 0.0,
    mask: Any | Callable[[base.Params], Any] | None = None,
) -> base.GradientTransformation: ...

class AddNoiseState(NamedTuple):
    count: jax.Array
    rng_key: jax.Array

def add_noise(
    eta: float,
    gamma: float,
    key: jax.Array | int | None = None,
    *,
    seed: int | None = None,
) -> base.GradientTransformation: ...
