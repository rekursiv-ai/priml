from typing import NamedTuple

from optax._src import (
    base as base,
    clipping as clipping,
    combine as combine,
    transform as transform,
    utils as utils,
)

import jax

class DifferentiallyPrivateAggregateState(NamedTuple):
    rng_key: jax.Array

def differentially_private_aggregate(
    l2_norm_clip: float,
    noise_multiplier: float,
    key: jax.Array | int | None = None,
    *,
    seed: int | None = None,
) -> base.GradientTransformation: ...
def dpsgd(
    learning_rate: base.ScalarOrSchedule,
    l2_norm_clip: float,
    noise_multiplier: float,
    seed: int,
    momentum: float | None = None,
    nesterov: bool = False,
) -> base.GradientTransformation: ...
