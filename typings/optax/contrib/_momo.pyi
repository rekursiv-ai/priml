from typing import NamedTuple

from optax._src import (
    base as base,
    numerics as numerics,
)

import chex

class MomoState(NamedTuple):
    exp_avg: base.Updates
    barf: chex.Array
    gamma: chex.Array
    lb: chex.Array
    count: chex.Array

def momo(
    learning_rate: base.ScalarOrSchedule = 1.0,
    beta: float = 0.9,
    lower_bound: float = 0.0,
    weight_decay: float = 0.0,
    adapt_lower_bound: bool = False,
) -> base.GradientTransformationExtraArgs: ...

class MomoAdamState(NamedTuple):
    exp_avg: base.Updates
    exp_avg_sq: base.Updates
    barf: chex.Array
    gamma: chex.Array
    lb: chex.Array
    count: chex.Array

def momo_adam(
    learning_rate: base.ScalarOrSchedule = 0.01,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-08,
    lower_bound: float = 0.0,
    weight_decay: float = 0.0,
    adapt_lower_bound: bool = False,
) -> base.GradientTransformationExtraArgs: ...
