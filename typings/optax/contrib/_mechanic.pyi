from typing import NamedTuple

from optax._src import (
    base as base,
    numerics as numerics,
)

import chex

class MechanicState(NamedTuple):
    base_optimizer_state: base.OptState
    count: chex.Array
    r: chex.Array
    m: chex.Array
    v: chex.Array
    s: chex.Array
    x0: base.Updates

def mechanize(
    base_optimizer: base.GradientTransformation | base.GradientTransformationExtraArgs,
    weight_decay: float = 0.01,
    eps: float = 1e-08,
    s_init: float = 1e-06,
    num_betas: int = 6,
) -> base.GradientTransformationExtraArgs: ...
