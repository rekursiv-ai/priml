from typing import NamedTuple

from optax._src import (
    base as base,
    numerics as numerics,
)

import chex

class ReduceLROnPlateauState(NamedTuple):
    scale: chex.Array
    best_value: chex.Array
    plateau_count: chex.Array
    cooldown_count: chex.Array
    count: chex.Array
    avg_value: chex.Array

def reduce_on_plateau(
    factor: float = 0.1,
    patience: int = 10,
    rtol: float = 0.0001,
    atol: float = 0.0,
    cooldown: int = 0,
    accumulation_size: int = 1,
    min_scale: float = 0.0,
) -> base.GradientTransformationExtraArgs: ...
