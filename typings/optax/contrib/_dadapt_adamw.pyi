from typing import NamedTuple

from optax._src import (
    base as base,
    numerics as numerics,
)

import chex

class DAdaptAdamWState(NamedTuple):
    exp_avg: base.Updates
    exp_avg_sq: base.Updates
    grad_sum: base.Updates
    estim_lr: chex.Array
    numerator_weighted: chex.Array
    count: chex.Array

def dadapt_adamw(
    learning_rate: base.ScalarOrSchedule = 1.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-08,
    estim_lr0: float = 1e-06,
    weight_decay: float = 0.0,
) -> base.GradientTransformationExtraArgs: ...
