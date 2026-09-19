from typing import NamedTuple

from optax._src import (
    base as base,
    numerics as numerics,
)

import chex

class ProdigyState(NamedTuple):
    exp_avg: base.Updates
    exp_avg_sq: base.Updates
    grad_sum: base.Updates
    params0: base.Updates
    estim_lr: chex.Array
    numerator_weighted: chex.Array
    count: chex.Array

def prodigy(
    learning_rate: base.ScalarOrSchedule = 1.0,
    betas: tuple[float, float] = (0.9, 0.999),
    beta3: float | None = None,
    eps: float = 1e-08,
    estim_lr0: float = 1e-06,
    estim_lr_coef: float = 1.0,
    weight_decay: float = 0.0,
    safeguard_warmup: bool = False,
) -> base.GradientTransformationExtraArgs: ...
