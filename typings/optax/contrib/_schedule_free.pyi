from typing import NamedTuple

from optax._src import (
    alias as alias,
    base as base,
    combine as combine,
    numerics as numerics,
    transform as transform,
)

import chex
import jax

class ScheduleFreeState(NamedTuple):
    b1: chex.Array
    weight_sum: chex.Array
    step_count: chex.Array
    max_lr: chex.Array
    base_optimizer_state: base.OptState
    z: base.Params

def schedule_free_eval_params(state: base.OptState, params: base.Params): ...
def schedule_free(
    base_optimizer: base.GradientTransformation,
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    weight_lr_power: float = 2.0,
    state_dtype: jax.typing.DTypeLike | None = None,
) -> base.GradientTransformationExtraArgs: ...
def schedule_free_sgd(
    learning_rate: float = 1.0,
    warmup_steps: int | None = None,
    b1: float = 0.9,
    weight_decay: float | None = None,
    weight_lr_power: float = 2.0,
    state_dtype: jax.typing.DTypeLike | None = None,
) -> base.GradientTransformationExtraArgs: ...
def schedule_free_adamw(
    learning_rate: float = 0.0025,
    warmup_steps: int | None = None,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-08,
    weight_decay: float = 0.0,
    weight_lr_power: float = 2.0,
    state_dtype: jax.typing.DTypeLike | None = None,
) -> base.GradientTransformationExtraArgs: ...
