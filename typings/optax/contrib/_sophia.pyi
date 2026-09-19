from collections.abc import Callable
from typing import Any, NamedTuple

from optax._src import (
    base as base,
    combine as combine,
    numerics as numerics,
    transform as transform,
    utils as utils,
)

import jax

class HutchinsonState(NamedTuple):
    key: jax.Array

def hutchinson_estimator_diag_hessian(random_seed: jax.Array | None = None): ...

class SophiaState(NamedTuple):
    count: jax.Array
    mu: base.Updates
    nu: base.Updates
    hessian_fn_state: Any

def scale_by_sophia(
    b1: float = 0.965,
    b2: float = 0.99,
    eps: float = 1e-08,
    gamma: float = 0.01,
    clip_threshold: float | None = 1.0,
    update_interval: int = 10,
    hessian_diagonal_fn: base.GradientTransformation
    | base.GradientTransformationExtraArgs = ...,
    mu_dtype: Any | None = None,
    verbose: bool = False,
    print_win_rate_every_n_steps: int = 0,
) -> base.GradientTransformationExtraArgs: ...
def sophia(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.965,
    b2: float = 0.99,
    eps: float = 1e-08,
    weight_decay: float = 0.0001,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    gamma: float = 0.01,
    clip_threshold: float | None = 1.0,
    update_interval: int = 10,
    hessian_diagonal_fn: base.GradientTransformation
    | base.GradientTransformationExtraArgs = ...,
    mu_dtype: Any | None = None,
    verbose: bool = False,
    print_win_rate_every_n_steps: int = 0,
) -> base.GradientTransformationExtraArgs: ...
