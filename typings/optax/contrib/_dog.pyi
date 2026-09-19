from collections.abc import Callable as Callable
from typing import Any, Literal, NamedTuple

from optax._src import (
    base as base,
    combine as combine,
    transform as transform,
)

import chex
import jax

class DoGState(NamedTuple):
    is_init_step: jax.Array
    init_params: chex.ArrayTree
    max_dist: jax.Array
    sum_sq_norm_grads: jax.Array

def scale_by_dog(
    init_step: tuple[Literal["distance", "learning_rate", "heuristic"], float],
    eps: float = 1e-08,
) -> base.GradientTransformation: ...
def dog(
    learning_rate: base.ScalarOrSchedule = 1.0,
    init_step: tuple[Literal["distance", "learning_rate", "heuristic"], float] = (
        "heuristic",
        1e-06,
    ),
    eps: float = 1e-08,
    weight_decay: float | None = None,
    mask: Any | Callable[[base.Params], Any] | None = None,
): ...

class DoWGState(NamedTuple):
    init_params: chex.ArrayTree
    weighted_sq_norm_grads: jax.Array
    estim_sq_dist: jax.Array

def scale_by_dowg(
    init_estim_sq_dist: float | None = None,
    eps: float = 0.0001,
) -> base.GradientTransformation: ...
def dowg(
    learning_rate: base.ScalarOrSchedule = 1.0,
    init_estim_sq_dist: float | None = None,
    eps: float = 0.0001,
    weight_decay: float | None = None,
    mask: Any | Callable[[base.Params], Any] | None = None,
): ...
