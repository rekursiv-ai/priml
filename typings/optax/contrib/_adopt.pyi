from collections.abc import Callable
from typing import Any

from optax._src import (
    base as base,
    combine as combine,
    numerics as numerics,
    transform as transform,
    utils as utils,
)

import chex
import jax.numpy as jnp

def scale_by_adopt(
    b1: float = 0.9,
    b2: float = 0.9999,
    eps: float = 1e-06,
    mu_dtype: chex.ArrayDType | None = None,
    *,
    nesterov: bool = False,
    use_clipping: bool = True,
    clip_value_fn: Callable[[jnp.ndarray], jnp.ndarray] = ...,
) -> base.GradientTransformation: ...
def adopt(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.9999,
    eps: float = 1e-06,
    mu_dtype: Any | None = None,
    *,
    nesterov: bool = False,
    use_clipping: bool = True,
    clip_value_fn: Callable[[jnp.ndarray], jnp.ndarray] = ...,
) -> base.GradientTransformationExtraArgs: ...
