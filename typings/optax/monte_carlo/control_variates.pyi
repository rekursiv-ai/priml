from collections.abc import Callable, Sequence
from typing import Any

from _typeshed import Incomplete
from optax._src import base as base

import chex
import jax.numpy as jnp

type CvState = Any
ComputeCv: Incomplete
CvExpectedValue: Incomplete
UpdateCvState: Incomplete
type ControlVariate = tuple[ComputeCv, CvExpectedValue, UpdateCvState]

@chex.warn_deprecated_function
def control_delta_method(function: Callable[[chex.Array], float]) -> ControlVariate: ...
@chex.warn_deprecated_function
def moving_avg_baseline(
    function: Callable[[chex.Array], float],
    decay: float = 0.99,
    zero_debias: bool = True,
    use_decay_early_training_heuristic: bool = True,
) -> ControlVariate: ...
@chex.warn_deprecated_function
def control_variates_jacobians(
    function: Callable[[chex.Array], float],
    control_variate_from_function: Callable[
        [Callable[[chex.Array], float]],
        ControlVariate,
    ],
    grad_estimator: Callable[..., jnp.ndarray],
    params: base.Params,
    dist_builder: Callable[..., Any],
    rng: chex.PRNGKey,
    num_samples: int,
    control_variate_state: CvState = None,
    estimate_cv_coeffs: bool = False,
    estimate_cv_coeffs_num_samples: int = 20,
) -> tuple[Sequence[chex.Array], CvState]: ...
@chex.warn_deprecated_function
def estimate_control_variate_coefficients(
    function: Callable[[chex.Array], float],
    control_variate_from_function: Callable[
        [Callable[[chex.Array], float]],
        ControlVariate,
    ],
    grad_estimator: Callable[..., jnp.ndarray],
    params: base.Params,
    dist_builder: Callable[..., Any],
    rng: chex.PRNGKey,
    num_samples: int,
    control_variate_state: CvState = None,
    eps: float = 0.001,
) -> Sequence[float]: ...
