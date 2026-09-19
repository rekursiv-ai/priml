from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

from optax._src import (
    base as base,
    utils as utils,
)

import chex

@chex.warn_deprecated_function
def score_function_jacobians(
    function: Callable[[chex.Array], float],
    params: base.Params,
    dist_builder: Callable[..., Any],
    rng: chex.PRNGKey,
    num_samples: int,
) -> Sequence[chex.Array]: ...
@chex.warn_deprecated_function
def pathwise_jacobians(
    function: Callable[[chex.Array], float],
    params: base.Params,
    dist_builder: Callable[..., Any],
    rng: chex.PRNGKey,
    num_samples: int,
) -> Sequence[chex.Array]: ...
@chex.warn_deprecated_function
def measure_valued_jacobians(
    function: Callable[[chex.Array], float],
    params: base.Params,
    dist_builder: Callable[..., Any],
    rng: chex.PRNGKey,
    num_samples: int,
    coupling: bool = True,
) -> Sequence[chex.Array]: ...
@chex.warn_deprecated_function
def measure_valued_estimation_mean(
    function: Callable[[chex.Array], float],
    dist: Any,
    rng: chex.PRNGKey,
    num_samples: int,
    coupling: bool = True,
) -> chex.Array: ...
@chex.warn_deprecated_function
def measure_valued_estimation_std(
    function: Callable[[chex.Array], float],
    dist: Any,
    rng: chex.PRNGKey,
    num_samples: int,
    coupling: bool = True,
) -> chex.Array: ...
