from collections.abc import Iterable

from optax._src import base as base

import chex

def constant_schedule(value: float) -> base.Schedule: ...
def polynomial_schedule(
    init_value: chex.Scalar,
    end_value: chex.Scalar,
    power: chex.Scalar,
    transition_steps: int,
    transition_begin: int = 0,
) -> base.Schedule: ...
def linear_schedule(
    init_value: chex.Scalar,
    end_value: chex.Scalar,
    transition_steps: int,
    transition_begin: int = 0,
) -> base.Schedule: ...
def piecewise_constant_schedule(
    init_value: float,
    boundaries_and_scales: dict[int, float] | None = None,
) -> base.Schedule: ...
def exponential_decay(
    init_value: float,
    transition_steps: int,
    decay_rate: float,
    transition_begin: int = 0,
    staircase: bool = False,
    end_value: float | None = None,
) -> base.Schedule: ...
def cosine_decay_schedule(
    init_value: float,
    decay_steps: int,
    alpha: float = 0.0,
    exponent: float = 1.0,
) -> base.Schedule: ...
def piecewise_interpolate_schedule(
    interpolate_type: str,
    init_value: float,
    boundaries_and_scales: dict[int, float] | None = None,
) -> base.Schedule: ...
def linear_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    pct_final: float = 0.85,
    div_factor: float = 25.0,
    final_div_factor: float = 10000.0,
) -> base.Schedule: ...
def cosine_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 10000.0,
) -> base.Schedule: ...
def warmup_constant_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
) -> base.Schedule: ...
def warmup_cosine_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0,
    exponent: float = 1.0,
) -> base.Schedule: ...
def warmup_exponential_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    transition_steps: int,
    decay_rate: float,
    transition_begin: int = 0,
    staircase: bool = False,
    end_value: float | None = None,
) -> base.Schedule: ...
def sgdr_schedule(
    cosine_kwargs: Iterable[dict[str, chex.Numeric]],
) -> base.Schedule: ...
