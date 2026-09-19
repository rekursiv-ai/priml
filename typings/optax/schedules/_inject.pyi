from collections.abc import (
    Callable as Callable,
    Iterable,
)
from typing import NamedTuple

from _typeshed import Incomplete
from optax._src import (
    base as base,
    numerics as numerics,
)

import chex
import jax.numpy as jnp

class InjectHyperparamsState(NamedTuple):
    count: jnp.ndarray
    hyperparams: dict[str, chex.Numeric]
    inner_state: base.OptState

class InjectStatefulHyperparamsState(NamedTuple):
    count: jnp.ndarray
    hyperparams: dict[str, chex.Numeric]
    hyperparams_states: dict[str, base.ScheduleState]
    inner_state: base.OptState

def inject_hyperparams(
    inner_factory: Callable[..., base.GradientTransformation],
    static_args: str | Iterable[str] = (),
    hyperparam_dtype: jnp.dtype | None = None,
) -> Callable[..., base.GradientTransformationExtraArgs]: ...
def inject_stateful_hyperparams(
    inner_factory: Callable[..., base.GradientTransformation],
    static_args: str | Iterable[str] = (),
    hyperparam_dtype: jnp.dtype | None = None,
) -> Callable[..., base.GradientTransformationExtraArgs]: ...

class WrappedScheduleState(NamedTuple):
    count: chex.Numeric

class WrappedSchedule:
    schedule_fn: Incomplete
    def __init__(self, schedule_fn: base.Schedule) -> None: ...
    def init(self) -> WrappedScheduleState: ...
    def update(
        self,
        state: WrappedScheduleState,
        **extra_args,
    ) -> WrappedScheduleState: ...
    def __call__(self, state: WrappedScheduleState, **extra_args) -> chex.Numeric: ...
