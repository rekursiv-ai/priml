from collections.abc import Callable as Callable
from typing import Any, NamedTuple, Protocol

from optax._src import (
    base as base,
    numerics as numerics,
    utils as utils,
)

import chex

class TraceState(NamedTuple):
    trace: base.Params

def trace(
    decay: float,
    nesterov: bool = False,
    accumulator_dtype: Any | None = None,
) -> base.GradientTransformation: ...

class EmaState(NamedTuple):
    count: chex.Array
    ema: base.Params

def ema(
    decay: float,
    debias: bool = True,
    accumulator_dtype: Any | None = None,
) -> base.GradientTransformation: ...

class ShouldSkipUpdateFunction(Protocol):
    def __call__(
        self,
        updates: base.Updates,
        gradient_step: chex.Array,
        params: base.Params | None,
    ) -> tuple[chex.Array, chex.ArrayTree]: ...

def skip_not_finite(
    updates: base.Updates,
    gradient_step: chex.Array,
    params: base.Params | None,
) -> tuple[chex.Array, chex.ArrayTree]: ...
def skip_large_updates(
    updates: base.Updates,
    gradient_step: chex.Array,
    params: base.Params | None,
    max_squared_norm: float,
) -> tuple[chex.Array, chex.ArrayTree]: ...

class MultiStepsState(NamedTuple):
    mini_step: chex.Array
    gradient_step: chex.Array
    inner_opt_state: Any
    acc_grads: Any
    skip_state: chex.ArrayTree = ...

class MultiSteps:
    def __init__(
        self,
        opt: base.GradientTransformation,
        every_k_schedule: int | Callable[[chex.Array], chex.Array],
        use_grad_mean: bool = True,
        should_skip_update_fn: ShouldSkipUpdateFunction | None = None,
    ) -> None: ...
    @property
    def inner_opt(self): ...
    def init(self, params: Any) -> MultiStepsState: ...
    def update(
        self,
        updates: base.Updates,
        state: MultiStepsState,
        params: base.Params | None = None,
        **extra_args: Any,
    ) -> tuple[base.Updates, MultiStepsState]: ...
    def has_updated(self, state: MultiStepsState | chex.ArrayTree) -> chex.Array: ...
    def gradient_transformation(self) -> base.GradientTransformation: ...
