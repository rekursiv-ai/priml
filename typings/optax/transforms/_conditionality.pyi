from typing import Any, NamedTuple, Protocol

from optax._src import (
    base as base,
    numerics as numerics,
)

import chex

class ConditionFn(Protocol):
    def __call__(self, step: chex.Array, **extra_args: Any) -> chex.Array: ...

class ConditionallyTransformState(NamedTuple):
    inner_state: Any
    step: chex.Array

def conditionally_transform(
    inner: base.GradientTransformation,
    should_transform_fn: ConditionFn,
    forward_extra_args: bool = False,
) -> base.GradientTransformationExtraArgs: ...

class ConditionallyMaskState(NamedTuple):
    step: chex.Array
    inner_state: base.OptState

def conditionally_mask(
    inner: base.GradientTransformation,
    should_transform_fn: ConditionFn,
    forward_extra_args: bool = False,
) -> base.GradientTransformationExtraArgs: ...

class ApplyIfFiniteState(NamedTuple):
    notfinite_count: Any
    last_finite: Any
    total_notfinite: Any
    inner_state: Any

def apply_if_finite(
    inner: base.GradientTransformation,
    max_consecutive_errors: int,
) -> base.GradientTransformation: ...
