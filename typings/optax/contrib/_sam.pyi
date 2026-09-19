from collections.abc import Callable as Callable
from typing import NamedTuple

from _typeshed import Incomplete
from optax._src import (
    base as base,
    update as update,
    utils as utils,
)

import jax

NormalizeState: Incomplete

def normalize() -> base.GradientTransformation: ...

class SAMState(NamedTuple):
    steps_since_sync: jax.Array
    opt_state: base.OptState
    adv_state: base.OptState
    cache: base.Params | None

def sam(
    optimizer: base.GradientTransformation,
    adv_optimizer: base.GradientTransformation,
    sync_period: int = 2,
    reset_state: bool = True,
    opaque_mode: bool = False,
    batch_axis_name: str | None = None,
) -> base.GradientTransformationExtraArgs: ...
