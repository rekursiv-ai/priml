from collections.abc import Callable as Callable
from typing import Any, NamedTuple

from optax._src import (
    base as base,
    combine as combine,
    transform as transform,
)

class COCOBState(NamedTuple):
    init_particles: base.Updates
    cumulative_gradients: base.Updates
    scale: base.Updates
    subgradients: base.Updates
    reward: base.Updates

def scale_by_cocob(
    alpha: float = 100,
    eps: float = 1e-08,
) -> base.GradientTransformation: ...
def cocob(
    learning_rate: base.ScalarOrSchedule = 1.0,
    alpha: float = 100,
    eps: float = 1e-08,
    weight_decay: float = 0,
    mask: Any | Callable[[base.Params], Any] | None = None,
) -> base.GradientTransformation: ...
