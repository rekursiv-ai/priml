from collections.abc import Callable as Callable
from typing import Any

from optax._src import (
    base as base,
    combine as combine,
    numerics as numerics,
    transform as transform,
)

def scale_by_acprop(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16,
) -> base.GradientTransformation: ...
def acprop(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16,
    weight_decay: float = 0.0,
    mask: Any | Callable[[base.Params], Any] | None = None,
) -> base.GradientTransformation: ...
