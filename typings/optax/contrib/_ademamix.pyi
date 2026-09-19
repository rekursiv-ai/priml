from collections.abc import Callable
from typing import Any, NamedTuple

from optax._src import (
    base as base,
    combine as combine,
    numerics as numerics,
    transform as transform,
    utils as utils,
)

import chex

class ScaleByAdemamixState(NamedTuple):
    count: chex.Array
    count_m2: chex.Array
    m1: base.Updates
    m2: base.Updates
    nu: base.Updates

def scale_by_ademamix(
    b1: float = 0.9,
    b2: float = 0.999,
    b3: base.ScalarOrSchedule = 0.9999,
    alpha: base.ScalarOrSchedule = 6.0,
    eps: float = 1e-08,
    eps_root: float = 0.0,
    mu_dtype: chex.ArrayDType | None = None,
) -> base.GradientTransformation: ...
def ademamix(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    b3: base.ScalarOrSchedule = 0.9999,
    alpha: base.ScalarOrSchedule = 5.0,
    eps: float = 1e-08,
    eps_root: float = 0.0,
    mu_dtype: Any | None = None,
    weight_decay: float = 0.0,
    mask: Any | Callable[[base.Params], Any] | None = None,
) -> base.GradientTransformation: ...

class ScaleBySimplifiedAdEMAMixState(NamedTuple):
    t: chex.Array
    m: base.Updates
    n: base.Updates

def lerp(t, a, b): ...
def scale_by_simplified_ademamix(
    b1: float = 0.99,
    b2: float = 0.95,
    alpha: base.ScalarOrSchedule = 0.0,
    eps: float = 1e-08,
    eps_root: float = 0.0,
) -> base.GradientTransformation: ...
def simplified_ademamix(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.99,
    b2: float = 0.95,
    alpha: base.ScalarOrSchedule = 0.0,
    eps: float = 1e-08,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    mask: Any | Callable[[base.Params], Any] | None = None,
) -> base.GradientTransformation: ...
