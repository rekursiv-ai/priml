from typing import Any, NamedTuple

from _typeshed import Incomplete
from optax._src import base as base

NonNegativeParamsState: Incomplete

def keep_params_nonnegative() -> base.GradientTransformation: ...

class ZeroNansState(NamedTuple):
    found_nan: Any

def zero_nans() -> base.GradientTransformation: ...
