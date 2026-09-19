from collections.abc import Callable
from typing import Any, NamedTuple

from optax._src import base as base

import jax

class SnapshotState(NamedTuple):
    measurement: dict[str, Any]

def snapshot(
    measure_name: str,
    measure: Callable[[base.Updates], jax.Array],
) -> base.GradientTransformation: ...
