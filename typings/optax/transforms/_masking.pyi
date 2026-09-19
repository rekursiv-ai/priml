from collections.abc import Callable as Callable
from typing import Any, NamedTuple

from optax._src import base as base

class MaskedState(NamedTuple):
    inner_state: Any

class MaskedNode(NamedTuple): ...

def masked(
    inner: base.GradientTransformation,
    mask: base.PyTree | Callable[[base.Params], base.PyTree],
    *,
    mask_compatible_extra_args: bool = False,
) -> base.GradientTransformationExtraArgs: ...
