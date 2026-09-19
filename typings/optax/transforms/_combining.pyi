from collections.abc import (
    Callable as Callable,
    Hashable,
    Mapping,
)
from typing import NamedTuple

from optax._src import (
    base as base,
    wrappers as wrappers,
)

def chain(
    *args: base.GradientTransformation,
) -> base.GradientTransformationExtraArgs: ...
def named_chain(
    *args: tuple[str, base.GradientTransformation],
) -> base.GradientTransformationExtraArgs: ...

class PartitionState(NamedTuple):
    inner_states: Mapping[Hashable, base.OptState]

def partition(
    transforms: Mapping[Hashable, base.GradientTransformation],
    param_labels: base.PyTree | Callable[[base.PyTree], base.PyTree],
    *,
    mask_compatible_extra_args: bool = False,
) -> base.GradientTransformationExtraArgs: ...
