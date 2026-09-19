from typing import Any, NamedTuple

class GradientTransformation(NamedTuple):
    init: Any
    update: Any

class GradientTransformationExtraArgs(GradientTransformation):
    update: Any

type EmptyState = Any
type OptState = Any
type Params = Any
type Updates = Any
type ScalarOrSchedule = float | Any
type Schedule = Any
type TransformInitFn = Any
type TransformUpdateFn = Any
type TransformUpdateExtraArgsFn = Any
type MaskOrFn = Any

def identity() -> GradientTransformation: ...
def set_to_zero() -> GradientTransformation: ...
def stateless(*args: Any, **kwargs: Any) -> GradientTransformation: ...
def stateless_with_tree_map(*args: Any, **kwargs: Any) -> GradientTransformation: ...
