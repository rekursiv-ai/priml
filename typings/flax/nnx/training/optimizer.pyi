import typing as tp

from _typeshed import Incomplete
from flax import nnx as nnx
from flax.nnx import filterlib as filterlib
from flax.nnx.pytreelib import Pytree as Pytree
from flax.nnx.variablelib import Variable as Variable

import optax

M = tp.TypeVar("M", bound=nnx.Module)
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])

class OptState(Variable): ...
class OptArray(OptState): ...
class OptVariable(OptState): ...

def to_opt_state(tree): ...

class _Missing: ...

MISSING: Incomplete

class Optimizer(Pytree, tp.Generic[M]):
    step: Incomplete
    tx: Incomplete
    opt_state: Incomplete
    wrt: Incomplete
    def __init__(
        self,
        model: M,
        tx: optax.GradientTransformation,
        *,
        wrt: filterlib.Filter,
    ) -> None: ...
    def update(self, model: M, grads, /, **kwargs): ...

class ModelAndOptimizer(Optimizer[M]):
    model: Incomplete
    def __init__(
        self,
        model: M,
        tx: optax.GradientTransformation,
        *,
        wrt: filterlib.Filter = ...,
    ) -> None: ...
    def update(self, grads, /, **kwargs): ...
