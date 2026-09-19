from collections.abc import Callable as Callable
from typing import Any, TypeVar

from _typeshed import Incomplete
from jax._src import (
    core as core,
    traceback_util as traceback_util,
)
from jax._src.core import (
    Primitive as Primitive,
    get_aval as get_aval,
    typeof as typeof,
    valid_jaxtype as valid_jaxtype,
)
from jax._src.tree_util import (
    register_pytree_node as register_pytree_node,
    tree_map as tree_map,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)
from jax._src.util import safe_map as safe_map

T = TypeVar("T")
map = safe_map

def add_jaxvals(x: ArrayLike, y: ArrayLike) -> Array: ...

add_jaxvals_p: Incomplete
add_any_p = add_jaxvals_p

@add_jaxvals_p.def_impl
def add_impl(x, y): ...

raw_jaxval_adders: Incomplete

@add_jaxvals_p.def_abstract_eval
def add_abstract(x, y): ...
def zeros_like_aval(aval: core.AbstractValue) -> Array: ...

aval_zeros_likers: dict[type, Callable[[Any], Array]]

def zeros_like_jaxval(val): ...
def instantiate(z: Zero | Array) -> Array: ...

class Zero:
    aval: Incomplete
    def __init__(self, aval: core.AbstractValue) -> None: ...
    def instantiate(self): ...

def p2tz(primal_value): ...
def p2cz(primal_value): ...

stop_gradient_p: Primitive

class SymbolicZero:
    aval: Incomplete
    def __init__(self, aval: core.AbstractValue) -> None: ...
    def __getattr__(self, name): ...
    @staticmethod
    def from_primal_value(val: Any) -> SymbolicZero: ...

def zero_from_primal(val, symbolic_zeros: bool = False): ...

type JaxTypeOrTracer = Any

def replace_internal_symbolic_zeros(
    x: JaxTypeOrTracer | Zero,
) -> JaxTypeOrTracer | SymbolicZero: ...
def replace_rule_output_symbolic_zeros(
    x: JaxTypeOrTracer | SymbolicZero,
) -> JaxTypeOrTracer | Zero: ...

zeros_like_p: Primitive
