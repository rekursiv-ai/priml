from collections.abc import Callable as Callable
from typing import Any

from _typeshed import Incomplete
from jax import (
    api_util as api_util,
    lax as lax,
)
from jax._src import (
    ad_util as ad_util,
    core as core,
    dispatch as dispatch,
    linear_util as lu,
    pjit as pjit,
    sharding_impls as sharding_impls,
)
from jax._src.util import (
    safe_zip as safe_zip,
    unzip2 as unzip2,
    weakref_lru_cache as weakref_lru_cache,
)
from jax.tree_util import (
    register_pytree_node as register_pytree_node,
    tree_flatten as tree_flatten,
    tree_structure as tree_structure,
    tree_unflatten as tree_unflatten,
    treedef_is_leaf as treedef_is_leaf,
)

def jet(fun, primals, series, factorial_scaled: bool = True, **_): ...

jet2: Incomplete

def fact(n): ...
@lu.transformation2
def jet_fun(f, order, primals, series): ...
@lu.transformation2
def jet_subtrace(f, tag, order, primals, series): ...
@lu.transformation_with_aux2
def traceable(f, store, in_tree_def, *primals_and_series): ...

class JetTracer(core.Tracer):
    primal: Incomplete
    terms: Incomplete
    def __init__(self, trace, primal, terms) -> None: ...
    @property
    def aval(self): ...
    def full_lower(self): ...

class JetTrace(core.Trace):
    tag: Incomplete
    parent_trace: Incomplete
    order: Incomplete
    def __init__(self, tag, parent_trace, order) -> None: ...
    def to_primal_terms_pair(self, val): ...
    def process_primitive(self, primitive, tracers, params): ...
    def process_call(self, call_primitive, f, tracers, params): ...
    def process_custom_jvp_call(
        self,
        primitive,
        fun,
        jvp,
        tracers,
        *,
        symbolic_zeros,
    ): ...
    def process_custom_vjp_call(self, primitive, fun, fwd, bwd, tracers, out_trees): ...

class ZeroTerm: ...

zero_term: Incomplete

class ZeroSeries: ...

zero_series: Incomplete
call_param_updaters: dict[core.Primitive, Callable[..., Any]]
jet_rules: Incomplete

def defzero(prim) -> None: ...
def zero_prop(prim, primals_in, series_in, **params): ...
def deflinear(prim) -> None: ...
def linear_prop(prim, primals_in, series_in, **params): ...
def def_deriv(prim, deriv) -> None: ...
def deriv_prop(prim, deriv, primals_in, series_in): ...
def def_comp(prim, comp, **kwargs) -> None: ...
