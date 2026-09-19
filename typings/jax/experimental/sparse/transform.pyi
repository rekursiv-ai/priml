from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any, NamedTuple

from _typeshed import Incomplete
from jax import lax as lax
from jax._src import (
    api_util as api_util,
    config as config,
    core as core,
    linear_util as lu,
    pjit as pjit,
    sharding_impls as sharding_impls,
)
from jax._src.api_util import flatten_fun_nokwargs as flatten_fun_nokwargs
from jax._src.custom_derivatives import lift_jvp as lift_jvp
from jax._src.lib import pytree as pytree
from jax._src.util import (
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
)
from jax.experimental import sparse as sparse
from jax.experimental.sparse.bcoo import (
    BCOO as BCOO,
    bcoo_multiply_dense as bcoo_multiply_dense,
    bcoo_multiply_sparse as bcoo_multiply_sparse,
)
from jax.experimental.sparse.bcsr import BCSR as BCSR
from jax.tree_util import (
    tree_flatten as tree_flatten,
    tree_map as tree_map,
    tree_unflatten as tree_unflatten,
)

sparse_rules_bcoo: dict[core.Primitive, Callable]
sparse_rules_bcsr: dict[core.Primitive, Callable]
type Array = Any
type ArrayOrSparse = Any

class SparsifyEnv:
    def __init__(self, bufs=()) -> None: ...
    def data(self, spvalue: SparsifyValue) -> Array: ...
    def indices(self, spvalue: SparsifyValue) -> Array: ...
    def indptr(self, spvalue: SparsifyValue) -> Array: ...
    def dense(self, data): ...
    def sparse(
        self,
        shape,
        data=None,
        indices=None,
        indptr=None,
        *,
        data_ref=None,
        indices_ref=None,
        indptr_ref=None,
        indices_sorted: bool = False,
        unique_indices: bool = False,
    ): ...

class SparsifyValue(NamedTuple):
    shape: tuple[int, ...]
    data_ref: int | None
    indices_ref: int | None = ...
    indptr_ref: int | None = ...
    indices_sorted: bool | None = ...
    unique_indices: bool | None = ...
    @property
    def ndim(self): ...
    def is_sparse(self): ...
    def is_dense(self): ...
    def is_bcoo(self): ...
    def is_bcsr(self): ...

def arrays_to_spvalues(spenv: SparsifyEnv, args: Any) -> Any: ...
def spvalues_to_arrays(spenv: SparsifyEnv, spvalues: Any) -> Any: ...
def spvalues_to_avals(spenv: SparsifyEnv, spvalues: Any) -> Any: ...

class SparseTracer(core.Tracer):
    def __init__(self, trace: core.Trace, *, spvalue) -> None: ...
    @property
    def spenv(self): ...
    @property
    def aval(self): ...
    def full_lower(self): ...

class SparseTrace(core.Trace):
    parent_trace: Incomplete
    tag: Incomplete
    spenv: Incomplete
    def __init__(self, parent_trace, tag, spenv) -> None: ...
    def to_sparse_tracer(self, val): ...
    def process_primitive(self, primitive, tracers, params): ...
    def process_call(self, call_primitive, f: lu.WrappedFun, tracers, params): ...
    def process_custom_jvp_call(
        self,
        primitive,
        fun,
        jvp,
        tracers,
        *,
        symbolic_zeros,
    ): ...

@lu.transformation_with_aux2
def sparsify_subtrace(f, store, tag, spenv, spvalues, *bufs): ...
def sparsify_fun(wrapped_fun, args: list[ArrayOrSparse]): ...
def eval_sparse(
    jaxpr: core.Jaxpr,
    consts: Sequence[Array],
    spvalues: Sequence[SparsifyValue],
    spenv: SparsifyEnv,
) -> Sequence[SparsifyValue]: ...
def sparsify_raw(f): ...
def sparsify(f, use_tracer: bool = False): ...
