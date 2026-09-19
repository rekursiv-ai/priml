from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

from _typeshed import Incomplete
from jax._src import (
    api_util as api_util,
    core as core,
    custom_api_util as custom_api_util,
    errors as errors,
    linear_util as lu,
    source_info_util as source_info_util,
    traceback_util as traceback_util,
    tree_util as tree_util,
    util as util,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete

class custom_dce:
    fun: Callable[..., Any]
    static_argnums: Sequence[int]
    dce_rule: Callable[..., Any] | None
    def __init__(
        self,
        fun: Callable[..., Any],
        *,
        static_argnums: Sequence[int] = (),
    ) -> None: ...
    __getattr__: Incomplete
    def def_dce(self, dce_rule: Callable[..., Any]) -> Callable[..., Any]: ...
    @traceback_util.api_boundary
    def __call__(self, *args, **kwargs): ...

def check_for_tracers(x) -> None: ...
@lu.transformation_with_aux2
def flatten_dce_rule(
    f: Callable[..., Any],
    store: lu.Store,
    fun_name: str,
    rule_name: str,
    used_outs: Sequence[bool],
    in_tree,
    out_tree,
    out_avals: Sequence[core.AbstractValue],
    *args_flat,
): ...
def custom_dce_impl(*args, fun_jaxpr: core.ClosedJaxpr, **_): ...
def custom_dce_abstract_eval(*args, fun_jaxpr: core.ClosedJaxpr, **_): ...
def custom_dce_batching(
    axis_data: batching.AxisData,
    args,
    dims,
    *,
    num_consts: int,
    fun_jaxpr: core.ClosedJaxpr,
    dce_jaxpr_thunk: Callable[..., tuple[core.ClosedJaxpr, Sequence[bool]]],
): ...
def custom_dce_jvp(primals, tangents, *, fun_jaxpr: core.ClosedJaxpr, **_): ...
def custom_dce_rule(used_outs: Sequence[bool], eqn: core.JaxprEqn): ...

custom_dce_p: Incomplete

def swap_primitives(
    jaxpr: core.Jaxpr,
    old: core.Primitive,
    new: core.Primitive,
) -> core.Jaxpr: ...

dce_sential_p: Incomplete
