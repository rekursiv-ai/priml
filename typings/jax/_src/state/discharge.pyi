from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any, Protocol, TypeVar

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    api_util as api_util,
    config as config,
    core as core,
    custom_derivatives as custom_derivatives,
    literals as literals,
    pjit as pjit,
    sharding_impls as sharding_impls,
    source_info_util as source_info_util,
    tree_util as tree_util,
)
from jax._src.interpreters import (
    ad as ad,
    mlir as mlir,
)
from jax._src.lax import lax as lax
from jax._src.state import indexing as indexing
from jax._src.state.primitives import (
    addupdate_p as addupdate_p,
    get_p as get_p,
    pin as pin,
    swap_p as swap_p,
    unpin as unpin,
)
from jax._src.state.types import (
    AbstractRef as AbstractRef,
    BitcastTransform as BitcastTransform,
    RefEffect as RefEffect,
    ReshapeTransform as ReshapeTransform,
    get_ref_aval_from_value as get_ref_aval_from_value,
    uninitialized as uninitialized,
)
from jax._src.state.utils import (
    bitcast as bitcast,
    hoist_consts_to_refs as hoist_consts_to_refs,
)
from jax._src.typing import Array as Array
from jax._src.util import (
    foreach as foreach,
    safe_map as safe_map,
    safe_zip as safe_zip,
    split_list as split_list,
    unzip2 as unzip2,
    weakref_lru_cache as weakref_lru_cache,
)

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
PyTreeDef: Incomplete

def discharge_state(
    jaxpr: core.Jaxpr,
    consts: Sequence[Any],
    *,
    should_discharge: bool | Sequence[bool] = True,
) -> tuple[core.Jaxpr, Sequence[Any]]: ...
def discharge_state2(
    jaxpr: core.ClosedJaxpr,
    should_discharge: bool | Sequence[bool] = True,
) -> core.ClosedJaxpr: ...

@dataclasses.dataclass
class Environment:
    env: dict[core.Var, Any]
    def read(self, v: core.Atom) -> Any: ...
    def write(self, v: core.Var, val: Any) -> None: ...

class DischargeRule(Protocol):
    def __call__(
        self,
        in_avals: Sequence[core.AbstractValue],
        out_avals: Sequence[core.AbstractValue],
        *args: Any,
        **params: Any,
    ) -> tuple[Sequence[Any | None], Any | Sequence[Any]]: ...

def register_discharge_rule(prim: core.Primitive): ...

class PartialDischargeRule(Protocol):
    def __call__(
        self,
        should_discharge: Sequence[bool],
        in_avals: Sequence[core.AbstractValue],
        out_avals: Sequence[core.AbstractValue],
        *args: Any,
        **params: Any,
    ) -> tuple[Sequence[Any | None], Any | Sequence[Any]]: ...

def register_partial_discharge_rule(prim: core.Primitive): ...
def transform_array(x, transforms): ...
def transform_swap_array(x, transforms, val): ...

run_state_p: Incomplete

def initial_style_jaxpr(
    fun: Callable,
    in_tree: PyTreeDef,
    in_avals: Sequence[core.AbstractValue],
    dbg: core.DebugInfo,
) -> tuple[core.Jaxpr, list[Any], PyTreeDef]: ...

T = TypeVar("T")

def run_state(f: Callable[..., None]) -> Callable[[T], T]: ...
def run_state_reference(f: Callable[..., None]): ...
def custom_vjp_call_discharge(
    in_avals,
    out_avals,
    *args,
    call_jaxpr,
    fwd_jaxpr_thunk,
    bwd,
    out_trees,
    symbolic_zeros,
    num_consts,
): ...
