from collections.abc import (
    Callable as Callable,
    Iterator,
)

from _typeshed import Incomplete
from jax._src import (
    config as config,
    core as core,
    path as path,
    source_info_util as source_info_util,
    util as util,
)
from jax._src.lib import xla_client as xla_client

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
logger: Incomplete

def all_eqns(
    jaxpr: core.Jaxpr,
    revisit_inner_jaxprs: bool = True,
) -> Iterator[tuple[core.Jaxpr, core.JaxprEqn]]: ...
def collect_eqns(jaxpr: core.Jaxpr, key: Callable): ...
def histogram(jaxpr: core.Jaxpr, key: Callable, key_fmt: Callable = ...): ...
def primitives(jaxpr: core.Jaxpr): ...
def primitives_by_source(jaxpr: core.Jaxpr): ...
def primitives_by_shape(jaxpr: core.Jaxpr): ...
def source_locations(jaxpr: core.Jaxpr): ...

MaybeEqn: Incomplete

def var_defs_and_refs(jaxpr: core.Jaxpr): ...

DEFAULT_WORKSPACE_ROOT: str | None

def pprof_equation_profile(
    jaxpr: core.Jaxpr,
    *,
    workspace_root: str | None = None,
) -> bytes: ...
def eqns_using_var_with_invar_index(
    jaxpr: core.Jaxpr,
    invar: core.Var,
) -> Iterator[tuple[core.JaxprEqn, int]]: ...
def jaxpr_and_binder_in_params(
    params,
    index: int,
) -> Iterator[tuple[core.Jaxpr, core.Var]]: ...
def eqns_using_var(jaxpr: core.Jaxpr, invar: core.Var) -> Iterator[core.JaxprEqn]: ...
def maybe_dump_jaxpr_to_file(fun_name: str, jaxpr: core.Jaxpr) -> str | None: ...
