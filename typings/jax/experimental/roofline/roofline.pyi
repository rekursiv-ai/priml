from collections.abc import (
    Callable as Callable,
    Sequence,
)
from dataclasses import dataclass
from typing import Any, Protocol

from _typeshed import Incomplete
from jax._src import (
    api as api,
    core as core,
    prng as prng,
    source_info_util as source_info_util,
    traceback_util as traceback_util,
    util as util,
)
from jax._src.api import make_jaxpr as make_jaxpr
from jax._src.interpreters.partial_eval import dce_jaxpr as dce_jaxpr
from jax._src.mesh import (
    AbstractMesh as AbstractMesh,
    Mesh as Mesh,
)
from jax._src.shard_map import (
    shard_map as shard_map,
    shard_map_p as shard_map_p,
)
from jax._src.tree_util import (
    broadcast_prefix as broadcast_prefix,
    tree_flatten as tree_flatten,
    tree_map as tree_map,
    tree_unflatten as tree_unflatten,
)
from jax._src.util import foreach as foreach
from jax.sharding import NamedSharding as NamedSharding

type ShapeDtypeStructTree = Any
type Specs = Any
ValidRooflineDtype: Incomplete
map: Incomplete

@dataclass(frozen=True, slots=True, kw_only=True)
class RooflineRuleContext:
    name_stack: source_info_util.NameStack
    primitive: core.Primitive
    avals_in: Sequence[core.AbstractValue]
    avals_out: Sequence[core.AbstractValue]
    jaxpr_eqn_ctx: core.JaxprEqnContext
    mesh: Mesh | AbstractMesh | None
    pin_lhs_in_vmem: bool
    pin_rhs_in_vmem: bool

@dataclass(frozen=True, slots=True, kw_only=True)
class RooflineShape:
    shape: tuple[int, ...]
    dtype: ValidRooflineDtype
    @classmethod
    def from_aval(cls, aval: core.AbstractValue) -> RooflineShape: ...
    @property
    def size(self) -> int: ...
    @property
    def bytes(self) -> int: ...
    @classmethod
    def total_bytes(cls, avals: Sequence[core.AbstractValue]) -> int: ...

@dataclass(frozen=True, slots=True, kw_only=True)
class RooflineResult:
    flops: int = ...
    unfused_flops: int = ...
    ici_bytes: dict[str, int] = ...
    ici_latency: dict[str, int] = ...
    hbm_bytes: int = ...
    peak_hbm_bytes: int = ...
    unfused_hbm_bytes: int = ...
    @classmethod
    def zeros(cls) -> RooflineResult: ...
    def __add__(self, other: RooflineResult) -> RooflineResult: ...
    def __mul__(self, constant: float) -> RooflineResult: ...
    def __rmul__(self, constant: float) -> RooflineResult: ...

class _RooflineRule(Protocol):
    def __call__(
        self,
        ctx: RooflineRuleContext,
        *args: RooflineShape,
        **kw,
    ) -> RooflineResult: ...

def roofline(
    f: Callable,
    mesh: Mesh | AbstractMesh | None = None,
    in_specs: Specs | None = None,
    out_specs: Specs | None = None,
    *,
    pin_lhs_in_vmem: bool = False,
    pin_rhs_in_vmem: bool = False,
    vjp: bool = False,
    print_jaxpr: bool = False,
) -> Callable[..., tuple[ShapeDtypeStructTree, RooflineResult]]: ...
def register_roofline(prim: core.Primitive): ...
def register_standard_roofline(prim: core.Primitive): ...
def roofline_and_grad(
    f: Callable,
    mesh: Mesh | AbstractMesh,
    in_specs: Specs,
    out_specs: Specs,
    *,
    pin_lhs_in_vmem: bool = False,
    pin_rhs_in_vmem: bool = False,
    print_jaxpr: bool = False,
) -> Callable[..., tuple[ShapeDtypeStructTree, RooflineResult, RooflineResult]]: ...
