from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any, TypeVar

import abc
import dataclasses

from _typeshed import Incomplete
from jax._src import (
    api_util as api_util,
    core as core,
    custom_derivatives as custom_derivatives,
    dtypes as dtypes,
    source_info_util as source_info_util,
    state as state,
    tree_util as tree_util,
    util as util,
)
from jax._src.lax.control_flow import conditionals as conditionals
from jax._src.pallas import pallas_call as pallas_call
from jax._src.pallas.fuser import block_spec as block_spec
from jax._src.pallas.fuser.fusible import fusible_p as fusible_p
from jax._src.util import foreach as foreach

map: Incomplete
unsafe_map: Incomplete
zip: Incomplete
unsafe_zip: Incomplete
T = TypeVar("T")
pack_dtype_p: Incomplete

@pack_dtype_p.def_abstract_eval
def pack_dtype_abstract_eval(*xs, dtype): ...
def pack(*xs, dtype): ...

unpack_dtype_p: Incomplete

@unpack_dtype_p.def_abstract_eval
def unpack_dtype_abstract_eval(x): ...
def unpack(x): ...

class FusibleElementDType(dtypes.extended, metaclass=abc.ABCMeta): ...

class FusibleTyRules:
    allow_conversion: bool

class FusionDType(dtypes.ExtendedDType, util.StrictABC, metaclass=abc.ABCMeta):
    type = FusibleElementDType
    @abc.abstractmethod
    def abstract_unpack(self, x) -> Sequence[Any]: ...
    @abc.abstractmethod
    def abstract_pack(self, *xs): ...
    @classmethod
    def register_op(cls, primitive): ...
    @classmethod
    def get_op_rule(cls, primitive): ...
    @property
    def name(self): ...
    @abc.abstractmethod
    def pull_block_spec_one_step(self, aval_out, *args, **kwargs): ...
    @abc.abstractmethod
    def unpack_push_block_spec(self, aval_in, *args, **kwargs): ...
    @abc.abstractmethod
    def unpack_pull_block_spec(self, aval_in, *args, **kwargs): ...

def physicalize(f): ...
@util.weakref_lru_cache
def physicalize_closed_jaxpr(jaxpr: core.ClosedJaxpr) -> core.ClosedJaxpr: ...
def physicalize_jaxpr(jaxpr: core.Jaxpr) -> core.Jaxpr: ...

@dataclasses.dataclass
class Context:
    avals_in: Sequence[Any]
    avals_out: Sequence[Any]

def physicalize_interp(
    jaxpr: core.Jaxpr,
    consts: Sequence[core.Value],
    *args: core.Value,
): ...
