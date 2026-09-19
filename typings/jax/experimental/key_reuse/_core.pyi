from collections.abc import (
    Callable as Callable,
    Iterator,
)
from typing import Any, NamedTuple

from _typeshed import Incomplete
from jax import (
    lax as lax,
    tree_util as tree_util,
)
from jax._src import (
    api_util as api_util,
    core as core,
    pjit as pjit,
    prng as prng,
    random as random,
    source_info_util as source_info_util,
    traceback_util as traceback_util,
    util as util,
)
from jax._src.ad_checkpoint import remat_p as remat_p
from jax._src.debugging import debug_callback_p as debug_callback_p
from jax._src.effects import Effect as Effect
from jax._src.hashable_array import HashableArray as HashableArray
from jax._src.shard_map import shard_map_p as shard_map_p
from jax._src.util import weakref_lru_cache as weakref_lru_cache
from jax.errors import KeyReuseError as KeyReuseError
from jax.interpreters import (
    batching as batching,
    mlir as mlir,
)

import numpy as np

def key_reuse_error_with_source_traceback(
    message: str,
    traceback: source_info_util.Traceback | None,
) -> KeyReuseError: ...

class _SourceSinkBase:
    idx: int
    mask: bool | np.ndarray
    def __init__(self, idx: int, mask: bool | np.bool_ | np.ndarray = True) -> None: ...
    def __setattr__(self, *args, **kwargs) -> None: ...
    def __eq__(self, other): ...
    def __lt__(self, other): ...
    def __hash__(self): ...

class Sink(_SourceSinkBase): ...
class Source(_SourceSinkBase): ...

class Forward(NamedTuple):
    in_idx: int
    out_idx: int

class KeyReuseSignature:
    def __init__(self, *args) -> None: ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    @property
    def sinks(self) -> Iterator[Sink]: ...
    @property
    def sources(self) -> Iterator[Source]: ...
    @property
    def forwards(self) -> Iterator[Forward]: ...
    def check_signature(
        self,
        *args,
        funcname: str = "function",
        context=None,
    ) -> None: ...
    def update_consumption(self, args_in, args_out) -> None: ...

class DynamicKeyReuseSignature(NamedTuple):
    signature: Callable[[core.JaxprEqn], KeyReuseSignature]

def dynamic_key_reuse_signature(
    f: Callable[[core.JaxprEqn], KeyReuseSignature],
) -> DynamicKeyReuseSignature: ...
def key_reuse_signature_from_eqn(eqn: core.JaxprEqn) -> KeyReuseSignature: ...
def key_reuse_signature_from_primitive(prim, *args, **params): ...

consume_effect: Incomplete
consume_p: Incomplete

def consume(key): ...

assert_effect: Incomplete
assert_consumed_value_p: Incomplete

def assert_unconsumed(key) -> None: ...
def assert_consumed(key, value: bool = True) -> None: ...

key_reuse_signatures: dict[core.Primitive, KeyReuseSignature | DynamicKeyReuseSignature]

def unknown_signature(eqn): ...
@weakref_lru_cache
def jaxpr_type_signature(jaxpr: core.Jaxpr) -> KeyReuseSignature: ...
def function_type_signature(
    fun: Callable[..., Any],
    *args: Any,
) -> KeyReuseSignature: ...
def check_key_reuse_jaxpr(jaxpr: core.Jaxpr) -> None: ...
def check_key_reuse(fun: Callable[..., Any], /, *args: Any) -> None: ...
def call_impl_with_key_reuse_checks(
    prim: core.Primitive,
    raw_impl: Callable[..., Any],
    *args,
    **kwargs,
) -> Any: ...
