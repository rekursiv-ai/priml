from typing import Any

import dataclasses
import typing as tp

from _typeshed import Incomplete
from flax import (
    core as core,
    linen as linen,
    nnx as nnx,
)
from flax.core import (
    FrozenDict as FrozenDict,
    meta as meta,
)
from flax.nnx import (
    graphlib as graphlib,
    variablelib as variablelib,
)
from flax.nnx.module import Module as Module
from flax.nnx.pytreelib import Pytree as Pytree
from flax.nnx.rnglib import Rngs as Rngs
from flax.nnx.statelib import State as State

import jax

M = tp.TypeVar("M", bound=Module)

@dataclasses.dataclass
class Functional(tp.Generic[M]):
    module_type: type[M]
    graphdef: graphlib.GraphDef[M] | None
    args: tuple[tp.Any, ...]
    kwargs: dict[str, tp.Any]
    def init(self, *, rngs: Rngs | None = None) -> State: ...
    def apply(self, *states: tp.Any): ...

def functional(cls) -> tp.Callable[..., Functional[M]]: ...
def lazy_init(fn: Module | tp.Callable[..., tp.Any], *args, **kwargs): ...
def current_linen_module() -> linen.Module | None: ...

class ToNNX(Module):
    to_nnx__module: Incomplete
    to_nnx__rngs: Rngs | None
    def __init__(
        self,
        module: linen.Module,
        rngs: Rngs | jax.Array | None = None,
    ) -> None: ...
    @property
    def rngs(self) -> Rngs | None: ...
    @property
    def module(self) -> linen.Module: ...
    def lazy_init(self, *args, **kwargs): ...
    def __getattr__(self, name: str): ...
    def __call__(
        self,
        *args: Any,
        rngs: Rngs | jax.Array | None = None,
        method: tp.Callable[..., Any] | str | None = None,
        mutable: tp.Any = None,
        **kwargs: Any,
    ) -> Any: ...

def linen_rngs_dict(linen_module: linen.Module, add_default: bool = False): ...

class ToLinen(linen.Module):
    nnx_class: tp.Callable[..., Module]
    args: tp.Sequence = ...
    kwargs: tp.Mapping[str, tp.Any] = ...
    skip_rng: bool = ...
    metadata_fn: tp.Callable[[variablelib.Variable], tp.Any] | None = ...
    @linen.compact
    def __call__(
        self,
        *args,
        nnx_method: tp.Callable[..., Any] | str | None = None,
        **kwargs,
    ): ...
    def __getattr__(self, name: str): ...

class _Missing: ...

def to_linen(
    nnx_class: tp.Callable[..., Module],
    *args,
    metadata_fn: tp.Callable[[variablelib.Variable], tp.Any] | None = ...,
    name: str | None = None,
    skip_rng: bool = False,
    abstract_init: bool = True,
    **kwargs,
): ...
def to_linen_class(
    base_nnx_class: type[M],
    base_metadata_fn: tp.Callable[[variablelib.VariableState], tp.Any] | None = ...,
    base_skip_rng: bool = False,
    **partial_kwargs: tp.Any,
) -> type[ToLinen]: ...
