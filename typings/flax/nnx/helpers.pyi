import typing as tp

from _typeshed import Incomplete
from flax.nnx import (
    graphlib as graphlib,
    reprlib as reprlib,
)
from flax.nnx.graphlib import GraphDef as GraphDef
from flax.nnx.module import Module as Module
from flax.nnx.proxy_caller import ApplyCaller as ApplyCaller
from flax.nnx.rnglib import Rngs as Rngs
from flax.nnx.statelib import State as State
from flax.nnx.variablelib import Variable as Variable
from flax.training.train_state import struct as struct

import jax
import optax

A = tp.TypeVar("A")
M = tp.TypeVar("M", bound=Module)
TS = tp.TypeVar("TS", bound=TrainState)

class Dict(reprlib.MappingReprMixin, Module, tp.MutableMapping[str, A]):
    @tp.overload
    def __init__(self, iterable: tp.Iterable[tuple[str, A]], /) -> None: ...
    @tp.overload
    def __init__(
        self,
        mapping: tp.Mapping[str, A] | None = None,
        /,
        **kwargs: A,
    ) -> None: ...
    def __getitem__(self, key) -> A: ...
    def __setitem__(self, key, value) -> None: ...
    def __iter__(self) -> tp.Iterator[str]: ...
    def __len__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __delitem__(self, key: str) -> None: ...
    def __getattr__(self, key: str) -> A: ...
    def __setattr__(self, key: str, value: A) -> None: ...

class List(reprlib.SequenceReprMixin, Module, tp.MutableSequence[A]):
    def __init__(self, it: tp.Iterable[A] | None = None, /) -> None: ...
    def __len__(self) -> int: ...
    def append(self, value: A) -> None: ...
    def insert(self, index: int, value: A) -> None: ...
    def __iter__(self) -> tp.Iterator[A]: ...
    @tp.overload
    def __getitem__(self, index: int) -> A: ...
    @tp.overload
    def __getitem__(self, index: slice) -> list[A]: ...
    def __setitem__(self, index: int | slice, value: A | tp.Iterable[A]) -> None: ...
    def __delitem__(self, index: int | slice) -> None: ...

class Sequential(Module):
    layers: Incomplete
    def __init__(self, *fns: tp.Callable[..., tp.Any]) -> None: ...
    def __call__(self, *args, rngs: Rngs | None = None, **kwargs) -> tp.Any: ...

class ModuleDefApply(tp.Protocol, tp.Generic[M]):
    def __call__(
        self,
        state: State,
        *states: State,
    ) -> ApplyCaller[tuple[State, GraphDef[M]]]: ...

class TrainState(struct.PyTreeNode, tp.Generic[M]):
    graphdef: graphlib.GraphDef[M]
    params: State
    opt_state: optax.OptState
    step: jax.Array
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    @classmethod
    def create(
        cls,
        graphdef: graphlib.GraphDef[M],
        *,
        params: State,
        tx: optax.GradientTransformation,
        step: int = 0,
        **kwargs,
    ): ...
    def __getattr__(self, key: str) -> tp.Any: ...
    def apply(
        self,
        state: State | str,
        *states: State | str,
    ) -> ApplyCaller[tuple[GraphDef[M], State]]: ...
    def apply_gradients(self, grads: State, **kwargs) -> TS: ...

def has_keyword_arg(func: tp.Callable[..., tp.Any], name: str) -> bool: ...
