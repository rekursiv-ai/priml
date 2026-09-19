import typing as tp

from _typeshed import Incomplete
from flax.nnx import (
    filterlib as filterlib,
    graphlib as graphlib,
    variablelib as variableslib,
)
from flax.nnx.graphlib import GraphState as GraphState
from flax.nnx.pytreelib import (
    Pytree as Pytree,
    PytreeMeta as PytreeMeta,
)
from flax.typing import (
    Key as Key,
    Path as Path,
    PathParts as PathParts,
)

A = tp.TypeVar("A")
B = tp.TypeVar("B")
M = tp.TypeVar("M", bound=Module)
S = tp.TypeVar("S", bound=GraphState | tuple[GraphState, ...])
V = tp.TypeVar("V", bound=variableslib.Variable[tp.Any])
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
type StateMapping = tp.Mapping[Path, tp.Any]
tuple_reduce: Incomplete
tuple_init: Incomplete

class ModuleMeta(PytreeMeta): ...

class Module(Pytree, metaclass=ModuleMeta):
    def sow(
        self,
        variable_type: type[variableslib.Variable[B]] | str,
        name: str,
        value: A,
        reduce_fn: tp.Callable[[B, A], B] = ...,
        init_fn: tp.Callable[[], B] = ...,
    ) -> bool: ...
    def perturb(
        self,
        name: str,
        value: tp.Any,
        variable_type: str | type[variableslib.Variable[tp.Any]] = ...,
    ): ...
    def iter_modules(self) -> tp.Iterator[tuple[PathParts, Module]]: ...
    def iter_children(self) -> tp.Iterator[tuple[Key, Module]]: ...
    def set_attributes(
        self,
        *filters: filterlib.Filter,
        raise_if_not_found: bool = True,
        graph: bool | None = None,
        **attributes: tp.Any,
    ) -> None: ...
    def train(self, **attributes): ...
    def eval(self, **attributes): ...

def view(
    node: A,
    /,
    *,
    only: filterlib.Filter = ...,
    raise_if_not_found: bool = True,
    graph: bool | None = None,
    **kwargs,
) -> A: ...
def view_info(
    node: Module,
    /,
    *,
    only: filterlib.Filter = ...,
    graph: bool | None = None,
) -> str: ...
def first_from(*args: A | None, error_msg: str) -> A: ...
def iter_modules(
    module: Module,
    /,
    *,
    graph: bool | None = None,
) -> tp.Iterator[tuple[PathParts, Module]]: ...

iter_children = graphlib.iter_children
