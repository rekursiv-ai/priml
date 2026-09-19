import abc
import dataclasses
import typing as tp

from flax.nnx import (
    extract as extract,
    graphlib as graphlib,
    variablelib as variablelib,
)
from flax.nnx.module import Module as Module
from flax.nnx.proxy_caller import (
    CallableProxy as CallableProxy,
    DelayedAccessor as DelayedAccessor,
)
from flax.nnx.transforms import general as general
from flax.typing import (
    MISSING as MISSING,
    Leaf as Leaf,
    Missing as Missing,
)
from jax._src import checkify as checkify_lib

A = tp.TypeVar("A")
C = tp.TypeVar("C")
B = tp.TypeVar("B")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
G = tp.TypeVar("G", bound=tp.Callable[..., tp.Any])
M = tp.TypeVar("M", bound=Module)
MA = tp.TypeVar("MA", bound=Module)
N = tp.TypeVar("N", bound=Module)
StrInt = tp.TypeVar("StrInt", str, int)
AxisName = tp.Hashable
type Leaves = list[Leaf]
Index = int

@tp.overload
def resolve_kwargs(
    fun: tp.Callable[..., tp.Any],
    args: tuple,
    kwargs: dict[str, tp.Any],
) -> tuple: ...
@tp.overload
def resolve_kwargs() -> tp.Callable[[F], F]: ...

class LiftedModule(Module, tp.Generic[M], metaclass=abc.ABCMeta):
    def __call__(self, *args, **kwargs) -> tp.Any: ...
    @property
    def call(self) -> tp.Any: ...

@dataclasses.dataclass(frozen=True)
class ValueMetadata:
    var_type: type[variablelib.Variable]
    value: tp.Any
    metadata: dict[str, tp.Any]

@dataclasses.dataclass(eq=False)
class TreeEvalShapeFn:
    f: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, *args, **kwargs): ...

def eval_shape(
    f: tp.Callable[..., A],
    *args: tp.Any,
    graph: bool | None = None,
    **kwargs: tp.Any,
) -> A: ...

@dataclasses.dataclass(eq=False)
class CheckifyFn:
    f: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    def __call__(self, *pure_args, **pure_kwargs): ...

@dataclasses.dataclass(eq=False)
class TreeCheckifyFn:
    f: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, *args): ...

def checkify(
    f: tp.Callable[..., checkify_lib.Out],
    errors: frozenset[type[checkify_lib.JaxException]] = ...,
    graph: bool | None = None,
) -> tp.Callable[..., tuple[checkify_lib.Error, checkify_lib.Out]]: ...

@dataclasses.dataclass(eq=False)
class TreeCondFn:
    f: tp.Callable[..., tp.Any]
    def __post_init__(self) -> None: ...
    @extract.treemap_copy_args
    def __call__(self, *args): ...

def cond(
    pred,
    true_fun: tp.Callable[..., A],
    false_fun: tp.Callable[..., A],
    *operands,
    graph: bool | None = None,
) -> A: ...
def switch(
    index,
    branches: tp.Sequence[tp.Callable[..., A]],
    *operands,
    graph: bool | None = None,
) -> A: ...
