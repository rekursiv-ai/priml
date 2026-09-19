from collections.abc import (
    Callable as Callable,
    Mapping,
)
from typing import Any, Self, TypeVar, dataclass_transform, overload

from . import serialization as serialization

_T = TypeVar("_T")

# Mirrors typeshed's `dataclasses.field` overloads: the default/default_factory
# arms return `_T` so the annotated attribute keeps its declared type, while the
# no-default arm returns `Any` because there is no value to infer one from.
@overload
def field(
    pytree_node: bool = True,
    *,
    metadata: Mapping[Any, Any] | None = None,
    default: _T,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    kw_only: bool = ...,
) -> _T: ...
@overload
def field(
    pytree_node: bool = True,
    *,
    metadata: Mapping[Any, Any] | None = None,
    default_factory: Callable[[], _T],
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    kw_only: bool = ...,
) -> _T: ...
@overload
def field(
    pytree_node: bool = True,
    *,
    metadata: Mapping[Any, Any] | None = None,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    kw_only: bool = ...,
) -> Any: ...
@dataclass_transform(field_specifiers=(field,))
@overload
def dataclass(clz: _T, **kwargs: Any) -> _T: ...
@dataclass_transform(field_specifiers=(field,))
@overload
def dataclass(**kwargs: Any) -> Callable[[_T], _T]: ...

TNode = TypeVar("TNode", bound=PyTreeNode)

@dataclass_transform(field_specifiers=(field,))
class PyTreeNode:
    def __init_subclass__(cls, **kwargs: Any) -> None: ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def replace(self, **overrides: Any) -> Self: ...
