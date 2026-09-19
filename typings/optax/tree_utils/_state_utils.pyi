from collections.abc import Callable as Callable
from typing import Any, Protocol

import dataclasses

from optax._src import base as base

class _ParamsPlaceholder:
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, aux, children): ...

@dataclasses.dataclass(frozen=True)
class NamedTupleKey:
    tuple_name: str
    name: str

class Initable(Protocol):
    def init(self, params: base.Params) -> base.OptState: ...

def tree_map_params(
    initable: Callable[[base.Params], base.OptState] | Initable,
    f: Callable[..., Any],
    state: base.OptState,
    /,
    *rest: Any,
    transform_non_params: Callable[..., Any] | None = None,
    is_leaf: Callable[[base.Params], bool] | None = None,
) -> base.OptState: ...
def tree_get_all_with_path(
    tree: base.PyTree,
    key: Any,
    filtering: Callable[[_KeyPath, Any], bool] | None = None,
) -> list[tuple[_KeyPath, Any]]: ...
def tree_get(
    tree: base.PyTree,
    key: Any,
    default: Any | None = None,
    filtering: Callable[[_KeyPath, Any], bool] | None = None,
) -> Any: ...
def tree_set(
    tree: base.PyTree,
    filtering: Callable[[_KeyPath, Any], bool] | None = None,
    /,
    **kwargs: Any,
) -> base.PyTree: ...
