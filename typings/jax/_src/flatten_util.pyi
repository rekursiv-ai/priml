from collections.abc import Callable, Iterable
from typing import Any

from jax._src import dtypes as dtypes
from jax._src.lax import lax as lax
from jax._src.tree_util import (
    Leaf as Leaf,
    PyTreeDef as PyTreeDef,
    tree_flatten as tree_flatten,
    tree_unflatten as tree_unflatten,
)
from jax._src.typing import Array as Array
from jax._src.util import (
    HashablePartial as HashablePartial,
    unzip2 as unzip2,
)

type Sizes = tuple[int, ...]
type Shapes = tuple[tuple[int, ...], ...]

def ravel_pytree(pytree: Any) -> tuple[Array, Callable[[Array], Any]]: ...
def unravel_pytree(
    treedef: PyTreeDef,
    unravel_list: Callable[[Array], Iterable[Leaf]],
    flat: Array,
) -> Any: ...
