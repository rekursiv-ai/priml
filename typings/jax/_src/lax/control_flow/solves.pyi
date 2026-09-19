from collections.abc import Callable as Callable
from typing import Any, NamedTuple

from _typeshed import Incomplete
from jax._src import (
    ad_util as ad_util,
    api as api,
    api_util as api_util,
    core as core,
    custom_derivatives as custom_derivatives,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
    pxla as pxla,
)
from jax._src.traceback_util import api_boundary as api_boundary
from jax._src.tree_util import (
    FlatTree as FlatTree,
    tree_leaves as tree_leaves,
)
from jax._src.util import (
    safe_map as safe_map,
    split_list as split_list,
)

class _RootTuple(NamedTuple):
    f: Incomplete
    solve: Incomplete
    l_and_s: Incomplete

@api_boundary
def custom_root(
    f: Callable,
    initial_guess: Any,
    solve: Callable[[Callable, Any], Any],
    tangent_solve: Callable[[Callable, Any], Any],
    has_aux: bool = False,
): ...

class _LinearSolveTuple(NamedTuple):
    matvec: Any
    vecmat: Any
    solve: Any
    transpose_solve: Any
    def transpose(self): ...

def custom_linear_solve(
    matvec: Callable,
    b: Any,
    solve: Callable[[Callable, Any], Any],
    transpose_solve: Callable[[Callable, Any], Any] | None = None,
    symmetric: bool = False,
    has_aux: bool = False,
): ...

linear_solve_p: Incomplete
