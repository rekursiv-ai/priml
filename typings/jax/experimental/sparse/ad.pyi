from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any

from _typeshed import Incomplete
from jax import tree_util as tree_util
from jax._src import core as core
from jax._src.traceback_util import api_boundary as api_boundary
from jax._src.util import (
    safe_zip as safe_zip,
    split_list as split_list,
    wraps as wraps,
)
from jax.experimental.sparse._base import JAXSparse as JAXSparse

is_sparse: Incomplete

def flatten_fun_for_sparse_ad(
    fun,
    argnums: int | tuple[int, ...],
    args: tuple[Any, ...],
): ...
def value_and_grad(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    **kwargs,
) -> Callable[..., tuple[Any, Any]]: ...
def grad(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    **kwargs,
) -> Callable: ...
def jacfwd(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    **kwargs,
) -> Callable: ...
def jacrev(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    **kwargs,
) -> Callable: ...

jacobian = jacrev
