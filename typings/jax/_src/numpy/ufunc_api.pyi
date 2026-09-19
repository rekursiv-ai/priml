from collections.abc import Callable as Callable
from typing import Any

from _typeshed import Incomplete
from jax._src import api as api
from jax._src.lax import (
    control_flow as control_flow,
    lax as lax,
    slicing as slicing,
)
from jax._src.numpy import indexing as indexing
from jax._src.numpy.util import check_arraylike as check_arraylike
from jax._src.numpy.vectorize import vectorize as vectorize
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DTypeLike as DTypeLike,
)
from jax._src.util import (
    canonicalize_axis as canonicalize_axis,
    set_module as set_module,
)

export: Incomplete

class ufunc:
    __doc__: Incomplete
    def __init__(
        self,
        func: Callable[..., Any],
        /,
        nin: int,
        nout: int,
        *,
        name: str | None = None,
        nargs: int | None = None,
        identity: Any = None,
        call: Callable[..., Any] | None = None,
        reduce: Callable[..., Any] | None = None,
        accumulate: Callable[..., Any] | None = None,
        at: Callable[..., Any] | None = None,
        reduceat: Callable[..., Any] | None = None,
    ) -> None: ...
    nin: Incomplete
    nout: Incomplete
    nargs: Incomplete
    identity: Incomplete
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __call__(
        self,
        *args: ArrayLike,
        out: None = None,
        where: None = None,
    ) -> Any: ...
    def reduce(
        self,
        a: ArrayLike,
        axis: int | None = 0,
        dtype: DTypeLike | None = None,
        out: None = None,
        keepdims: bool = False,
        initial: ArrayLike | None = None,
        where: ArrayLike | None = None,
    ) -> Array: ...
    def accumulate(
        self,
        a: ArrayLike,
        axis: int = 0,
        dtype: DTypeLike | None = None,
        out: None = None,
    ) -> Array: ...
    def at(
        self,
        a: ArrayLike,
        indices: Any,
        b: ArrayLike | None = None,
        /,
        *,
        inplace: bool = True,
    ) -> Array: ...
    def reduceat(
        self,
        a: ArrayLike,
        indices: Any,
        axis: int = 0,
        dtype: DTypeLike | None = None,
        out: None = None,
    ) -> Array: ...
    def outer(self, A: ArrayLike, B: ArrayLike, /) -> Array: ...

@export
def frompyfunc(
    func: Callable[..., Any],
    /,
    nin: int,
    nout: int,
    *,
    identity: Any = None,
) -> ufunc: ...
