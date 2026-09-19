import types

from _typeshed import Incomplete
from jax._src import (
    config as config,
    dtypes as dtypes,
)
from jax._src.numpy import (
    array_constructors as array_constructors,
    array_creation as array_creation,
    reductions as reductions,
    ufuncs as ufuncs,
)
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

Category: Incomplete
Behavior: Incomplete

class error_checking_behavior:
    new_settings: Incomplete
    stack: Incomplete
    def __init__(
        self,
        *,
        all: Behavior | None = None,
        nan: Behavior | None = None,
        divide: Behavior | None = None,
        oob: Behavior | None = None,
    ) -> None: ...
    def __enter__(self): ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
