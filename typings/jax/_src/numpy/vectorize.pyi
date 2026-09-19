from collections.abc import Callable as Callable
from typing import Any

from _typeshed import Incomplete
from jax._src import (
    api as api,
    config as config,
)
from jax._src.lax import lax as lax
from jax._src.util import set_module as set_module

export: Incomplete
type CoreDims = tuple[str, ...]
type NDArray = Any

@export
def vectorize(pyfunc, *, excluded=..., signature=None): ...
