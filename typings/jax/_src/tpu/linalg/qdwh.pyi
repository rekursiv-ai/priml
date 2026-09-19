from jax._src import (
    api as api,
    config as config,
    core as core,
    dtypes as dtypes,
    lax as lax,
)
from jax._src.typing import Array as Array

def qdwh(
    x,
    *,
    is_hermitian: bool = False,
    max_iterations: int | None = None,
    eps: float | None = None,
    dynamic_shape: tuple[int, int] | None = None,
): ...
