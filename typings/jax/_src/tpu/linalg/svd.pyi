from collections.abc import Sequence
from typing import Any

from jax._src import (
    api as api,
    config as config,
    core as core,
    dtypes as dtypes,
    lax as lax,
)
from jax._src.interpreters import mlir as mlir
from jax._src.typing import Array as Array

def svd(
    a: Any,
    full_matrices: bool,
    compute_uv: bool = True,
    hermitian: bool = False,
    max_iterations: int = 10,
    subset_by_index: tuple[int, int] | None = None,
) -> Sequence[Any]: ...
