from jax._src.api import jit as jit
from jax._src.numpy import lax_numpy as lax_numpy
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
)

def trapezoid(
    y: ArrayLike,
    x: ArrayLike | None = None,
    dx: ArrayLike = 1.0,
    axis: int = -1,
) -> Array: ...
