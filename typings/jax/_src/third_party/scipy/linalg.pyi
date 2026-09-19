from collections.abc import Callable as Callable
from jax._src import api as api, dtypes as dtypes, lax as lax
from jax._src.numpy.linalg import norm as norm
from jax._src.scipy.linalg import rsf2csf as rsf2csf, schur as schur
from jax._src.typing import Array as Array, ArrayLike as ArrayLike

def funm(A: ArrayLike, func: Callable[[Array], Array], disp: bool = True) -> Array | tuple[Array, Array]: ...
