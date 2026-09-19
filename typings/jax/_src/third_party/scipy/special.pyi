from jax._src import api as api, custom_derivatives as custom_derivatives, dtypes as dtypes
from jax._src.numpy.util import promote_args_inexact as promote_args_inexact
from jax._src.typing import Array as Array, ArrayLike as ArrayLike

@api.jit
def sincospisquaredhalf(x: Array) -> tuple[Array, Array]: ...
@custom_derivatives.custom_jvp
def fresnel(x: ArrayLike) -> tuple[Array, Array]: ...
