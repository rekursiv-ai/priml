from _typeshed import Incomplete
from jax._src import dtypes as dtypes
from jax._src.numpy import asarray as asarray, broadcast_arrays as broadcast_arrays, empty as empty, searchsorted as searchsorted, where as where, zeros as zeros
from jax._src.numpy.util import check_arraylike as check_arraylike, promote_dtypes_inexact as promote_dtypes_inexact
from jax._src.tree_util import register_pytree_node as register_pytree_node

class RegularGridInterpolator:
    method: Incomplete
    bounds_error: Incomplete
    fill_value: Incomplete
    grid: Incomplete
    values: Incomplete
    def __init__(self, points, values, method: str = 'linear', bounds_error: bool = False, fill_value=...) -> None: ...
    def __call__(self, xi, method=None): ...
