from typing import NamedTuple

from jax import (
    lax as lax,
    tree_util as tree_util,
    vmap as vmap,
)
from jax._src import core as core
from jax._src.api_util import flatten_axes as flatten_axes
from jax._src.lax.lax import DotDimensionNumbers as DotDimensionNumbers
from jax._src.typing import Array as Array
from jax._src.util import safe_zip as safe_zip

class SparseEfficiencyError(ValueError): ...
class SparseEfficiencyWarning(UserWarning): ...
class CuSparseEfficiencyWarning(SparseEfficiencyWarning): ...

type Shape = tuple[int, ...]

class SparseInfo(NamedTuple):
    shape: Shape
    indices_sorted: bool = ...
    unique_indices: bool = ...

def nfold_vmap(fun, N, *, broadcasted: bool = True, in_axes: int = 0): ...
def broadcasting_vmap(fun, in_axes: int = 0, out_axes: int = 0): ...
