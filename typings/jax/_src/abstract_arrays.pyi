from _typeshed import Incomplete
from jax._src import (
    config as config,
    core as core,
    dtypes as dtypes,
    literals as literals,
    traceback_util as traceback_util,
)

ShapedArray: Incomplete
AbstractToken: Incomplete
abstract_token: Incomplete
canonicalize_shape: Incomplete
numpy_scalar_types: set[type]
array_types: set[type]

def masked_array_error(*args, **kwargs) -> None: ...
