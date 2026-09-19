from jax._src import util as util
from jax._src.lib.mlir import (
    ir as ir,
    passmanager as passmanager,
)
from jax._src.lib.mlir.dialects import (
    func as func,
    hlo as hlo,
)

def as_torch_kernel(fn): ...
