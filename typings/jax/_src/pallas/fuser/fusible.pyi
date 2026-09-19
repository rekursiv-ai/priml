from typing import Any

from _typeshed import Incomplete
from jax._src import (
    api_util as api_util,
    tree_util as tree_util,
    util as util,
)
from jax._src.interpreters import (
    batching as batching,
    mlir as mlir,
)
from jax._src.traceback_util import api_boundary as api_boundary

fusible_p: Incomplete

def fusible(f=None, *, output_fusion_prefix: Any = True): ...
@fusible_p.def_impl
def _(*consts_and_args, jaxpr, num_consts, **_): ...
