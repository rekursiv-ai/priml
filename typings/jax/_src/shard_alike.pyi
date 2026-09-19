from _typeshed import Incomplete
from jax._src import (
    config as config,
    core as core,
)
from jax._src.dispatch import apply_primitive as apply_primitive
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.lib.mlir import (
    dialects as dialects,
    ir as ir,
)
from jax._src.tree_util import (
    tree_flatten as tree_flatten,
    tree_unflatten as tree_unflatten,
)
from jax._src.util import safe_zip as safe_zip

def shard_alike(x, y): ...

shard_alike_p: Incomplete

def shard_alike_transpose(ct, **kwargs): ...
def shard_alike_lowering(ctx, x, y): ...
