from _typeshed import Incomplete
from jax._src import (
    api as api,
    tree_util as tree_util,
)
from jax._src.interpreters import mlir as mlir
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import hlo as hlo

cudnn_fusion_p: Incomplete

def call_cudnn_fusion(f, *args, **kwargs): ...
def cudnn_fusion(f): ...
