from collections.abc import Callable as Callable

from _typeshed import Incomplete
from jax._src import (
    core as core,
    dispatch as dispatch,
    effects as effects,
    ffi as ffi,
    tree_util as tree_util,
    util as util,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)

export: Incomplete
Buffer: Incomplete
ExecutionStage: Incomplete
ExecutionContext: Incomplete

def buffer_callback(
    callback: Callable[..., None],
    result_shape_dtypes: object,
    *,
    has_side_effect: bool = False,
    vmap_method: str | None = None,
    input_output_aliases: dict[int, int] | None = None,
    command_buffer_compatible: bool = False,
): ...

buffer_callback_p: Incomplete

class BufferCallbackEffect(effects.Effect): ...
