import types

from _typeshed import Incomplete
from jax._src import (
    config as config,
    core as core,
    dispatch as dispatch,
    tree_util as tree_util,
    xla_metadata_lib as xla_metadata_lib,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.lib import xla_client as xla_client
from jax._src.lib.mlir import ir as ir

config_ext: Incomplete

class _XlaMetadataWrapper:
    def __init__(self, f, ctx) -> None: ...
    def __call__(self, *args, **kwargs): ...
    def __getattr__(self, name): ...

class XlaMetadataContextManager:
    updates: Incomplete
    def __init__(self, updates) -> None: ...
    prev: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
    def __call__(self, f): ...

def set_xla_metadata(x=None, **kwargs): ...

xla_metadata_value_p: Incomplete
