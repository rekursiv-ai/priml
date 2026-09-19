import enum

from _typeshed import Incomplete
from jax._src import config as config
from jax._src.lib import xla_client as xla_client
from jax._src.lib.mlir import ir as ir

import numpy as np

logger: Incomplete

def add_flag_prefixes(flag_prefixes: list[str]) -> None: ...
def clear_flag_prefixes() -> None: ...
def get_flag_prefixes() -> list[str]: ...
def custom_hook() -> str: ...

class IgnoreCallbacks(enum.IntEnum):
    NO = ...
    ALL = ...
    CUSTOM_PARTITIONING = ...

def get(
    module: ir.Module,
    devices: np.ndarray,
    compile_options: xla_client.CompileOptions,
    backend: xla_client.Client,
    compression_algorithm: str = "zstandard",
    ignore_callbacks: IgnoreCallbacks = ...,
) -> str: ...

xla_flags_to_exclude_from_cache_key: Incomplete
env_override_flags_to_exclude_from_cache_key: Incomplete
