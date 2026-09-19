from collections.abc import (
    Callable as Callable,
    Sequence,
)
from os import PathLike
from typing import Any

import os

from _typeshed import Incomplete
from jax._src import (
    array as array,
    typing as typing,
)
from jax._src.layout import Format as Format

import jax
import tensorstore as ts

Future: Incomplete
Transaction: Incomplete
logger: Incomplete

class _LimitInFlightBytes:
    def __init__(self, host_memory_bytes_limit: int) -> None: ...
    async def wait_for_bytes(self, requested_bytes): ...
    async def release_bytes(self, requested_bytes) -> None: ...

def is_tensorstore_spec_leaf(leaf: Any): ...
def merge_nested_ts_specs(dict1: dict[Any, Any], dict2: dict[Any, Any] | None): ...
def verify_tensorstore_spec(
    spec: dict[str, Any],
    arr: jax.Array | None,
    path: str | os.PathLike[str],
    ocdbt: bool,
    check_metadata: bool = True,
) -> None: ...
def get_tensorstore_spec(
    ckpt_path: str | PathLike[str],
    ocdbt: bool = True,
    process_idx: int | None = None,
    arr: jax.Array | None = None,
    driver: str = ...,
) -> dict[str, Any]: ...
async def combine_kvstores(
    combined_kvstore: dict[str, Any],
    kvstores: list[dict[str, Any]],
    context: ts.Context | dict[str, Any] = ...,
) -> None: ...
async def async_serialize(
    arr_inp,
    tensorstore_spec,
    commit_future=None,
    context=...,
    chunk_layout=...,
    primary_host: int | None = None,
    replica_id: int = 0,
    transaction: ts.Transaction | None = None,
): ...
def estimate_read_memory_footprint(
    t: ts.TensorStore,
    domain: ts.IndexDomain,
) -> int: ...
async def async_deserialize(
    in_type: jax.sharding.Sharding | Format | jax.ShapeDtypeStruct,
    tensorstore_spec: ts.Spec | dict[str, Any],
    global_shape: Sequence[int] | None = None,
    dtype=None,
    byte_limiter: _LimitInFlightBytes | None = None,
    context=...,
    chunk_layout=...,
    assume_metadata: bool = False,
): ...
