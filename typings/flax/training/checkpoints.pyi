from collections.abc import (
    Callable as Callable,
    Iterable,
)
from typing import Any

import os

from _typeshed import Incomplete
from flax import (
    config as config,
    core as core,
    errors as errors,
    io as io,
    serialization as serialization,
    traverse_util as traverse_util,
)
from flax.training import orbax_utils as orbax_utils
from jax.experimental.array_serialization.serialization import (
    GlobalAsyncCheckpointManager as GlobalAsyncCheckpointManager,
)

import orbax.checkpoint as ocp

SIGNED_FLOAT_RE: Incomplete
UNSIGNED_FLOAT_RE: Incomplete
MODULE_NUM_RE: Incomplete
SCHEME_RE: Incomplete
MP_ARRAY_POSTFIX: str
MP_ARRAY_PH: str
COMMIT_SUCCESS_FILE: str
ORBAX_CKPT_FILENAME: str
ORBAX_MANIFEST_OCDBT: str
ORBAX_METADATA_FILENAME: str
type PyTree = Any
type MultiprocessArrayType = Any

class AsyncManager:
    executor: Incomplete
    save_future: Incomplete
    def __init__(self, max_workers: int = 1) -> None: ...
    def wait_previous_save(self) -> None: ...
    def save_async(self, task: Callable[[], Any]): ...

def natural_sort(file_list: Iterable[str], signed: bool = True) -> list[str]: ...
def safe_normpath(path: str) -> str: ...
def save_checkpoint(
    ckpt_dir: str | os.PathLike,
    target: PyTree,
    step: float,
    prefix: str = "checkpoint_",
    keep: int = 1,
    overwrite: bool = False,
    keep_every_n_steps: int | None = None,
    async_manager: AsyncManager | None = None,
    orbax_checkpointer: ocp.Checkpointer | None = None,
) -> str: ...
def save_checkpoint_multiprocess(
    ckpt_dir: str | os.PathLike,
    target: PyTree,
    step: float,
    prefix: str = "checkpoint_",
    keep: int = 1,
    overwrite: bool = False,
    keep_every_n_steps: int | None = None,
    async_manager: AsyncManager | None = None,
    gda_manager: GlobalAsyncCheckpointManager | None = None,
    orbax_checkpointer: ocp.Checkpointer | None = None,
) -> str: ...
def latest_checkpoint(
    ckpt_dir: str | os.PathLike,
    prefix: str = "checkpoint_",
) -> str | None: ...
def available_steps(
    ckpt_dir: str | os.PathLike,
    prefix: str = "checkpoint_",
    step_type: type = ...,
) -> list[int | float]: ...
def restore_checkpoint(
    ckpt_dir: str | os.PathLike,
    target: Any | None,
    step: float | None = None,
    prefix: str = "checkpoint_",
    parallel: bool = True,
    gda_manager: GlobalAsyncCheckpointManager | None = None,
    allow_partial_mpa_restoration: bool = False,
    orbax_checkpointer: ocp.Checkpointer | None = None,
    orbax_transforms: dict | None = None,
) -> PyTree: ...
def convert_pre_linen(params: PyTree) -> PyTree: ...
