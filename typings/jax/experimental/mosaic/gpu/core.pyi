from collections.abc import (
    Callable as Callable,
    Generator,
    Sequence,
)
from typing import Any, Generic, TypeVar

import contextlib
import dataclasses
import enum

from _typeshed import Incomplete
from jax._src import (
    dtypes as dtypes,
    lib as lib,
    sharding_impls as sharding_impls,
)
from jax._src.interpreters import mlir as mlir
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import gpu

from . import (
    dialect_lowering as dialect_lowering,
    launch_context as launch_context,
    layout_inference as layout_inference,
    layouts as layouts,
    profiler as profiler,
    tcgen05 as tcgen05,
    utils as utils,
)

cuda_root: Incomplete
PYTHON_RUNFILES: Incomplete

@contextlib.contextmanager
def artificial_shared_memory_limit(limit) -> Generator[None]: ...

FWD_COMPAT_IR_VERSION: int
c = utils.c
RUNTIME_PATH: Incomplete
libdevice_path: Incomplete

def is_nvshmem_available(): ...
def is_single_process_multi_device_topology(): ...
def supports_cross_device_collectives(): ...

mosaic_gpu_p: Incomplete
KNOWN_KERNELS: dict[bytes, bytes]
type ShapeTree = Any
type RefTree = Any
T = TypeVar("T")

@dataclasses.dataclass(frozen=True)
class Union(Generic[T]):
    members: Sequence[T]
    def __iter__(self): ...

@dataclasses.dataclass(frozen=True)
class TMABarrier:
    num_barriers: int = ...

@dataclasses.dataclass(frozen=True)
class Barrier:
    arrival_count: int
    num_barriers: int = ...
    def __post_init__(self) -> None: ...

@dataclasses.dataclass(frozen=True)
class ClusterBarrier:
    collective_dims: Sequence[gpu.Dimension]
    arrival_count: int = ...
    num_barriers: int = ...

@dataclasses.dataclass(frozen=True)
class TMEM:
    shape: tuple[int, int]
    dtype: Any
    _: dataclasses.KW_ONLY
    layout: tcgen05.TMEMLayout | None = ...
    collective: bool = ...
    packing: int | None = ...
    def __post_init__(self) -> None: ...

class LoweringSemantics(enum.Enum):
    Lane = ...
    Warpgroup = ...

@dataclasses.dataclass(frozen=True)
class _TMEMAlloc:
    addr_ref: ir.Value
    num_cols: int
    collective: bool
    def alloc(self) -> int: ...
    def dealloc(self) -> None: ...

@dataclasses.dataclass()
class _TMEMDialectAlloc:
    addr_ref: ir.Value
    shape: tuple[int, int]
    dtype: ir.Type
    packing: int
    collective: bool
    tmem_ref: ir.Value | None = ...
    def alloc(self) -> int: ...
    def dealloc(self) -> None: ...

def as_gpu_kernel(
    body,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    in_shape,
    out_shape,
    smem_scratch_shape: ShapeTree | Union[ShapeTree],
    prof_spec: profiler.ProfilerSpec | None = None,
    cluster: tuple[int, int, int] = (1, 1, 1),
    module_name: str = "unknown",
    kernel_name: str | None = None,
    ir_version: int | None = None,
    thread_semantics: LoweringSemantics = ...,
    inout_shape=(),
): ...
def as_torch_gpu_kernel(
    body,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    in_shape,
    out_shape,
    smem_scratch_shape: ShapeTree | Union[ShapeTree],
    prof_spec: profiler.ProfilerSpec | None = None,
    cluster: tuple[int, int, int] = (1, 1, 1),
    module_name: str = "unknown",
    kernel_name: str | None = None,
    thread_semantics: LoweringSemantics = ...,
    inout_shape=(),
): ...
