from collections.abc import (
    Callable as Callable,
    Generator,
    Sequence,
)
from typing import Any

import contextlib
import dataclasses
import enum
import functools

from _typeshed import Incomplete
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import gpu

from . import (
    profiler as profiler,
    utils as utils,
)

TMA_DESCRIPTOR_BYTES: int
TMA_DESCRIPTOR_ALIGNMENT: int
TMAReductionOp: Incomplete
COLLECTIVE_METADATA_SIZE: int
COLLECTIVE_ATTR: str
KERNEL_ARG_ID_ATTR: str
ORIGINAL_KERNEL_ARG_ATTR: str
DEVICE_ID_ATTR: str

def uses_collective_metadata(module): ...

c = utils.c

class GlobalBroadcast: ...

GLOBAL_BROADCAST: Incomplete

@dataclasses.dataclass(frozen=True)
class MemRefTransform:
    def apply(self, ref: ir.Value) -> ir.Value: ...
    def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]: ...
    def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]: ...
    def transform_strides(self, shape: Sequence[int]) -> tuple[int, ...]: ...
    def batch(self, leading_rank: int) -> MemRefTransform: ...
    def to_attr(self) -> ir.Attribute: ...

class Rounding(enum.Enum):
    UP = ...
    DOWN = ...

@dataclasses.dataclass(frozen=True)
class TileTransform(MemRefTransform):
    tiling: tuple[int, ...]
    rounding: Rounding | None = ...
    def __post_init__(self) -> None: ...
    def apply(self, ref: ir.Value) -> ir.Value: ...
    def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]: ...
    def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]: ...
    def transform_strides(self, strides: Sequence[int]) -> tuple[int, ...]: ...
    def batch(self, leading_rank: int) -> MemRefTransform: ...
    def to_attr(self) -> ir.Attribute: ...

@dataclasses.dataclass(frozen=True)
class SwizzleTransform(MemRefTransform):
    swizzle: int
    def to_attr(self) -> ir.Attribute: ...

@dataclasses.dataclass(frozen=True)
class TransposeTransform(MemRefTransform):
    permutation: tuple[int, ...]
    def __post_init__(self) -> None: ...
    def apply(self, ref: ir.Value) -> ir.Value: ...
    def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]: ...
    def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]: ...
    def transform_strides(self, strides: Sequence[int]) -> tuple[int, ...]: ...
    def batch(self, leading_rank: int) -> MemRefTransform: ...
    def to_attr(self) -> ir.Attribute: ...

@dataclasses.dataclass(frozen=True)
class CollapseLeadingIndicesTransform(MemRefTransform):
    strides: tuple[int, ...]
    @functools.cached_property
    def common_stride(self) -> int: ...
    def apply(self, ref: ir.Value) -> ir.Value: ...
    def transform_index(self, idx: Sequence[ir.Value]) -> tuple[ir.Value, ...]: ...
    def transform_shape(self, shape: Sequence[int]) -> tuple[int, ...]: ...
    def batch(self, leading_rank: int) -> MemRefTransform: ...

OnDeviceProfiler = profiler.OnDeviceProfiler
MOSAIC_GPU_SMEM_ALLOC_ATTR: str

class Scratch:
    next_offset: int
    host_init: list[Callable[[ir.Value], None]]
    def __init__(self, gpu_launch_op: gpu.LaunchOp) -> None: ...
    def device_ptr(self) -> ir.Value: ...
    def finalize_size(self) -> None: ...

class _DefaultPredicate: ...

class AsyncCopyImplementation(enum.Enum):
    TMA = ...
    CP_ASYNC = ...

@dataclasses.dataclass()
class LaunchContext:
    module: ir.Module
    scratch: Scratch
    cluster_size: tuple[int, int, int]
    profiler: OnDeviceProfiler | None = ...
    device_collective_metadata: ir.Value | None = ...
    host_collective_metadata: ir.Value | None = ...
    num_peers: int = ...
    tma_descriptors: dict[
        tuple[
            ir.Value,
            tuple[int, ...],
            int | None,
            tuple[MemRefTransform, ...],
            Any,
            int,
        ],
        ir.Value,
    ] = ...
    is_device_collective: bool = ...
    @contextlib.contextmanager
    def named_region(self, *args, **kwargs) -> Generator[None]: ...
    def async_copy(
        self,
        *,
        src_ref: ir.Value,
        dst_ref: ir.Value,
        gmem_slice: Any = (),
        gmem_transform: MemRefTransform | tuple[MemRefTransform, ...] = (),
        gmem_peer_id: int | ir.Value | GlobalBroadcast | None = None,
        barrier: utils.BarrierRef | None = None,
        swizzle: int | None = None,
        arrive: bool | None = None,
        collective: Sequence[gpu.Dimension] | gpu.Dimension | None = None,
        partitioned: int | None = None,
        predicate: ir.Value | _DefaultPredicate | None = ...,
        reduction_op: TMAReductionOp | None = None,
        implementation: AsyncCopyImplementation = ...,
    ): ...
    def async_prefetch(
        self,
        *,
        gmem_ref: ir.Value,
        gmem_slice: Any = (),
        gmem_transform: MemRefTransform | tuple[MemRefTransform, ...] = (),
        gmem_peer_id: int | ir.Value | None = None,
        swizzle: int | None = None,
        collective: Sequence[gpu.Dimension] | gpu.Dimension | None = None,
        partitioned: int | None = None,
        predicate: ir.Value | _DefaultPredicate | None = ...,
    ): ...
    def await_async_copy(
        self,
        allow_groups: int,
        await_read_only: bool = False,
        scope: utils.ThreadSubset = ...,
    ): ...
    def await_cp_async_copy(self, allow_groups: int): ...
    def to_remote(
        self,
        ref: ir.Value,
        peer: ir.Value,
        *,
        _kernel_arg_idx: int | None = None,
        on_host: bool = False,
    ): ...
    def to_remote_multicast(self, ref: ir.Value): ...
    def device_id(self, on_host: bool = False) -> ir.Value: ...

class ReplicationError(Exception): ...
