from typing import Any, Protocol

import dataclasses

from _typeshed import Incomplete
from jax._src.lib import mosaic_gpu_dialect as mgpu
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import (
    arith as arith,
    builtin as builtin,
    func as func,
    gpu as gpu,
    memref as memref,
    nvvm as nvvm,
    scf as scf,
    vector as vector,
)

from . import (
    fragmented_array as fa,
    inference_utils as inference_utils,
    launch_context as lc,
    tcgen05 as tcgen05,
    utils as utils,
    wgmma as wgmma,
)

@dataclasses.dataclass()
class LoweringContext:
    launch_context: lc.LaunchContext | None
    single_thread_per_block_predicate: ir.Value | None
    single_thread_per_warpgroup_predicate: ir.Value | None
    single_warp_per_block_predicate: ir.Value | None
    auto_barriers: bool
    smem_requested_bytes: int
    lowered_operations: set[ir.Operation | ir.OpView] = ...
    is_collective_kernel: bool | None = ...
    def check_collective(self, op: ir.OpView) -> None: ...
    def lower_op(self, op: ir.OpView): ...

class Recursed: ...

RECURSED: Incomplete
MlirLoweringRuleResult: Incomplete
MlirLoweringRule: Incomplete

def fragmented_array_to_ir(
    fragmented_array: fa.FragmentedArray,
    ty: ir.Type,
) -> ir.Value: ...
def wrap_transformed_memref(
    transformed_memref: ir.Value,
    logical_type: ir.Type,
    transforms: ir.ArrayAttr,
) -> ir.Value: ...
def unwrap_transformed_memref(
    ref: ir.Value,
    expected_transforms: ir.ArrayAttr,
) -> ir.Value: ...

class _Transfer(Protocol):
    def __call__(self, optimized: bool) -> Any: ...

def pprint_layout(v: fa.FragmentedArray | tcgen05.TMEMRef) -> str: ...
def swizzle_and_transforms_from_transforms_attr(
    transforms: ir.ArrayAttr,
) -> tuple[mgpu.SwizzlingMode, tuple[lc.MemRefTransform, ...]]: ...
def tile_offset(
    offsets: tuple[int, ...],
    tiling: tuple[int, ...],
) -> tuple[int, ...]: ...
def tile_strides(
    strides: tuple[int, ...],
    tiling: tuple[int, ...],
) -> tuple[int, ...]: ...
def transform_type(
    ref_ty: ir.MemRefType,
    transforms: tuple[lc.MemRefTransform, ...],
) -> ir.MemRefType: ...

CMPI_IMPLS: Incomplete
CMPF_IMPLS: Incomplete

def lower_mgpu_dialect(
    module: ir.Module,
    launch_context: lc.LaunchContext | None,
    auto_barriers: bool = True,
): ...
