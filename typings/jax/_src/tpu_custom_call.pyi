from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any, TypedDict

import dataclasses
import enum

from _typeshed import Incomplete
from jax._src import (
    api as api,
    config as config,
    core as core,
    dispatch as dispatch,
    sharding_impls as sharding_impls,
)
from jax._src.cloud_tpu_init import is_cloud_tpu_older_than as is_cloud_tpu_older_than
from jax._src.frozen_dict import FrozenDict as FrozenDict
from jax._src.interpreters import (
    batching as batching,
    mlir as mlir,
)
from jax._src.lib import tpu as tpu
from jaxlib.mlir import ir

FLAGS: Incomplete

def get_ir_version(ctx: mlir.LoweringRuleContext) -> int | None: ...

tpu_custom_call_p: Incomplete

def tpu_custom_call_batcher(axis_data, args, dims, **kwargs): ...

class MemorySpace(enum.Enum):
    HBM = ...
    VMEM = ...
    SEMAPHORE_MEM = ...
    SMEM = ...
    HOST = ...
    SC_SCALAR_SEMAPHORE_MEM = ...
    @property
    def color(self) -> int: ...

class CostEstimate(TypedDict):
    flops: int
    transcendentals: int
    bytes_accessed: int
    remote_bytes_transferred: int

class TpuSideEffectType(enum.Enum):
    PURE = "pure"
    DATAFLOW_SIDE_EFFECTING = "dataflow_side_effecting"
    SIDE_EFFECTING = "side_effecting"

class Tiling(enum.Enum):
    COMPACT = "TILING_COMPACT"
    SPARSE_CORE = "TILING_SPARSE_CORE"

@dataclasses.dataclass(frozen=True)
class CustomCallBackendConfig:
    lowered_module_asm: bytes
    has_communication: bool
    collective_id: int | None
    device_type: str | None
    cost_estimate: CostEstimate | None
    needs_hlo_passes: bool
    needs_layout_passes: bool
    vmem_limit_bytes: int | None
    flags: dict[str, bool | int | float] | None
    allow_input_fusion: Sequence[bool] | None
    serialization_format: int | None
    internal_scratch_in_bytes: int | None
    output_memory_spaces: tuple[MemorySpace | None, ...] | None
    disable_bounds_checks: bool
    disable_semaphore_checks: bool
    active_core_count: int | None
    input_memory_spaces: tuple[MemorySpace | None, ...] | None
    skip_device_barrier: bool
    shape_invariant_numerics: bool
    tiling: Tiling | None = ...
    def __post_init__(self) -> None: ...
    def to_json(self) -> bytes: ...

def lower_module_to_custom_call(
    ctx: mlir.LoweringRuleContext,
    *in_nodes: ir.Value,
    module: ir.Module,
    out_type: Any,
    kernel_name: str,
    cost_estimate: CostEstimate | None,
    vmem_limit_bytes: int | None,
    flags: dict[str, bool | int | float] | None,
    allow_input_fusion: Sequence[bool] | None,
    input_output_aliases: tuple[tuple[int, int], ...],
    internal_scratch_in_bytes: int | None,
    collective_id: int | None,
    has_side_effects: bool | TpuSideEffectType,
    serialization_format: int | None,
    output_memory_spaces: tuple[MemorySpace | None, ...] | None,
    disable_bounds_checks: bool = False,
    disable_semaphore_checks: bool = False,
    input_memory_spaces: tuple[MemorySpace | None, ...] | None,
    metadata: Any | None = None,
    skip_device_barrier: bool = False,
    allow_collective_id_without_custom_barrier: bool = False,
    shape_invariant_numerics: bool = False,
    needs_layout_passes: bool | None = None,
    tiling: Tiling | None = None,
) -> Sequence[ir.Value]: ...
def as_tpu_kernel(
    module: ir.Module,
    out_type: Any,
    *,
    cost_estimate: CostEstimate | None = None,
    kernel_name: str | None = None,
    vmem_limit_bytes: int | None = None,
    flags: dict[str, bool | int | float] | None = None,
    allow_input_fusion: Sequence[bool] | None = None,
    input_output_aliases: tuple[tuple[int, int], ...] = (),
    internal_scratch_in_bytes: int | None = None,
    collective_id: int | None = None,
    has_side_effects: TpuSideEffectType = ...,
    serialization_format: int | None = 1,
    output_memory_spaces: tuple[MemorySpace | None, ...] | None = None,
    disable_bounds_checks: bool = False,
    disable_semaphore_checks: bool = False,
    input_memory_spaces: tuple[MemorySpace | None, ...] | None = None,
    shape_invariant_numerics: bool = False,
    needs_layout_passes: bool | None = None,
    metadata: Any | None = None,
    tiling: Tiling | None = None,
    _ir_version: int | None = None,
) -> Callable[..., Any]: ...
def lowered_as_tpu_kernel(
    lowered_module: ir.Module,
    out_type: Any,
    *,
    collective_id: int | None = None,
    cost_estimate: CostEstimate | None = None,
    needs_hlo_passes: bool = False,
    needs_layout_passes: bool = False,
    has_communication: bool = False,
    has_side_effects: bool | TpuSideEffectType = False,
    has_custom_barrier: bool = False,
    kernel_name: str | None = None,
    vmem_limit_bytes: int | None = None,
    flags: dict[str, bool | int | float] | None = None,
    allow_input_fusion: Sequence[bool] | None = None,
    input_output_aliases: tuple[tuple[int, int], ...] = (),
    serialization_format: int | None = None,
    internal_scratch_in_bytes: int | None = None,
    disable_bounds_checks: bool = False,
    disable_semaphore_checks: bool = False,
    metadata: Any | None = None,
    allow_collective_id_without_custom_barrier: bool = False,
) -> Callable[..., Any]: ...
