from collections.abc import Callable

import dataclasses
import enum

from jax import numpy as jnp
from jax._src import (
    core as jax_core,
    dtypes as dtypes,
)
from jax._src.pallas.mosaic import core as core

class ChipVersionBase: ...

class ChipVersion(ChipVersionBase, enum.Enum):
    TPU_V2 = "v2"
    TPU_V3 = "v3"
    TPU_V4I = "v4i"
    TPU_V4 = "v4"
    TPU_V5E = "v5e"
    TPU_V5P = "v5p"
    TPU_V6E = "v6e"
    TPU_7X = "7x"

@dataclasses.dataclass(frozen=True, kw_only=True)
class SparseCoreInfo:
    num_cores: int
    num_subcores: int
    num_lanes: int
    dma_granule_size_bytes: int

@dataclasses.dataclass(frozen=True, kw_only=True)
class TpuInfo:
    chip_version: ChipVersionBase
    generation: int
    num_cores: int
    num_lanes: int
    num_sublanes: int
    mxu_column_size: int
    vmem_capacity_bytes: int
    cmem_capacity_bytes: int
    smem_capacity_bytes: int
    hbm_capacity_bytes: int
    mem_bw_bytes_per_second: int
    bf16_ops_per_second: int
    int8_ops_per_second: int
    fp8_ops_per_second: int
    int4_ops_per_second: int
    sparse_core: SparseCoreInfo | None = ...
    @property
    def is_lite(self) -> bool: ...
    @property
    def is_split_chip(self) -> bool: ...
    def is_matmul_supported(
        self,
        lhs_dtype: dtypes.DTypeLike,
        rhs_dtype: dtypes.DTypeLike,
    ) -> bool: ...
    def get_sublane_tiling(self, dtype: jnp.dtype) -> int: ...

def is_tpu_device() -> bool: ...

registry: dict[str, Callable[[], TpuInfo]]

def get_tpu_info() -> TpuInfo: ...

class Tiling(enum.Enum):
    COMPACT = ...
    SPARSE_CORE = ...
    @property
    def shape(self) -> tuple[int, ...]: ...

def infer_tiling(
    ty: jax_core.AbstractValue,
    tiling: Tiling | None = None,
) -> tuple[int | None, ...] | None: ...
