import dataclasses

from _typeshed import Incomplete
from jaxlib.mlir import ir

from . import (
    fragmented_array as fa,
    mma_utils as mma_utils,
    utils as utils,
)
from .launch_context import LaunchContext as LaunchContext

TMEM_ROWS: int
TMEM_MAX_COLS: int
TCGEN05_SMEM_DESCRIPTOR_BIT: Incomplete
LAYOUT: Incomplete
TRANSPOSED_LAYOUT: Incomplete
ROW_LAYOUT: Incomplete
COL_LAYOUT: Incomplete
TMEM_NATIVE_LAYOUT: Incomplete

def create_instr_descriptor(
    m: int,
    n: int,
    acc_dtype,
    input_dtype,
    transpose_a: bool = False,
    transpose_b: bool = False,
    sparsity_selector: int | None = None,
) -> ir.Value: ...
def create_scaled_f8f6f4_instr_descriptor(*args, **kwargs) -> ir.Value: ...
def create_scaled_f4_instr_descriptor(*args, **kwargs) -> ir.Value: ...
def mma(
    d: TMEMRef,
    a: ir.Value | TMEMRef,
    b: ir.Value,
    *,
    a_swizzle: int = 128,
    b_swizzle: int = 128,
    a_scale: TMEMRef | None = None,
    b_scale: TMEMRef | None = None,
    a_sparse_metadata: TMEMRef | None = None,
    accumulate: ir.Value | bool = True,
    collective: bool = False,
) -> None: ...
def commit_arrive(
    barrier: utils.BarrierRef | ir.Value,
    collective: bool = False,
    ctx: LaunchContext | None = None,
) -> None: ...
def tmem_alloc_exact_ncols(ncols: int, exact: bool) -> int: ...
def tmem_alloc(
    tmem_addr: ir.Value,
    ncols: int,
    collective: bool = False,
    exact: bool = True,
) -> tuple[ir.Value, int]: ...
def tmem_dealloc(
    tmem_addr: ir.Value,
    ncols: int,
    collective: bool = False,
    exact: bool = True,
) -> None: ...
def tmem_relinquish_alloc_permit(collective: bool) -> None: ...

class TMEMLayout(fa.TiledLayout):
    def check_type(self, shape: tuple[int, ...], bitwidth: int) -> None: ...
    def cols_in_shape(self, shape: tuple[int, int], bitwidth: int) -> int: ...
    def canonicalize(self) -> TMEMLayout: ...
    def as_tiled_layout(self) -> fa.TiledLayout: ...

def tmem_default_layout(packing: int = 1) -> TMEMLayout: ...
def tmem_half_lane_layout(columns, packing: int = 1) -> TMEMLayout: ...
def tmem_m64_collective_layout(columns: int, packing: int = 1) -> TMEMLayout: ...
def fa_m64_collective_layout(columns: int) -> fa.TiledLayout: ...
def scales_layout() -> TMEMLayout: ...
def sparse_meta_layout() -> TMEMLayout: ...

@dataclasses.dataclass(frozen=True)
class TMEMRef:
    address: ir.Value
    shape: tuple[int, int]
    dtype: ir.Type
    layout: TMEMLayout
    @property
    def packing(self) -> int: ...
    def __post_init__(self) -> None: ...
    @classmethod
    def from_alloc(
        cls,
        tmem_addr_ref: ir.Value,
        shape: tuple[int, int],
        dtype,
        collective: bool | None = None,
        layout: TMEMLayout | None = None,
    ) -> TMEMRef: ...
    def slice(self, *idxs) -> TMEMRef: ...
    def load(
        self,
        layout: fa.TiledLayout | None = None,
        is_signed: bool | None = None,
    ) -> fa.FragmentedArray: ...
    def store(self, value: fa.FragmentedArray): ...

def commit_tmem() -> None: ...
def wait_load_tmem() -> None: ...
def async_copy_scales_smem_to_tmem(
    smem_ref: ir.Value,
    tmem_ref: TMEMRef,
    collective: bool = False,
) -> None: ...
def async_copy_sparse_metadata_smem_to_tmem(
    smem_ref: ir.Value,
    tmem_ref: TMEMRef,
    collective: bool = False,
) -> None: ...
def async_copy_smem_to_tmem(
    smem_ref: ir.Value,
    tmem_ref: TMEMRef,
    swizzle: int,
    collective: bool = False,
) -> None: ...
