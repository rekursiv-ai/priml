import enum

from jax._src.lib import mosaic_gpu_dialect as mgpu_dialect
from jaxlib.mlir import ir

from . import utils as utils

def tiled_memref_shape(ref: ir.Value): ...

class Dim(enum.Enum):
    K = ...
    MN = ...

def create_descriptor(
    ref: ir.Value,
    swizzle: int,
    group_size: tuple[int, int],
    logical_k_major: bool,
    large_tile: tuple[int, int] | None = None,
    mma_bytewidth_k: int = 32,
    split_const: bool = False,
): ...
def encode_addr(x: int): ...
def encode_descriptor(
    ref_arg,
    leading_byte_offset: int,
    stride_byte_offset: int,
    swizzle: int | mgpu_dialect.SwizzlingMode | None,
    const_init: int = 0,
    split_const: bool = False,
): ...
