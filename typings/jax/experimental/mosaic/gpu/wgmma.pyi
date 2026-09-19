import dataclasses

from jaxlib.mlir import ir

import numpy as np

from . import (
    fragmented_array as fa,
    mma_utils as mma_utils,
    utils as utils,
)

c = utils.c
bytewidth = utils.bytewidth

@dataclasses.dataclass
class WGMMAAccumulator:
    def __init__(
        self,
        *,
        _value: fa.FragmentedArray,
        _original_layout: fa.FragmentedLayout,
        _sync: bool = True,
    ) -> None: ...
    @property
    def value(self) -> fa.FragmentedArray: ...
    @classmethod
    def zero(cls, m, n, dtype=None, *, is_signed: bool | None = None): ...
    @classmethod
    def from_registers(cls, registers): ...
    def tree_flatten(self): ...
    @classmethod
    def tree_unflatten(cls, aux, value): ...

def wgmma_m64(
    acc: np.ndarray,
    a,
    b_descriptor: ir.Value,
    a_transpose: bool | None,
    b_transpose: bool,
    a_k_stride: int | None,
    b_k_stride: int,
    n: int,
    swizzle: int,
    element_type: ir.Type,
): ...
def wgmma(
    acc: WGMMAAccumulator,
    a: fa.FragmentedArray | ir.Value,
    b: ir.Value,
    *,
    swizzle: int = 128,
): ...
def wgmma_fence(array: fa.FragmentedArray) -> fa.FragmentedArray: ...
