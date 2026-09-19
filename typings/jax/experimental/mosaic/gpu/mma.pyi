from _typeshed import Incomplete
from jax.experimental.mosaic.gpu import fragmented_array as fa
from jaxlib.mlir import ir

from . import utils as utils

SUPPORTED_F8_TYPES: Incomplete

class MMALayouts:
    lhs: Incomplete
    rhs: Incomplete
    acc: Incomplete
    def __init__(self, element_type: ir.Type) -> None: ...

def mma(
    acc: fa.FragmentedArray,
    a: fa.FragmentedArray,
    b: fa.FragmentedArray,
) -> fa.FragmentedArray: ...
