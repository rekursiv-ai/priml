from collections.abc import (
    Callable as Callable,
    Sequence,
)

import enum

from jax._src import (
    api as api,
    core as core,
    dtypes as dtypes,
)
from jax._src.lax import lax as lax
from jax._src.numpy.util import promote_dtypes_inexact as promote_dtypes_inexact
from jax._src.util import canonicalize_axis as canonicalize_axis

def compute_weight_mat(
    input_size: core.DimSize,
    output_size: core.DimSize,
    scale,
    translation,
    kernel: Callable,
    antialias: bool,
): ...

class ResizeMethod(enum.Enum):
    NEAREST = 0
    LINEAR = 1
    LANCZOS3 = 2
    LANCZOS5 = 3
    CUBIC = 4
    @staticmethod
    def from_string(s: str): ...

def scale_and_translate(
    image,
    shape: core.Shape,
    spatial_dims: Sequence[int],
    scale,
    translation,
    method: str | ResizeMethod,
    antialias: bool = True,
    precision=...,
): ...
def resize(
    image,
    shape: core.Shape,
    method: str | ResizeMethod,
    antialias: bool = True,
    precision=...,
): ...
