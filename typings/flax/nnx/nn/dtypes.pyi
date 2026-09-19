import typing as tp

from flax.typing import Dtype as Dtype

T = tp.TypeVar("T", bound=tuple)

def canonicalize_dtype(
    *args,
    dtype: Dtype | None = None,
    inexact: bool = True,
) -> Dtype: ...
def promote_dtype(args: T, /, *, dtype=None, inexact: bool = True) -> T: ...
