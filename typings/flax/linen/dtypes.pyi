from typing import Any, TypeVar

from flax.typing import Dtype as Dtype

T = TypeVar("T", bound=tuple)

def canonicalize_dtype(
    *args,
    dtype: Dtype | None = None,
    inexact: bool = True,
) -> Dtype: ...
def promote_dtype(*args, dtype=None, inexact: bool = True) -> list[Any]: ...
