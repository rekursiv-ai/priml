from collections.abc import Callable as Callable
from typing import Any

from _typeshed import Incomplete
from jax import core

type ScanAxis = int | None

class _Broadcast: ...

broadcast: Incomplete

def build_shaped_array(x, batch_dim: bool = False) -> core.ShapedArray: ...
def scan(
    fn: Callable[..., Any],
    in_axes: Any,
    out_axes: Any,
    length: int | None = None,
    reverse: bool = False,
    unroll: int = 1,
    _split_transpose: bool = False,
    check_constancy_invariants: bool = True,
): ...
