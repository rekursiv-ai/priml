from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any, NamedTuple

from flax import struct as struct
from flax.typing import Array as Array

class DynamicScaleResult(NamedTuple):
    dynamic_scale: DynamicScale
    finite: Array
    aux: Any
    grad: Any

class DynamicScale(struct.PyTreeNode):
    growth_factor: float = struct.field(pytree_node=False, default=2.0)
    backoff_factor: float = struct.field(pytree_node=False, default=0.5)
    growth_interval: int = struct.field(pytree_node=False, default=2000)
    fin_steps: int = ...
    scale: float = ...
    minimum_scale: float | None = ...
    def value_and_grad(
        self,
        fun: Callable[..., Any],
        argnums: int | Sequence[int] = 0,
        has_aux: bool = False,
        axis_name: str | None = None,
    ) -> Callable[..., DynamicScaleResult]: ...
