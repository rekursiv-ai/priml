from collections.abc import Callable as Callable
from typing import Any, Self

from flax import (
    core as core,
    struct as struct,
)
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT as OVERWRITE_WITH_GRADIENT

import jax
import optax

class TrainState(struct.PyTreeNode):
    step: int | jax.Array
    apply_fn: Callable[..., Any] = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)
    def apply_gradients(self, *, grads: Any, **kwargs: Any) -> Self: ...
    def replace(self, **kwargs: Any) -> Self: ...
    @classmethod
    def create(cls, *, apply_fn: Any, params: Any, tx: Any, **kwargs: Any) -> Self: ...
