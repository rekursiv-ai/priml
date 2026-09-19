from typing import Any

from optax.second_order import _base

import jax

def hvp(
    loss: _base.LossFn,
    v: jax.Array,
    params: Any,
    inputs: jax.Array,
    targets: jax.Array,
) -> jax.Array: ...
def hessian_diag(
    loss: _base.LossFn,
    params: Any,
    inputs: jax.Array,
    targets: jax.Array,
) -> jax.Array: ...
