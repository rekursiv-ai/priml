from typing import Any

from optax.second_order import _base

import jax

def fisher_diag(
    negative_log_likelihood: _base.LossFn,
    params: Any,
    inputs: jax.Array,
    targets: jax.Array,
) -> jax.Array: ...
