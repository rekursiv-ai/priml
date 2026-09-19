from collections.abc import Callable as Callable

import chex

def ranking_softmax_loss(
    logits: chex.Array,
    labels: chex.Array,
    *,
    where: chex.Array | None = None,
    weights: chex.Array | None = None,
    reduce_fn: Callable[..., chex.Array] | None = ...,
) -> chex.Array: ...
