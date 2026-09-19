import jax

def softmax(
    x: jax.Array,
    *,
    axis: int = -1,
    num_warps: int = 4,
    interpret: bool = False,
    debug: bool = False,
) -> jax.Array: ...
