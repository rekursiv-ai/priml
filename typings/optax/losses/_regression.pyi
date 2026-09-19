import chex

def squared_error(
    predictions: chex.Array,
    targets: chex.Array | None = None,
) -> chex.Array: ...
def l2_loss(
    predictions: chex.Array,
    targets: chex.Array | None = None,
) -> chex.Array: ...
def huber_loss(
    predictions: chex.Array,
    targets: chex.Array | None = None,
    delta: float = 1.0,
) -> chex.Array: ...
def log_cosh(
    predictions: chex.Array,
    targets: chex.Array | None = None,
) -> chex.Array: ...
def cosine_similarity(
    predictions: chex.Array,
    targets: chex.Array,
    epsilon: float = 0.0,
    axis: int | tuple[int, ...] | None = -1,
    where: chex.Array | None = None,
) -> chex.Array: ...
def cosine_distance(
    predictions: chex.Array,
    targets: chex.Array,
    epsilon: float = 0.0,
    axis: int | tuple[int, ...] | None = -1,
    where: chex.Array | None = None,
) -> chex.Array: ...
