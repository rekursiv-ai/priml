import chex

def dice_loss(
    predictions: chex.Array,
    targets: chex.Array,
    *,
    class_weights: chex.Array | None = None,
    smooth: float = 1.0,
    apply_softmax: bool = True,
    reduction: str = "mean",
    ignore_background: bool = False,
    axis: chex.Array | None = None,
) -> chex.Array: ...
def multiclass_generalized_dice_loss(
    predictions: chex.Array,
    targets: chex.Array,
    *,
    smooth: float = 1.0,
    apply_softmax: bool = True,
    ignore_background: bool = False,
) -> chex.Array: ...
def binary_dice_loss(
    predictions: chex.Array,
    targets: chex.Array,
    *,
    smooth: float = 1.0,
    apply_sigmoid: bool = True,
) -> chex.Array: ...
