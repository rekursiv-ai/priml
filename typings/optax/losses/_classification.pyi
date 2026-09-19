from optax import projections as projections

import chex
import jax

def canonicalize_axis(axis, ndim): ...
def canonicalize_axes(axes, ndim) -> tuple[int, ...]: ...
def sigmoid_binary_cross_entropy(logits, labels): ...
def binary_logistic_loss(logits, labels): ...
def hinge_loss(predictor_outputs: chex.Array, targets: chex.Array) -> chex.Array: ...
def perceptron_loss(
    predictor_outputs: chex.Numeric,
    targets: chex.Numeric,
) -> chex.Numeric: ...
def sparsemax_loss(logits: chex.Array, labels: chex.Array) -> chex.Array: ...
def binary_sparsemax_loss(logits, labels): ...
@jax.custom_jvp
def weighted_logsoftmax(x: chex.Array, weights: chex.Array) -> chex.Array: ...
def safe_softmax_cross_entropy(
    logits: chex.Array,
    labels: chex.Array,
) -> chex.Array: ...
def softmax_cross_entropy(
    logits: chex.Array,
    labels: chex.Array,
    axis: int | tuple[int, ...] | None = -1,
    where: chex.Array | None = None,
) -> chex.Array: ...
def softmax_cross_entropy_with_integer_labels(
    logits: chex.Array,
    labels: chex.Array,
    axis: int | tuple[int, ...] = -1,
    where: chex.Array | None = None,
) -> chex.Array: ...
def multiclass_logistic_loss(logits, labels): ...
def multiclass_hinge_loss(scores: chex.Array, labels: chex.Array) -> chex.Array: ...
def multiclass_perceptron_loss(
    scores: chex.Array,
    labels: chex.Array,
) -> chex.Array: ...
def poly_loss_cross_entropy(
    logits: chex.Array,
    labels: chex.Array,
    epsilon: float = 2.0,
    axis: int | tuple[int, ...] | None = -1,
    where: chex.Array | None = None,
) -> chex.Array: ...
def kl_divergence(
    log_predictions: chex.Array,
    targets: chex.Array,
    axis: int | tuple[int, ...] | None = -1,
    where: chex.Array | None = None,
) -> chex.Array: ...
def kl_divergence_with_log_targets(
    log_predictions: chex.Array,
    log_targets: chex.Array,
    axis: int | tuple[int, ...] | None = -1,
    where: chex.Array | None = None,
) -> chex.Array: ...
def convex_kl_divergence(
    log_predictions: chex.Array,
    targets: chex.Array,
    axis: int | tuple[int, ...] | None = -1,
    where: chex.Array | None = None,
) -> chex.Array: ...
def ctc_loss_with_forward_probs(
    logits: chex.Array,
    logit_paddings: chex.Array,
    labels: chex.Array,
    label_paddings: chex.Array,
    blank_id: int = 0,
    log_epsilon: float = -100000.0,
) -> tuple[chex.Array, chex.Array, chex.Array]: ...
def ctc_loss(
    logits: chex.Array,
    logit_paddings: chex.Array,
    labels: chex.Array,
    label_paddings: chex.Array,
    blank_id: int = 0,
    log_epsilon: float = -100000.0,
) -> chex.Array: ...
def sigmoid_focal_loss(
    logits: chex.Array,
    labels: chex.Array,
    alpha: float | None = None,
    gamma: float = 2.0,
) -> chex.Array: ...
def multiclass_sparsemax_loss(scores: chex.Array, labels: chex.Array) -> chex.Array: ...
