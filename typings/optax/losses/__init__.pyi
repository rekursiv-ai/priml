from optax.losses._classification import (
    convex_kl_divergence as convex_kl_divergence,
    ctc_loss as ctc_loss,
    ctc_loss_with_forward_probs as ctc_loss_with_forward_probs,
    hinge_loss as hinge_loss,
    kl_divergence as kl_divergence,
    kl_divergence_with_log_targets as kl_divergence_with_log_targets,
    multiclass_hinge_loss as multiclass_hinge_loss,
    multiclass_perceptron_loss as multiclass_perceptron_loss,
    multiclass_sparsemax_loss as multiclass_sparsemax_loss,
    perceptron_loss as perceptron_loss,
    poly_loss_cross_entropy as poly_loss_cross_entropy,
    safe_softmax_cross_entropy as safe_softmax_cross_entropy,
    sigmoid_binary_cross_entropy as sigmoid_binary_cross_entropy,
    sigmoid_focal_loss as sigmoid_focal_loss,
    softmax_cross_entropy as softmax_cross_entropy,
    softmax_cross_entropy_with_integer_labels as softmax_cross_entropy_with_integer_labels,
    sparsemax_loss as sparsemax_loss,
)
from optax.losses._fenchel_young import (
    make_fenchel_young_loss as make_fenchel_young_loss,
)
from optax.losses._ranking import ranking_softmax_loss as ranking_softmax_loss
from optax.losses._regression import (
    cosine_distance as cosine_distance,
    cosine_similarity as cosine_similarity,
    huber_loss as huber_loss,
    l2_loss as l2_loss,
    log_cosh as log_cosh,
    squared_error as squared_error,
)
from optax.losses._segmentation import (
    binary_dice_loss as binary_dice_loss,
    dice_loss as dice_loss,
    multiclass_generalized_dice_loss as multiclass_generalized_dice_loss,
)
from optax.losses._self_supervised import (
    ntxent as ntxent,
    triplet_margin_loss as triplet_margin_loss,
)
from optax.losses._smoothing import smooth_labels as smooth_labels
