from _typeshed import Incomplete
from optax import assignment as assignment
from optax._src.alias import (
    MaskOrFn as MaskOrFn,
    adabelief as adabelief,
    adadelta as adadelta,
    adafactor as adafactor,
    adagrad as adagrad,
    adam as adam,
    adamax as adamax,
    adamaxw as adamaxw,
    adamw as adamw,
    adan as adan,
    amsgrad as amsgrad,
    fromage as fromage,
    lamb as lamb,
    lars as lars,
    lbfgs as lbfgs,
    lion as lion,
    nadam as nadam,
    nadamw as nadamw,
    noisy_sgd as noisy_sgd,
    novograd as novograd,
    polyak_sgd as polyak_sgd,
    radam as radam,
    rmsprop as rmsprop,
    rprop as rprop,
    sgd as sgd,
    sign_sgd as sign_sgd,
    sm3 as sm3,
    yogi as yogi,
)
from optax._src.base import (
    EmptyState as EmptyState,
    GradientTransformation as GradientTransformation,
    GradientTransformationExtraArgs as GradientTransformationExtraArgs,
    OptState as OptState,
    Params as Params,
    ScalarOrSchedule as ScalarOrSchedule,
    Schedule as Schedule,
    TransformInitFn as TransformInitFn,
    TransformUpdateExtraArgsFn as TransformUpdateExtraArgsFn,
    TransformUpdateFn as TransformUpdateFn,
    Updates as Updates,
    identity as identity,
    set_to_zero as set_to_zero,
    stateless as stateless,
    stateless_with_tree_map as stateless_with_tree_map,
)
from optax._src.factorized import (
    FactoredState as FactoredState,
    scale_by_factored_rms as scale_by_factored_rms,
)
from optax._src.linear_algebra import (
    global_norm as global_norm,
    matrix_inverse_pth_root as matrix_inverse_pth_root,
    nnls as nnls,
    power_iteration as power_iteration,
)
from optax._src.linesearch import (
    ScaleByBacktrackingLinesearchState as ScaleByBacktrackingLinesearchState,
    ScaleByZoomLinesearchState as ScaleByZoomLinesearchState,
    ZoomLinesearchInfo as ZoomLinesearchInfo,
    scale_by_backtracking_linesearch as scale_by_backtracking_linesearch,
    scale_by_zoom_linesearch as scale_by_zoom_linesearch,
)
from optax._src.lookahead import (
    LookaheadParams as LookaheadParams,
    LookaheadState as LookaheadState,
    lookahead as lookahead,
)
from optax._src.numerics import (
    safe_increment as safe_increment,
    safe_int32_increment as safe_int32_increment,
    safe_norm as safe_norm,
    safe_root_mean_squares as safe_root_mean_squares,
)
from optax._src.transform import (
    ApplyEvery as ApplyEvery,
    ScaleByAdaDeltaState as ScaleByAdaDeltaState,
    ScaleByAdamState as ScaleByAdamState,
    ScaleByAdanState as ScaleByAdanState,
    ScaleByAmsgradState as ScaleByAmsgradState,
    ScaleByBeliefState as ScaleByBeliefState,
    ScaleByLBFGSState as ScaleByLBFGSState,
    ScaleByLionState as ScaleByLionState,
    ScaleByNovogradState as ScaleByNovogradState,
    ScaleByRmsState as ScaleByRmsState,
    ScaleByRpropState as ScaleByRpropState,
    ScaleByRssState as ScaleByRssState,
    ScaleByRStdDevState as ScaleByRStdDevState,
    ScaleByScheduleState as ScaleByScheduleState,
    ScaleBySM3State as ScaleBySM3State,
    apply_every as apply_every,
    centralize as centralize,
    scale as scale,
    scale_by_adadelta as scale_by_adadelta,
    scale_by_adam as scale_by_adam,
    scale_by_adamax as scale_by_adamax,
    scale_by_adan as scale_by_adan,
    scale_by_amsgrad as scale_by_amsgrad,
    scale_by_belief as scale_by_belief,
    scale_by_lbfgs as scale_by_lbfgs,
    scale_by_lion as scale_by_lion,
    scale_by_novograd as scale_by_novograd,
    scale_by_param_block_norm as scale_by_param_block_norm,
    scale_by_param_block_rms as scale_by_param_block_rms,
    scale_by_polyak as scale_by_polyak,
    scale_by_radam as scale_by_radam,
    scale_by_rms as scale_by_rms,
    scale_by_rprop as scale_by_rprop,
    scale_by_rss as scale_by_rss,
    scale_by_schedule as scale_by_schedule,
    scale_by_sign as scale_by_sign,
    scale_by_sm3 as scale_by_sm3,
    scale_by_stddev as scale_by_stddev,
    scale_by_trust_ratio as scale_by_trust_ratio,
    scale_by_yogi as scale_by_yogi,
)
from optax._src.update import (
    apply_updates as apply_updates,
    incremental_update as incremental_update,
    periodic_update as periodic_update,
)
from optax._src.utils import (
    multi_normal as multi_normal,
    scale_gradient as scale_gradient,
    value_and_grad_from_state as value_and_grad_from_state,
)
from optax.contrib import (
    DifferentiallyPrivateAggregateState as DifferentiallyPrivateAggregateState,
    differentially_private_aggregate as differentially_private_aggregate,
    dpsgd as dpsgd,
)
from optax.transforms._clipping import clip_by_global_norm as clip_by_global_norm
from optax.transforms._combining import chain as chain

__all__ = [
    "AdaptiveGradClipState",
    "AddDecayedWeightsState",
    "AddNoiseState",
    "ApplyEvery",
    "ApplyIfFiniteState",
    "ClipByGlobalNormState",
    "ClipState",
    "ConditionallyMaskState",
    "ConditionallyTransformState",
    "DifferentiallyPrivateAggregateState",
    "EmaState",
    "EmptyState",
    "FactoredState",
    "GradientTransformation",
    "GradientTransformationExtraArgs",
    "InjectHyperparamsState",
    "LookaheadParams",
    "LookaheadState",
    "MaskOrFn",
    "MaskedState",
    "MultiSteps",
    "MultiStepsState",
    "MultiTransformState",
    "NonNegativeParamsState",
    "OptState",
    "Params",
    "PartitionState",
    "ScalarOrSchedule",
    "ScaleByAdaDeltaState",
    "ScaleByAdamState",
    "ScaleByAdanState",
    "ScaleByAmsgradState",
    "ScaleByBacktrackingLinesearchState",
    "ScaleByBeliefState",
    "ScaleByLBFGSState",
    "ScaleByLionState",
    "ScaleByNovogradState",
    "ScaleByRStdDevState",
    "ScaleByRmsState",
    "ScaleByRpropState",
    "ScaleByRssState",
    "ScaleBySM3State",
    "ScaleByScheduleState",
    "ScaleByTrustRatioState",
    "ScaleByZoomLinesearchState",
    "ScaleState",
    "Schedule",
    "ShouldSkipUpdateFunction",
    "TraceState",
    "TransformInitFn",
    "TransformUpdateExtraArgsFn",
    "TransformUpdateFn",
    "Updates",
    "ZeroNansState",
    "ZoomLinesearchInfo",
    "adabelief",
    "adadelta",
    "adafactor",
    "adagrad",
    "adam",
    "adamax",
    "adamaxw",
    "adamw",
    "adan",
    "adaptive_grad_clip",
    "add_decayed_weights",
    "add_noise",
    "amsgrad",
    "apply_every",
    "apply_if_finite",
    "apply_updates",
    "assignment",
    "centralize",
    "chain",
    "clip",
    "clip_by_block_rms",
    "clip_by_global_norm",
    "conditionally_mask",
    "conditionally_transform",
    "constant_schedule",
    "convex_kl_divergence",
    "cosine_decay_schedule",
    "cosine_distance",
    "cosine_onecycle_schedule",
    "cosine_similarity",
    "ctc_loss",
    "ctc_loss_with_forward_probs",
    "differentially_private_aggregate",
    "dpsgd",
    "ema",
    "exponential_decay",
    "flatten",
    "fromage",
    "global_norm",
    "hinge_loss",
    "huber_loss",
    "identity",
    "incremental_update",
    "inject_hyperparams",
    "join_schedules",
    "keep_params_nonnegative",
    "kl_divergence",
    "l2_loss",
    "lamb",
    "lars",
    "lbfgs",
    "linear_onecycle_schedule",
    "linear_schedule",
    "lion",
    "log_cosh",
    "lookahead",
    "masked",
    "matrix_inverse_pth_root",
    "multi_normal",
    "multi_transform",
    "nadam",
    "nadamw",
    "nnls",
    "noisy_sgd",
    "novograd",
    "ntxent",
    "partition",
    "per_example_global_norm_clip",
    "per_example_layer_norm_clip",
    "periodic_update",
    "piecewise_constant_schedule",
    "piecewise_interpolate_schedule",
    "polyak_sgd",
    "polynomial_schedule",
    "power_iteration",
    "radam",
    "rmsprop",
    "rprop",
    "safe_increment",
    "safe_int32_increment",
    "safe_norm",
    "safe_root_mean_squares",
    "scale",
    "scale_by_adadelta",
    "scale_by_adam",
    "scale_by_adamax",
    "scale_by_adan",
    "scale_by_amsgrad",
    "scale_by_backtracking_linesearch",
    "scale_by_belief",
    "scale_by_factored_rms",
    "scale_by_lbfgs",
    "scale_by_lion",
    "scale_by_novograd",
    "scale_by_param_block_norm",
    "scale_by_param_block_rms",
    "scale_by_polyak",
    "scale_by_radam",
    "scale_by_rms",
    "scale_by_rprop",
    "scale_by_rss",
    "scale_by_schedule",
    "scale_by_sign",
    "scale_by_sm3",
    "scale_by_stddev",
    "scale_by_trust_ratio",
    "scale_by_yogi",
    "scale_by_zoom_linesearch",
    "scale_gradient",
    "set_to_zero",
    "sgd",
    "sgdr_schedule",
    "sigmoid_binary_cross_entropy",
    "sign_sgd",
    "skip_large_updates",
    "skip_not_finite",
    "sm3",
    "smooth_labels",
    "softmax_cross_entropy",
    "softmax_cross_entropy_with_integer_labels",
    "stateless",
    "stateless_with_tree_map",
    "trace",
    "value_and_grad_from_state",
    "warmup_cosine_decay_schedule",
    "warmup_exponential_decay_schedule",
    "yogi",
    "zero_nans",
]

adaptive_grad_clip: Incomplete
AdaptiveGradClipState = EmptyState
clip: Incomplete
clip_by_block_rms: Incomplete
ClipByGlobalNormState = EmptyState
ClipState = EmptyState
per_example_global_norm_clip: Incomplete
per_example_layer_norm_clip: Incomplete
keep_params_nonnegative: Incomplete
NonNegativeParamsState: Incomplete
zero_nans: Incomplete
ZeroNansState: Incomplete
partition: Incomplete
PartitionState: Incomplete
multi_transform: Incomplete
MultiTransformState: Incomplete
trace: Incomplete
TraceState: Incomplete
ema: Incomplete
EmaState: Incomplete
add_noise: Incomplete
AddNoiseState: Incomplete
add_decayed_weights: Incomplete
AddDecayedWeightsState = EmptyState
ScaleByTrustRatioState = EmptyState
ScaleState = EmptyState
apply_if_finite: Incomplete
ApplyIfFiniteState: Incomplete
conditionally_mask: Incomplete
conditionally_transform: Incomplete
ConditionallyMaskState: Incomplete
ConditionallyTransformState: Incomplete
flatten: Incomplete
masked: Incomplete
MaskedState: Incomplete
MultiSteps: Incomplete
MultiStepsState: Incomplete
ShouldSkipUpdateFunction: Incomplete
skip_large_updates: Incomplete
skip_not_finite: Incomplete
constant_schedule: Incomplete
cosine_decay_schedule: Incomplete
cosine_onecycle_schedule: Incomplete
exponential_decay: Incomplete
inject_hyperparams: Incomplete
InjectHyperparamsState: Incomplete
join_schedules: Incomplete
linear_onecycle_schedule: Incomplete
linear_schedule: Incomplete
piecewise_constant_schedule: Incomplete
piecewise_interpolate_schedule: Incomplete
polynomial_schedule: Incomplete
sgdr_schedule: Incomplete
warmup_cosine_decay_schedule: Incomplete
warmup_exponential_decay_schedule: Incomplete
convex_kl_divergence: Incomplete
cosine_distance: Incomplete
cosine_similarity: Incomplete
ctc_loss: Incomplete
ctc_loss_with_forward_probs: Incomplete
hinge_loss: Incomplete
huber_loss: Incomplete
kl_divergence: Incomplete
l2_loss: Incomplete
log_cosh: Incomplete
ntxent: Incomplete
sigmoid_binary_cross_entropy: Incomplete
smooth_labels: Incomplete
softmax_cross_entropy: Incomplete
softmax_cross_entropy_with_integer_labels: Incomplete
