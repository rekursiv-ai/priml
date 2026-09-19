from optax._src.base import (
    Schedule as Schedule,
    StatefulSchedule as StatefulSchedule,
)
from optax.schedules._inject import (
    InjectHyperparamsState as InjectHyperparamsState,
    InjectStatefulHyperparamsState as InjectStatefulHyperparamsState,
    WrappedSchedule as WrappedSchedule,
    inject_hyperparams as inject_hyperparams,
    inject_stateful_hyperparams as inject_stateful_hyperparams,
)
from optax.schedules._join import join_schedules as join_schedules
from optax.schedules._schedule import (
    constant_schedule as constant_schedule,
    cosine_decay_schedule as cosine_decay_schedule,
    cosine_onecycle_schedule as cosine_onecycle_schedule,
    exponential_decay as exponential_decay,
    linear_onecycle_schedule as linear_onecycle_schedule,
    linear_schedule as linear_schedule,
    piecewise_constant_schedule as piecewise_constant_schedule,
    piecewise_interpolate_schedule as piecewise_interpolate_schedule,
    polynomial_schedule as polynomial_schedule,
    sgdr_schedule as sgdr_schedule,
    warmup_constant_schedule as warmup_constant_schedule,
    warmup_cosine_decay_schedule as warmup_cosine_decay_schedule,
    warmup_exponential_decay_schedule as warmup_exponential_decay_schedule,
)
