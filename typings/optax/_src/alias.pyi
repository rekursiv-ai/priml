from typing import Any

from optax._src.base import GradientTransformation, MaskOrFn, ScalarOrSchedule

def adamw(
    learning_rate: ScalarOrSchedule = ...,
    b1: float = ...,
    b2: float = ...,
    eps: float = ...,
    eps_root: float = ...,
    mu_dtype: Any | None = ...,
    weight_decay: float = ...,
    mask: MaskOrFn | None = ...,
) -> GradientTransformation: ...
def adam(
    learning_rate: ScalarOrSchedule = ...,
    b1: float = ...,
    b2: float = ...,
    eps: float = ...,
    eps_root: float = ...,
    mu_dtype: Any | None = ...,
    *,
    nesterov: bool = ...,
) -> GradientTransformation: ...
def sgd(
    learning_rate: ScalarOrSchedule = ...,
    momentum: float | None = ...,
    nesterov: bool = ...,
    accumulator_dtype: Any | None = ...,
) -> GradientTransformation: ...
def adabelief(
    learning_rate: ScalarOrSchedule = ...,
    b1: float = ...,
    b2: float = ...,
    eps: float = ...,
    eps_root: float = ...,
) -> GradientTransformation: ...
def adadelta(
    learning_rate: ScalarOrSchedule = ...,
    rho: float = ...,
    eps: float = ...,
) -> GradientTransformation: ...
def adafactor(
    learning_rate: ScalarOrSchedule | None = ...,
    **kwargs: Any,
) -> GradientTransformation: ...
def adagrad(
    learning_rate: ScalarOrSchedule = ...,
    initial_accumulator_value: float = ...,
    eps: float = ...,
) -> GradientTransformation: ...
def adamax(
    learning_rate: ScalarOrSchedule = ...,
    b1: float = ...,
    b2: float = ...,
    eps: float = ...,
) -> GradientTransformation: ...
def adamaxw(
    learning_rate: ScalarOrSchedule = ...,
    b1: float = ...,
    b2: float = ...,
    eps: float = ...,
    weight_decay: float = ...,
    mask: MaskOrFn | None = ...,
) -> GradientTransformation: ...
def adan(
    learning_rate: ScalarOrSchedule = ...,
    **kwargs: Any,
) -> GradientTransformation: ...
def amsgrad(
    learning_rate: ScalarOrSchedule = ...,
    **kwargs: Any,
) -> GradientTransformation: ...
def fromage(
    learning_rate: float = ...,
    min_norm: float = ...,
) -> GradientTransformation: ...
def lamb(
    learning_rate: ScalarOrSchedule = ...,
    **kwargs: Any,
) -> GradientTransformation: ...
def lars(
    learning_rate: ScalarOrSchedule = ...,
    **kwargs: Any,
) -> GradientTransformation: ...
def lbfgs(**kwargs: Any) -> GradientTransformation: ...
def lion(
    learning_rate: ScalarOrSchedule = ...,
    **kwargs: Any,
) -> GradientTransformation: ...
def nadam(
    learning_rate: ScalarOrSchedule = ...,
    **kwargs: Any,
) -> GradientTransformation: ...
def nadamw(
    learning_rate: ScalarOrSchedule = ...,
    **kwargs: Any,
) -> GradientTransformation: ...
def noisy_sgd(
    learning_rate: ScalarOrSchedule = ...,
    **kwargs: Any,
) -> GradientTransformation: ...
def novograd(
    learning_rate: ScalarOrSchedule = ...,
    **kwargs: Any,
) -> GradientTransformation: ...
def polyak_sgd(**kwargs: Any) -> GradientTransformation: ...
def radam(
    learning_rate: ScalarOrSchedule = ...,
    **kwargs: Any,
) -> GradientTransformation: ...
def rmsprop(
    learning_rate: ScalarOrSchedule = ...,
    **kwargs: Any,
) -> GradientTransformation: ...
def rprop(learning_rate: float = ..., **kwargs: Any) -> GradientTransformation: ...
def sign_sgd(learning_rate: ScalarOrSchedule = ...) -> GradientTransformation: ...
def sm3(
    learning_rate: float = ...,
    momentum: float = ...,
) -> GradientTransformation: ...
def yogi(
    learning_rate: ScalarOrSchedule = ...,
    **kwargs: Any,
) -> GradientTransformation: ...
