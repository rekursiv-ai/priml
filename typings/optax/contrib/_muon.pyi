from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

from _typeshed import Incomplete
from optax._src import (
    alias as alias,
    base as base,
    combine as combine,
    numerics as numerics,
    transform as transform,
    utils as utils,
)

import chex
import jax

ReshapeFn: Incomplete

class MuonDimensionNumbers(NamedTuple):
    reduction_axis: Sequence[int] | int = ...
    output_axis: Sequence[int] | int = ...

WeightDimNumOrFn: Incomplete

def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-08,
    dimension_numbers: MuonDimensionNumbers | None = None,
) -> jax.Array: ...

class MuonState(NamedTuple):
    count: chex.Array
    mu: base.Updates
    ns_coeffs: chex.Array

def scale_by_muon(
    ns_coeffs: tuple[float, float, float] | tuple[tuple[float, float, float], ...] = (
        3.4445,
        -4.775,
        2.0315,
    ),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-08,
    mu_dtype: chex.ArrayDType | None = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
) -> base.GradientTransformation: ...
def muon(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: tuple[float, float, float] | tuple[tuple[float, float, float], ...] = (
        3.4445,
        -4.775,
        2.0315,
    ),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-08,
    weight_decay: float = 0.0,
    weight_decay_mask: Any | Callable[[base.Params], Any] | None = None,
    mu_dtype: chex.ArrayDType | None = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps_root: float = 0.0,
    adam_weight_decay: float = 0.0,
    muon_weight_dimension_numbers: WeightDimNumOrFn | None = None,
) -> base.GradientTransformation: ...
