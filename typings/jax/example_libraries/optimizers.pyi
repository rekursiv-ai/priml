from collections.abc import Callable
from typing import Any, NamedTuple

from _typeshed import Incomplete
from jax._src.util import (
    safe_map as safe_map,
    safe_zip as safe_zip,
    unzip2 as unzip2,
)

map = safe_map
zip = safe_zip

class OptimizerState(NamedTuple):
    packed_state: Incomplete
    tree_def: Incomplete
    subtree_defs: Incomplete

type Array = Any
type Params = Any
type State = Any
Updates = Params
type InitFn = Callable[[Params], OptimizerState]
Step = int
type UpdateFn = Callable[[Step, Updates, OptimizerState], OptimizerState]
type ParamsFn = Callable[[OptimizerState], Params]

class Optimizer(NamedTuple):
    init_fn: InitFn
    update_fn: UpdateFn
    params_fn: ParamsFn

type Schedule = Callable[[Step], float]

def optimizer(
    opt_maker: Callable[
        ...,
        tuple[
            Callable[[Params], State],
            Callable[[Step, Updates, Params], Params],
            Callable[[State], Params],
        ],
    ],
) -> Callable[..., Optimizer]: ...
@optimizer
def sgd(step_size): ...
@optimizer
def momentum(step_size: Schedule, mass: float): ...
@optimizer
def nesterov(step_size: Schedule, mass: float): ...
@optimizer
def adagrad(step_size, momentum: float = 0.9): ...
@optimizer
def rmsprop(step_size, gamma: float = 0.9, eps: float = 1e-08): ...
@optimizer
def rmsprop_momentum(
    step_size,
    gamma: float = 0.9,
    eps: float = 1e-08,
    momentum: float = 0.9,
): ...
@optimizer
def adam(step_size, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-08): ...
@optimizer
def adamax(step_size, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-08): ...
@optimizer
def sm3(step_size, momentum: float = 0.9): ...
def constant(step_size) -> Schedule: ...
def exponential_decay(step_size, decay_steps, decay_rate): ...
def inverse_time_decay(step_size, decay_steps, decay_rate, staircase: bool = False): ...
def polynomial_decay(step_size, decay_steps, final_step_size, power: float = 1.0): ...
def piecewise_constant(boundaries: Any, values: Any): ...
def make_schedule(scalar_or_schedule: float | Schedule) -> Schedule: ...
def l2_norm(tree): ...
def clip_grads(grad_tree, max_norm): ...

class JoinPoint:
    subtree: Incomplete
    def __init__(self, subtree) -> None: ...
    def __iter__(self): ...

def unpack_optimizer_state(opt_state): ...
def pack_optimizer_state(marked_pytree): ...
