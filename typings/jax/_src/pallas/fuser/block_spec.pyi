from collections.abc import (
    Callable as Callable,
    Sequence,
)
from typing import Any, Protocol

import dataclasses
import enum
import threading

from _typeshed import Incomplete
from jax import lax as lax
from jax._src import (
    ad_util as ad_util,
    core as core,
    custom_derivatives as custom_derivatives,
    hijax as hijax,
    pjit as pjit,
    prng as prng,
    state as state,
    tree_util as tree_util,
    typing as typing,
    util as util,
)
from jax._src.pallas import core as pallas_core
from jax._src.pallas.fuser import fuser_utils as fuser_utils
from jax._src.state import indexing as indexing
from jax._src.traceback_util import api_boundary as api_boundary

import jax

pull_block_spec_rules: dict[core.Primitive, PullBlockSpecRuleFn]

@dataclasses.dataclass
class PullRuleContext:
    avals_in: tuple[core.AbstractValue, ...]
    avals_out: tuple[core.AbstractValue, ...]
    out_usages: tuple[set[Usage], ...]
    eval_function: Any = ...
    scalar_prefetch_fn: Any = ...
    scalar_prefetch_handler: Any | None
    grid_len: int | None
    def set_eval_function(self, eval_function): ...

@dataclasses.dataclass
class PushRuleContext:
    avals_in: tuple[core.AbstractValue, ...]
    avals_out: tuple[core.AbstractValue, ...]

def make_scalar_prefetch_handler(*args): ...

@dataclasses.dataclass
class UsageRuleContext:
    avals_in: tuple[core.AbstractValue, ...]
    avals_out: tuple[core.AbstractValue, ...]

def compute_usage(jaxpr: core.Jaxpr, jaxpr_out_usages): ...

@dataclasses.dataclass(frozen=True)
class KernelEvalContext:
    scalar_prefetch: Any | None
    program_ids: tuple[int | jax.Array, ...] | None
    avals_in: tuple[core.AbstractValue, ...] | None
    avals_out: tuple[core.AbstractValue, ...] | None
    in_block_specs: tuple[pallas_core.BlockSpec, ...]
    out_block_specs: tuple[pallas_core.BlockSpec, ...]
    grid_len: int | None
    scalar_prefetch_handler: Any | None
    out_usages: tuple[set[Usage], ...] | None
    def get_program_ids(self): ...
    def get_in_block_indices(self): ...
    def get_out_block_indices(self): ...

class _SpEnv(threading.local):
    scalar_prefetch: Incomplete
    def __init__(self) -> None: ...

def pull_block_spec(
    f: Callable,
    out_block_specs: pallas_core.BlockSpec | tuple[pallas_core.BlockSpec, ...],
    *,
    scalar_prefetch_handler: Any | None = None,
    grid_len: int | None = None,
): ...
def make_kernel_function(
    jaxpr: core.Jaxpr,
    consts,
    in_tree,
    out_tree,
    read_usage_env,
    in_block_specs,
    block_spec_env,
    scalar_prefetch_handler,
    grid_len,
): ...
def get_fusion_values(
    fusion: Callable,
    *args,
    **kwargs,
) -> tuple[
    Callable,
    tuple[typing.SupportsShape, ...],
    tuple[typing.SupportsShape, ...],
]: ...

class Usage(enum.Enum):
    REGULAR = 0
    SCALAR_PREFETCH = 1

class UsageRuleFn(Protocol):
    def __call__(
        self,
        ctx: UsageRuleContext,
        used_outs: Sequence[set[Usage]] | set[Usage],
        **params: Any,
    ) -> Sequence[set[Usage]]: ...

usage_rules: dict[core.Primitive, UsageRuleFn]

def register_usage_rule(prim: core.Primitive) -> Callable[[Any], UsageRuleFn]: ...

class EvalRuleFn(Protocol):
    def __call__(
        self,
        ctx: KernelEvalContext,
        *args: Any,
        **params: Any,
    ) -> Sequence[Any]: ...

eval_rules: dict[core.Primitive, EvalRuleFn]

def register_eval_rule(prim: core.Primitive) -> Callable[[Any], EvalRuleFn]: ...

class PullBlockSpecRuleFn(Protocol):
    def __call__(
        self,
        ctx: PullRuleContext,
        block_spec: pallas_core.BlockSpec | tuple[pallas_core.BlockSpec, ...],
        **params: Any,
    ) -> Sequence[pallas_core.BlockSpec]: ...

def register_pull_block_spec_rule(
    prim: core.Primitive,
) -> Callable[[Any], PullBlockSpecRuleFn]: ...
def register_default_eval_rule(prim: core.Primitive): ...
def register_binop_rule(prim: core.Primitive): ...
def push_block_spec(f: Callable, *in_spec_args, **in_spec_kwargs): ...

push_block_spec_rules: dict[core.Primitive, PushBlockSpecRuleFn]

class PushBlockSpecRuleFn(Protocol):
    def __call__(
        self,
        ctx: PushRuleContext,
        block_spec: pallas_core.BlockSpec | tuple[pallas_core.BlockSpec, ...],
        **params: Any,
    ) -> pallas_core.BlockSpec | tuple[pallas_core.BlockSpec, ...]: ...

def register_push_block_spec_rule(
    prim: core.Primitive,
) -> Callable[[Any], PushBlockSpecRuleFn]: ...

register_binop_push_rule: Incomplete

def register_eltwise_rule(prim: core.Primitive): ...
