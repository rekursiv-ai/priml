from collections.abc import Callable, Sequence
from typing import Any, Protocol

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    api_util as api_util,
    core as core,
    custom_api_util as custom_api_util,
    tree_util as tree_util,
    util as util,
)
from jax._src.interpreters import mlir as mlir
from jax._src.pallas import core as pallas_core
from jax._src.traceback_util import api_boundary as api_boundary

custom_fusion_p: Incomplete
CustomPullBlockSpecRuleFn: Incomplete
CustomPushBlockSpecRuleFn: Incomplete

@dataclasses.dataclass(frozen=True)
class CustomEvalContext:
    out_block_specs: tuple[pallas_core.BlockSpec, ...]
    out_block_indices: tuple[Any, ...]

class CustomEvalRuleFn(Protocol):
    def __call__(self, ctx: CustomEvalContext, *args: Any) -> Sequence[Any]: ...

class custom_fusion:
    fun: Callable[..., Any]
    eval_rule: CustomEvalRuleFn | None
    pull_block_spec_rule: CustomPullBlockSpecRuleFn | None
    push_block_spec_rule: CustomPushBlockSpecRuleFn | None
    pallas_impl: Callable[..., Any] | None
    def __init__(self, fun: Callable[..., Any]) -> None: ...
    def def_pallas_impl(self, pallas_impl): ...
    def def_pull_block_spec(self, pull_block_spec_rule: CustomPullBlockSpecRuleFn): ...
    def def_push_block_spec(self, push_block_spec_rule: CustomPushBlockSpecRuleFn): ...
    def def_eval_rule(self, eval_rule: CustomEvalRuleFn): ...
    def __call__(self, *args, **kwargs): ...
