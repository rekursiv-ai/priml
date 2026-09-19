from typing import Any

from _typeshed import Incomplete
from jax._src import (
    api as api,
    api_util as api_util,
    config as config,
    core as core,
    custom_api_util as custom_api_util,
    dispatch as dispatch,
    errors as errors,
    sharding_impls as sharding_impls,
    tree_util as tree_util,
)
from jax._src.custom_partitioning_sharding_rule import (
    SdyShardingRule as SdyShardingRule,
    sdy_sharding_rule_to_mlir as sdy_sharding_rule_to_mlir,
    str_to_sdy_sharding_rule as str_to_sdy_sharding_rule,
)
from jax._src.interpreters import mlir as mlir
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import hlo as hlo
from jax._src.sharding import Sharding as Sharding

class _ShardingCallbackInfo:
    propagate_user_sharding: Incomplete
    partition: Incomplete
    to_mesh_pspec_sharding: Incomplete
    in_tree: Incomplete
    out_tree: Incomplete
    infer_sharding_from_operands: Incomplete
    module_context: Incomplete
    mesh: Incomplete
    static_args: Incomplete
    def __init__(
        self,
        propagate_user_sharding,
        partition,
        to_mesh_pspec_sharding,
        in_tree,
        out_tree,
        infer_sharding_from_operands,
        module_context,
        mesh,
        static_args,
    ) -> None: ...
    def unflatten_arg_shape(self, s, sharding): ...
    def unflatten_arg_shapes(self, arg_shapes, arg_shardings): ...

custom_partitioning_p: Incomplete

class custom_partitioning:
    fun: Incomplete
    partition: Incomplete
    static_argnums: Incomplete
    propagate_user_sharding: Incomplete
    infer_sharding_from_operands: Incomplete
    sharding_rule: Incomplete
    def __init__(self, fun, static_argnums=()) -> None: ...
    __getattr__: Any
    decode_shardings: Incomplete
    def def_partition(
        self,
        partition,
        infer_sharding_from_operands=None,
        propagate_user_sharding=None,
        decode_shardings: bool = True,
        sharding_rule=None,
        *,
        reduction_factors=(),
        need_replication_factors=(),
        permutation_factors=(),
        **factor_sizes,
    ): ...
    def __call__(self, *args, **kwargs): ...
