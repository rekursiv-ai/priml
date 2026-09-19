from _typeshed import Incomplete
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import sdy as sdy

BATCHING: str
IrTypes: Incomplete

class CompoundFactor(tuple):
    def __init__(self, *factors) -> None: ...
    def __new__(cls, *factors): ...

class ArrayMapping(tuple):
    def __init__(self, *dim_mappings) -> None: ...
    def __new__(cls, *dim_mappings): ...

class SdyShardingRule:
    operand_mappings: tuple[ArrayMapping, ...]
    result_mappings: tuple[ArrayMapping, ...]
    factor_sizes: dict[str, int]
    reduction_factors: tuple[str, ...]
    need_replication_factors: tuple[str, ...]
    permutation_factors: tuple[str, ...]
    def __init__(
        self,
        operand_mappings: tuple[ArrayMapping, ...],
        result_mappings: tuple[ArrayMapping, ...],
        *,
        reduction_factors: tuple[str, ...] = (),
        need_replication_factors: tuple[str, ...] = (),
        permutation_factors: tuple[str, ...] = (),
        **factor_sizes: int,
    ) -> None: ...

def str_to_sdy_sharding_rule(
    rule: str,
    *,
    reduction_factors: tuple[str, ...] = (),
    need_replication_factors: tuple[str, ...] = (),
    permutation_factors: tuple[str, ...] = (),
    **factor_sizes: int,
) -> SdyShardingRule: ...
def sdy_sharding_rule_to_mlir(
    rule: SdyShardingRule,
    operand_types: list[IrTypes],
    result_types: list[IrTypes],
) -> ir.Attribute: ...
