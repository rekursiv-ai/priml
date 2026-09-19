from collections.abc import Callable, Iterator, Sequence

import dataclasses
import enum

from _typeshed import Incomplete
from jax._src.lib.mlir import ir as ir
from jax._src.lib.mlir.dialects import (
    arith as arith,
    memref as memref,
    scf as scf,
    vector as vector,
)

from . import (
    constraints as cs,
    fragmented_array as fa,
    inference_utils as inference_utils,
    launch_context as lc,
    tcgen05 as tcgen05,
    utils as utils,
)

class VariableType(enum.IntEnum):
    OPERAND = 0
    RESULT = 1
    ARGUMENT = 2

class MemorySpace(enum.Enum):
    REG = ...
    SMEM = ...
    TMEM = ...

@dataclasses.dataclass(frozen=True)
class ValueSite:
    operation: ir.OpView
    type: VariableType
    index: int
    region_index: int | None = ...
    def __post_init__(self) -> None: ...
    @property
    def value(self) -> ir.Value: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def memory_space(self) -> MemorySpace: ...

def extract_assignment_candidates_from_reduce_equation(
    small: cs.RegisterLayout,
    large: cs.Variable,
    reduction_dims: tuple[int, ...],
) -> Iterator[cs.RegisterLayout]: ...
def conjure_assignment(
    unknowns: Sequence[cs.Variable],
    constraint_system: cs.ConstraintSystem,
) -> Iterator[tuple[cs.Variable, cs.Constant]]: ...
def find_assignments_for(
    unknowns: Sequence[cs.Variable],
    constraint_system: cs.ConstraintSystem,
    *,
    fuel: int,
) -> tuple[dict[cs.Variable, cs.Constant] | cs.Unsatisfiable, int]: ...

@dataclasses.dataclass()
class DerivationContext:
    variable_for_value_site: dict[ValueSite, cs.Variable] = ...
    value_sites_for_variable: ValueSitesForVariable = ...
    def update(self, mapping: ValueSitesForVariable) -> None: ...
    def producer_ref(self, operand: ValueSite) -> cs.Variable: ...

type ValueSitesForVariable = dict[cs.Variable, list[ValueSite]]
type ConstraintSystemDerivationRuleResult = (
    cs.Unsatisfiable | tuple[cs.ConstraintSystem, ValueSitesForVariable]
)
ConstraintSystemDerivationRule: Incomplete

def is_vector(v: ir.Value) -> bool: ...
def prime_decomposition(n: int) -> list[int]: ...
def dynamic_gcd(a: int, b: ir.Value) -> int: ...

@dataclasses.dataclass(frozen=True)
class _TypeAndLayout:
    type: ir.Type
    layout: cs.Constant

def assign_layouts(solution: dict[ValueSite, cs.Constant]) -> None: ...
def vector_value_sites(op: ir.OpView) -> list[ValueSite]: ...
def producer_result(operand: ValueSite) -> ValueSite: ...
def consumer_operands(result: ValueSite) -> Sequence[ValueSite]: ...
def derive_relayout_constraints(
    value_sites_for_variable: ValueSitesForVariable,
) -> list[cs.Relayout]: ...
def is_terminator(op: ir.OpView) -> bool: ...
def traverse_op(op: ir.OpView, callback: Callable[[ir.OpView], None]): ...
def is_valid_register_layout_assignment(
    shape: tuple[int, ...],
    layout: fa.FragmentedLayout,
) -> bool: ...
def is_valid_smem_layout_assignment(
    shape: tuple[int, ...],
    tiling: lc.TileTransform,
) -> bool: ...
def is_valid_tmem_layout_assignment(
    shape: tuple[int, ...],
    layout: tcgen05.TMEMLayout,
) -> bool: ...
def check_layout_assignment(v: ValueSite, layout: cs.Constant) -> None: ...
def infer_layout(module: ir.Module, *, fuel: int = ...): ...
