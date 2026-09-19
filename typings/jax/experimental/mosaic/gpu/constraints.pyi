from collections.abc import Sequence
from typing import Any

import abc
import dataclasses

from . import (
    fragmented_array as fa,
    inference_utils as inference_utils,
    launch_context as lc,
    tcgen05 as tcgen05,
)

type VariableKey = Any

@dataclasses.dataclass(frozen=True)
class Variable:
    key: VariableKey

class Constant(abc.ABC): ...

@dataclasses.dataclass(frozen=True)
class RegisterLayout(Constant):
    value: fa.FragmentedLayout

@dataclasses.dataclass(frozen=True)
class TMEMLayout(Constant):
    value: tcgen05.TMEMLayout

@dataclasses.dataclass(frozen=True)
class SMEMTiling(Constant):
    value: lc.TileTransform | None

@dataclasses.dataclass(frozen=True)
class Reduce:
    expression: Expression
    axes: tuple[int, ...]

@dataclasses.dataclass(frozen=True)
class BroadcastInDim:
    expression: Expression
    axes: tuple[int, ...]
    shape: tuple[int, ...]

@dataclasses.dataclass(frozen=True)
class Reshape:
    expression: Expression
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]

@dataclasses.dataclass(frozen=True)
class Transpose:
    expression: Expression

type Expression = Variable | Constant | Reduce | BroadcastInDim | Reshape | Transpose

def reduce_broadcast_expression(
    broadcast: BroadcastInDim,
    assignments: dict[Variable, Constant],
) -> Expression | Unsatisfiable: ...
def reduce_reshape_expression(
    reshape: Reshape,
    assignments: dict[Variable, Constant],
) -> Expression | Unsatisfiable: ...
def reduce_transpose_expression(
    transpose: Transpose,
    assignments: dict[Variable, Constant],
) -> Expression | Unsatisfiable: ...
def reduce_expression(
    expr: Expression,
    assignments: dict[Variable, Constant],
) -> Expression | Unsatisfiable: ...

@dataclasses.dataclass(frozen=True)
class Equals:
    lhs: Expression
    rhs: Expression
    def holds(self) -> bool | None: ...

@dataclasses.dataclass(frozen=True)
class Relayout:
    source: Expression
    target: Expression
    bitwidth: int
    def holds(self) -> bool | None: ...

@dataclasses.dataclass(frozen=True)
class IsTransferable:
    source: Expression
    target: Expression
    shape: tuple[int, ...]
    def supported_tmem_transfers(
        self,
        packing: int,
    ) -> list[tuple[tcgen05.TMEMLayout, fa.FragmentedLayout]]: ...
    def holds(self) -> bool | None: ...

@dataclasses.dataclass(frozen=True)
class NotOfType:
    expr: Expression
    type: type[fa.FragmentedLayout]
    def holds(self) -> bool | None: ...

@dataclasses.dataclass(frozen=True)
class Divides:
    expr: Expression
    tiling_multiple: tuple[int, ...]
    def holds(self) -> bool | None: ...

@dataclasses.dataclass(frozen=True)
class IsValidMmaTiling:
    expr: Expression
    bitwidth: int
    def holds(self) -> bool | None: ...

type Constraint = (
    Equals | Relayout | NotOfType | IsTransferable | IsValidMmaTiling | Divides
)

def reduce_constraint(
    constraint: Constraint,
    assignments: dict[Variable, Constant],
) -> Constraint | Unsatisfiable: ...

@dataclasses.dataclass
class ConstraintSystem:
    assignments: dict[Variable, Constant] = ...
    constraints: Sequence[Constraint] = ...
    def unknowns(self) -> list[Variable]: ...
    def __and__(
        self,
        other: ConstraintSystem | Unsatisfiable,
    ) -> ConstraintSystem | Unsatisfiable: ...

class Unsatisfiable:
    def __and__(self, other: ConstraintSystem | Unsatisfiable) -> Unsatisfiable: ...

def non_splat_variables(constraints: Sequence[Constraint]) -> set[Variable]: ...
def saturate_distinct_from_splat(
    constraint_system: ConstraintSystem,
) -> ConstraintSystem | Unsatisfiable: ...
def compute_transitively_equal_vars(
    system: ConstraintSystem,
) -> dict[Variable, list[Variable]]: ...
def saturate_divides_constraints_for_equal_vars(
    system: ConstraintSystem,
) -> ConstraintSystem: ...
def merge_divides_constraints(
    constraints: Sequence[Constraint],
) -> list[Constraint]: ...
def reduce(constraint_system: ConstraintSystem) -> ConstraintSystem | Unsatisfiable: ...
