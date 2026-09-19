from collections.abc import Sequence

from _typeshed import Incomplete
from jax._src import core as core
from jax._src.export import shape_poly as shape_poly
from jax._src.export.shape_poly import (
    BoundsPrecision as BoundsPrecision,
    Comparator as Comparator,
    DimSize as DimSize,
    InconclusiveDimensionOperation as InconclusiveDimensionOperation,
    SymbolicScope as SymbolicScope,
    _DimExpr,
    _DimTerm,
)

def sgn(x): ...
def bounds_decision(e: DimSize, prec: BoundsPrecision) -> tuple[float, float]: ...

class _DecisionByElimination:
    scope: Incomplete
    def __init__(self, scope: SymbolicScope) -> None: ...
    def initialize(self) -> _DecisionByElimination: ...
    @staticmethod
    def build(scope: SymbolicScope) -> _DecisionByElimination: ...
    def combine_and_add_constraint(
        self,
        cmp: Comparator,
        e1: _DimExpr | float,
        e2: _DimExpr | float,
        debug_str: str | None = None,
    ): ...
    def add_to_state(self, cmp: Comparator, e: _DimExpr, debug_str: str | None): ...
    def combine_term_with_existing(
        self,
        t: _DimTerm,
        t_k: int,
        *,
        scope: shape_poly.SymbolicScope,
        only_smaller_than_t: bool = True,
    ) -> Sequence[tuple[Comparator, _DimExpr, int, int]]: ...
    def combine_constraint_with_existing(
        self,
        eq: Comparator,
        e: _DimExpr,
        debug_str: str | None,
    ) -> set[tuple[Comparator, _DimExpr]]: ...
    def bounds(
        self,
        e: DimSize,
        prec: BoundsPrecision,
        add_implicit_constraints: bool = False,
    ) -> tuple[float, float]: ...
    def add_implicit_constraints_expr(self, e: _DimExpr): ...
    def add_implicit_constraints_term(self, t: _DimTerm): ...
