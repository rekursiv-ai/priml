"""Parameter routing for the reference TRM optimizer stacks."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from priml.optimizers.parameter_filter import filter_name


if TYPE_CHECKING:
    from torch.nn import Parameter

    from priml.optimizers.parameter_filter import ParameterFilter


class with_ndim:  # noqa: N801 -- The lowercase name matches the public combinator syntax.
    """Narrow a filter to parameters of exactly one rank.

    Muon folds a rank-3 weight's leading axis into an ensemble of matrices, so
    the reference routes rank-2 and rank-3 weights to separately configured
    Muon members; this is the predicate that splits them.

    Args:
      select: Filter to narrow.
      ndim: Rank to accept.

    """

    __slots__ = ("ndim", "select")

    def __init__(self, select: ParameterFilter, ndim: int) -> None:
        self.select = select
        self.ndim = ndim

    def __call__(self, name: str, parameter: Parameter) -> bool:
        """Whether ``select`` takes ``parameter`` and it has rank ``ndim``."""
        return parameter.ndim == self.ndim and self.select(name, parameter)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, with_ndim)
            and self.select == other.select
            and self.ndim == other.ndim
        )

    @override
    def __hash__(self) -> int:
        return hash((type(self), self.select, self.ndim))

    @override
    def __repr__(self) -> str:
        return f"with_ndim({filter_name(self.select)}, {self.ndim})"
