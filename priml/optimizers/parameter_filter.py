"""Predicates over ``(name, parameter)`` choosing which parameters a consumer takes.

Several consumers walk ``named_parameters`` and keep a subset: a
``CompositeOptimizer`` routes each member its share, an ``EMA`` averages its
own. One vocabulary serves them all, so a recipe states "the embeddings" once
and every consumer reads it the same way.

Each combinator is a comparable object rather than a closure: two identical
filters must be equal, or a config carrying one could never equal its own
parent, which breaks both experiment diffing and serialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, override


if TYPE_CHECKING:
    from torch.nn import Parameter


class ParameterFilter(Protocol):
    """Decides whether one named parameter is taken."""

    def __call__(self, name: str, parameter: Parameter) -> bool:
        """Apply to the input."""
        ...


def everything(name: str, parameter: Parameter) -> bool:
    """Take every parameter, frozen ones included.

    Args:
      name: Name.
      parameter: Parameter.

    Returns:
      result: Always True.

    """
    del name, parameter
    return True


def trainable(name: str, parameter: Parameter) -> bool:
    """Take the parameters an optimizer moves.

    Args:
      name: Name.
      parameter: Parameter.

    Returns:
      result: Whether ``parameter`` requires a gradient.

    """
    del name
    return parameter.requires_grad


class excluding:  # noqa: N801 -- The lowercase name matches the public combinator syntax.
    """Narrow a filter by dropping parameters whose name contains a fragment.

    Args:
      select: Filter to narrow.
      fragments: Name fragments to reject, e.g. ``"head"``.

    """

    __slots__ = ("fragments", "select")

    def __init__(self, select: ParameterFilter, *fragments: str) -> None:
        self.select = select
        self.fragments = fragments

    def __call__(self, name: str, parameter: Parameter) -> bool:
        """Whether ``select`` takes this parameter and its name is allowed."""
        return self.select(name, parameter) and not any(
            fragment in name for fragment in self.fragments
        )

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, excluding)
            and self.select == other.select
            and self.fragments == other.fragments
        )

    @override
    def __hash__(self) -> int:
        return hash((type(self), self.select, self.fragments))

    @override
    def __repr__(self) -> str:
        names = ", ".join(repr(f) for f in self.fragments)
        return f"excluding({filter_name(self.select)}, {names})"


class matching:  # noqa: N801 -- The lowercase name matches the public combinator syntax.
    """Take parameters whose name contains any of the given fragments.

    The positive counterpart to :class:`excluding`, and what a recipe needs to
    put ONE class of parameter on its own rate: a partition built only from
    exclusions can carve a remainder but cannot name a part.

    Args:
      fragments: Name fragments to accept, e.g. ``"embed"``.

    """

    __slots__ = ("fragments",)

    def __init__(self, *fragments: str) -> None:
        self.fragments = fragments

    def __call__(self, name: str, parameter: Parameter) -> bool:
        """Whether this parameter's name carries one of the fragments."""
        del parameter
        return any(fragment in name for fragment in self.fragments)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, matching) and self.fragments == other.fragments

    @override
    def __hash__(self) -> int:
        return hash((type(self), self.fragments))

    @override
    def __repr__(self) -> str:
        return f"matching({', '.join(repr(f) for f in self.fragments)})"


class complement:  # noqa: N801 -- The lowercase name matches the public combinator syntax.
    """Take exactly what ``select`` does not, so a pair partitions the model.

    Args:
      select: Filter to invert.

    """

    __slots__ = ("select",)

    def __init__(self, select: ParameterFilter) -> None:
        self.select = select

    def __call__(self, name: str, parameter: Parameter) -> bool:
        """Whether ``select`` does NOT take this parameter."""
        return not self.select(name, parameter)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, complement) and self.select == other.select

    @override
    def __hash__(self) -> int:
        return hash((type(self), self.select))

    @override
    def __repr__(self) -> str:
        return f"complement({filter_name(self.select)})"


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


def filter_name(select: ParameterFilter) -> str:
    """Return a stable name for a filter, never an address.

    Args:
      select: The filter.

    Returns:
      name: Its qualified name, or its ``repr`` for a combinator.

    """
    qualname = getattr(select, "__qualname__", None)
    return qualname if isinstance(qualname, str) else repr(select)
