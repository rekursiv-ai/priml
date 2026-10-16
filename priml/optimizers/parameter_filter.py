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

from typing import TYPE_CHECKING, Protocol, cast, override


if TYPE_CHECKING:
    from torch.nn import Parameter


class ParameterFilter(Protocol):
    """Decides whether one named parameter is taken."""

    def __call__(self, name: str, parameter: Parameter) -> bool:
        """Return whether this named parameter is taken."""
        ...


def everything(name: str, parameter: Parameter) -> bool:
    """Take every parameter, frozen ones included."""
    del name, parameter
    return True


def trainable(name: str, parameter: Parameter) -> bool:
    """Take the parameters an optimizer moves: those that require a gradient."""
    del name
    return parameter.requires_grad


class _Combinator:
    """A filter whose slots are its value: compared, hashed, and set once."""

    __slots__ = ()

    @override
    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, name):
            raise AttributeError(f"{type(self).__name__}.{name} is fixed once set.")
        object.__setattr__(self, name, value)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _Combinator)
            and type(other) is type(self)
            and self._fields() == other._fields()
        )

    @override
    def __hash__(self) -> int:
        return hash((type(self), self._fields()))

    def _fields(self) -> tuple[object, ...]:
        return tuple(
            cast(object, getattr(self, name))
            for cls in type(self).__mro__
            for name in cast("tuple[str, ...]", vars(cls).get("__slots__", ()))
        )


class excluding(_Combinator):  # noqa: N801 -- The lowercase name matches the public combinator syntax.
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
    def __repr__(self) -> str:
        names = ", ".join(repr(f) for f in self.fragments)
        return f"excluding({filter_name(self.select)}, {names})"


class matching(_Combinator):  # noqa: N801 -- The lowercase name matches the public combinator syntax.
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
    def __repr__(self) -> str:
        return f"matching({', '.join(repr(f) for f in self.fragments)})"


class complement(_Combinator):  # noqa: N801 -- The lowercase name matches the public combinator syntax.
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
    def __repr__(self) -> str:
        return f"complement({filter_name(self.select)})"


class with_ndim(_Combinator):  # noqa: N801 -- The lowercase name matches the public combinator syntax.
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
    def __repr__(self) -> str:
        return f"with_ndim({filter_name(self.select)}, {self.ndim})"


def filter_name(select: ParameterFilter) -> str:
    """Return a stable name for a filter, never an address.

    Args:
      select: A function, a combinator, or any other callable filter.

    Returns:
      name: A function's qualified name, else the filter's ``repr``, or its
        class's qualified name when that ``repr`` holds an address.

    """
    qualname = getattr(select, "__qualname__", None)
    if isinstance(qualname, str):
        return qualname
    text = repr(select)
    return type(select).__qualname__ if " at 0x" in text else text
