"""One optimizer that drives several, so a caller only ever holds one.

A recipe that routes parameters to different algorithms -- Muon on weight
matrices, SGD on the vectors beside them -- naturally produces several
optimizers. Handing that list to the training loop pushes the fan-out into
every caller: each has to step them all, zero them all, and checkpoint them
all, and forgetting one is silent.

:class:`CompositeOptimizer` absorbs that. It IS a ``Optimizer``,
holding every group of every member, so a train step written for one optimizer
runs a split recipe unchanged and a single ``state_dict`` round-trips the whole
stack.

Routing lives here too. :class:`CompositeOptimizer.Config` pairs each member
with a :class:`Selector` -- a predicate over ``(name, parameter)`` -- and hands
each member only the parameters it claims. A parameter claimed twice is an
error rather than a silent double update, and one claimed by nobody is left
frozen only if the recipe says so.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, overload, override

from configgle import Fig, Makeable
from torch.optim import Optimizer


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from torch import Tensor, nn
    from torch.nn import Parameter


class Selector(Protocol):
    """Decides whether one named parameter belongs to a member optimizer."""

    def __call__(self, name: str, parameter: Parameter) -> bool: ...


def everything(name: str, parameter: Parameter) -> bool:
    """Select every parameter; the single-group recipe."""
    del name, parameter
    return True


class excluding:  # noqa: N801 -- reads as a combinator at the call site
    """Narrow a selector by dropping parameters whose name contains a fragment.

    A comparable object rather than a closure: two identical selectors must be
    equal, or a config carrying one could never equal its own parent, which
    breaks both experiment diffing and serialization.

    Args:
      select: Selector to narrow.
      fragments: Name fragments to reject, e.g. ``"head"``.

    """

    __slots__ = ("fragments", "select")

    def __init__(self, select: Selector, *fragments: str) -> None:
        self.select = select
        self.fragments = fragments

    def __call__(self, name: str, parameter: Parameter) -> bool:
        """Whether ``select`` claims this parameter and its name is allowed."""
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
        return f"excluding({_name(self.select)}, {names})"


class matching:  # noqa: N801 -- reads as a combinator at the call site
    """Select parameters whose name contains any of the given fragments.

    The positive counterpart to :class:`excluding`, and what a recipe needs to
    put ONE class of parameter on its own rate: a partition built only from
    exclusions can carve a remainder but cannot name a part.

    A comparable object rather than a closure, for the reason
    :class:`excluding` documents.

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


class complement:  # noqa: N801 -- reads as a combinator at the call site
    """Select exactly what ``select`` does not, so a pair partitions the model.

    Args:
      select: Selector to invert.

    """

    __slots__ = ("select",)

    def __init__(self, select: Selector) -> None:
        self.select = select

    def __call__(self, name: str, parameter: Parameter) -> bool:
        """Whether ``select`` does NOT claim this parameter."""
        return not self.select(name, parameter)

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, complement) and self.select == other.select

    @override
    def __hash__(self) -> int:
        return hash((type(self), self.select))

    @override
    def __repr__(self) -> str:
        return f"complement({_name(self.select)})"


def _name(select: Selector) -> str:
    """Return a stable name for a selector, never an address."""
    qualname = getattr(select, "__qualname__", None)
    return qualname if isinstance(qualname, str) else repr(select)


class CompositeOptimizer(Optimizer):
    """Drives several optimizers as one.

    Args:
      optimizers: Members, stepped in the order given. Each parameter should
        belong to exactly one; overlapping members would apply two updates per
        step.

    Raises:
      ValueError: If ``optimizers`` is empty, or a parameter appears in more
        than one member.

    """

    class Config(Fig["Callable[..., CompositeOptimizer]"]):
        """The members this composite drives, and what each one claims.

        ``make()`` returns a builder taking the MODEL, so each member receives
        only the parameters its selector claims::

            config = CompositeOptimizer.Config()
            config.optimizers = [SignSGD.Config(), Muon.Config()]
            config.select = [complement(Muon.eligible_tensor), Muon.eligible_tensor]
            optimizer = config.make()(model)
        """

        optimizers: list[Makeable[Callable[..., Optimizer]]] = field(
            default_factory=list[Makeable[Callable[..., Optimizer]]],
        )
        """Member configs, e.g. ``[Muon.Config(), SignSGD.Config()]``."""

        select: list[Selector] = field(default_factory=list[Selector])
        """One selector per member. Empty gives every member every parameter,
        which is correct only when the members select for themselves."""

        require_total: bool = True
        """Reject a recipe that leaves a trainable parameter unclaimed."""

        drop_empty: bool = False
        """Drop a member whose selector claims nothing, instead of raising.

        Off by default because an empty selector is normally a misspelled name
        fragment, and silently training nothing with that member is the worst
        possible response. Turn it on for a recipe that names a class the model
        MAY not instantiate -- an ablation that switches a mechanism off still
        wants the rates its siblings use."""

        @override
        def make(self) -> Callable[..., CompositeOptimizer]:
            """Return a builder awaiting the model whose parameters to split.

            Returns:
              build: Takes an ``nn.Module``, returns the composite.

            Raises:
              ValueError: If no member is configured, or ``select`` is given
                but does not name exactly one selector per member.

            """
            final = (
                self.copy_tree()
                if getattr(self, "_finalized", False)
                else self.copy_tree().finalize()
            )
            if not final.optimizers:
                raise ValueError("CompositeOptimizer.Config needs a member.")
            if final.select and len(final.select) != len(final.optimizers):
                raise ValueError(
                    f"select names {len(final.select)} selectors for "
                    f"{len(final.optimizers)} optimizers.",
                )
            members = [member.make() for member in final.optimizers]
            selectors = final.select or [everything] * len(members)
            require_total = final.require_total
            drop_empty = final.drop_empty

            def compose(model: nn.Module) -> CompositeOptimizer:
                return CompositeOptimizer(
                    _route(
                        model,
                        members,
                        selectors,
                        require_total=require_total,
                        drop_empty=drop_empty,
                    ),
                )

            return partial(compose)

    def __init__(self, optimizers: Sequence[Optimizer]) -> None:
        if not optimizers:
            raise ValueError("CompositeOptimizer requires at least one optimizer.")
        _reject_shared_parameters(optimizers)
        self.optimizers = list(optimizers)
        self.defaults: dict[str, Any] = {}
        # The members' OWN group dicts and state, aliased rather than copied:
        # a scheduler writing ``composite.param_groups[i]["lr"]`` must reach the
        # optimizer that will read it, and a copy would silently discard the
        # write. ``super().__init__`` is skipped because it would build fresh
        # groups from a parameter list instead.
        self.param_groups: list[dict[str, Any]] = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]
        self.state: dict[Any, Any] = _ChainedState(self.optimizers)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], Tensor | float]) -> Tensor | float: ...

    @override
    def step(
        self,
        closure: Callable[[], Tensor | float] | None = None,
    ) -> Tensor | float | None:
        """Step every member in order.

        Args:
          closure: Loss-recomputing closure. Forwarded ONLY to members that set
            ``requires_closure`` (e.g. exact-Hessian Newton, which needs a
            graph-bearing loss to differentiate twice); a first-order torch
            optimizer executes any closure it is handed, running a wasteful
            second forward that would also double-count BatchNorm stats. Called
            once here for the return value, so a member that never sees it still
            steps against gradients the caller already populated.

        Returns:
          loss: The closure's value, or None when no closure was given.

        """
        loss = closure() if closure is not None else None
        for optimizer in self.optimizers:
            if closure is not None and getattr(optimizer, "requires_closure", False):
                optimizer.step(closure)
            else:
                optimizer.step()
        return loss

    @override
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients across every member."""
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    @override
    def state_dict(self) -> dict[str, Any]:
        """Return every member's state, keyed by position.

        Returns:
          state: ``{"optimizers": [<member state>, ...]}``.

        """
        return {"optimizers": [o.state_dict() for o in self.optimizers]}

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`.

        Raises:
          ValueError: If the checkpoint holds a different number of members,
            which means the recipe changed and the state cannot be matched up.

        """
        saved = state_dict["optimizers"]
        if len(saved) != len(self.optimizers):
            raise ValueError(
                f"Checkpoint holds {len(saved)} optimizers but this composite "
                f"has {len(self.optimizers)}; the recipe changed.",
            )
        for optimizer, member_state in zip(self.optimizers, saved, strict=True):
            optimizer.load_state_dict(member_state)
        # ``Optimizer.load_state_dict`` REPLACES a member's group dicts, which
        # orphans the aliases captured in __init__: a scheduler writing
        # ``composite.param_groups[i]["lr"]`` would then reach a dict no member
        # reads, and a resumed run would silently ignore its schedule.
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    @override
    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Reject: a composite cannot know which member should own the group."""
        del param_group
        raise NotImplementedError(
            "Add the parameter group to one of the composite's members instead; "
            "the composite cannot know which optimizer should own it.",
        )

    @override
    def __repr__(self) -> str:
        members = ", ".join(type(o).__name__ for o in self.optimizers)
        return f"{type(self).__name__}({members})"


def _route(
    model: nn.Module,
    members: Sequence[Callable[..., Optimizer]],
    selectors: Sequence[Selector],
    *,
    require_total: bool,
    drop_empty: bool = False,
) -> list[Optimizer]:
    """Build each member over the trainable parameters its selector claims.

    Args:
      model: Model whose parameters to partition.
      members: Optimizer constructors, one per selector.
      selectors: Predicates deciding ownership, tried in order.
      require_total: Reject a partition that leaves a parameter unclaimed.
      drop_empty: Skip a member whose selector claims nothing rather than
        raising, for a recipe naming a class the model may not instantiate.

    Returns:
      optimizers: One built optimizer per non-empty member, in the given order.

    Raises:
      ValueError: If a selector claims nothing and ``drop_empty`` is false, if
        two claim the same parameter, or if ``require_total`` and a parameter
        is unclaimed.

    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    claimed: dict[int, str] = {}
    groups: list[list[Parameter]] = []
    kept: list[Callable[..., Optimizer]] = []
    for index, select in enumerate(selectors):
        group: list[Parameter] = []
        for name, parameter in named:
            if not select(name, parameter):
                continue
            owner = claimed.get(id(parameter))
            if owner is not None:
                raise ValueError(
                    f"Parameter {name!r} is claimed by selector {owner} and "
                    f"{index}; it would be updated twice per step.",
                )
            claimed[id(parameter)] = str(index)
            group.append(parameter)
        if not group:
            if drop_empty:
                continue
            raise ValueError(f"Selector {index} claimed no parameters.")
        groups.append(group)
        kept.append(members[index])
    members = kept
    if require_total:
        unclaimed = [n for n, p in named if id(p) not in claimed]
        if unclaimed:
            raise ValueError(
                f"No selector claims {len(unclaimed)} trainable parameter(s), "
                f"e.g. {unclaimed[0]!r}; they would never be updated.",
            )
    return [member(group) for member, group in zip(members, groups, strict=True)]


class _ChainedState(dict[Any, Any]):
    """A live view of every member's per-parameter state.

    ``Optimizer.state`` is a plain attribute, so the composite
    cannot expose it as a property without breaking the base class contract.
    This subclasses ``dict`` instead and refreshes from the members on each
    read, so state a member creates lazily (on its first step) still appears.
    """

    def __init__(self, optimizers: Sequence[Optimizer]) -> None:
        super().__init__()
        self._optimizers = optimizers

    @override
    def __getitem__(self, key: Any) -> Any:
        self._refresh()
        return super().__getitem__(key)

    @override
    def __len__(self) -> int:
        self._refresh()
        return super().__len__()

    @override
    def __iter__(self) -> Iterator[Any]:
        self._refresh()
        return super().__iter__()

    def _refresh(self) -> None:
        for optimizer in self._optimizers:
            super().update(optimizer.state)


def _reject_shared_parameters(optimizers: Sequence[Optimizer]) -> None:
    """Raise if any parameter belongs to more than one optimizer."""
    seen: set[int] = set()
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if id(parameter) in seen:
                    raise ValueError(
                        "A parameter belongs to more than one optimizer in the "
                        "composite, so it would be updated twice per step.",
                    )
                seen.add(id(parameter))
