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
with a :class:`~priml.optimizers.parameter_filter.ParameterFilter` -- a predicate
over ``(name, parameter)`` -- and hands each member only the parameters it
claims. A parameter claimed twice is an
error rather than a silent double update, and one claimed by nobody is left
frozen only if the recipe says so.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import field
from functools import partial
from typing import TYPE_CHECKING, TypedDict, cast, overload, override

from configgle import Fig, Makeable
from torch.optim import Optimizer

from priml.optimizers.parameter_filter import ParameterFilter, everything


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from torch import Tensor, nn
    from torch.nn import Parameter
    from torch.optim.optimizer import StateDict as OptimizerStateDict


class _ChainedState(dict[object, object]):
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
    def __getitem__(self, key: object) -> object:
        self._refresh()
        return super().__getitem__(key)

    @override
    def __len__(self) -> int:
        self._refresh()
        return super().__len__()

    @override
    def __iter__(self) -> Iterator[object]:
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
            parameters = cast("list[Parameter]", group["params"])
            for parameter in parameters:
                if id(parameter) in seen:
                    raise ValueError(
                        "A parameter belongs to more than one optimizer in the "
                        "composite, so it would be updated twice per step.",
                    )
                seen.add(id(parameter))


def _route(
    model: nn.Module,
    members: Sequence[Callable[..., Optimizer]],
    filters: Sequence[ParameterFilter],
    *,
    require_total: bool,
    drop_empty: bool = False,
) -> list[Optimizer]:
    """Build each member over the trainable parameters its filter claims."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    claimed: dict[int, str] = {}
    groups: list[list[Parameter]] = []
    kept: list[Callable[..., Optimizer]] = []
    for index, select in enumerate(filters):
        group: list[Parameter] = []
        for name, parameter in named:
            if not select(name, parameter):
                continue
            owner = claimed.get(id(parameter))
            if owner is not None:
                raise ValueError(
                    f"Parameter {name!r} is claimed by filter {owner} and "
                    f"{index}; it would be updated twice per step.",
                )
            claimed[id(parameter)] = str(index)
            group.append(parameter)
        if not group:
            if drop_empty:
                continue
            raise ValueError(f"Filter {index} claimed no parameters.")
        groups.append(group)
        kept.append(members[index])
    members = kept
    if require_total:
        unclaimed = [n for n, p in named if id(p) not in claimed]
        if unclaimed:
            raise ValueError(
                f"No filter claims {len(unclaimed)} trainable parameter(s), "
                f"e.g. {unclaimed[0]!r}; they would never be updated.",
            )
    return [member(group) for member, group in zip(members, groups, strict=True)]


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
        only the parameters its filter claims::

            config = CompositeOptimizer.Config()
            config.optimizers = [SignSGD.Config(), Muon.Config()]
            config.select = [complement(Muon.eligible_tensor), Muon.eligible_tensor]
            optimizer = config.make()(model)
        """

        optimizers: list[Makeable[Callable[..., Optimizer]]] = field(
            default_factory=list[Makeable[Callable[..., Optimizer]]],
        )
        """Member configs, e.g. ``[Muon.Config(), SignSGD.Config()]``."""

        select: list[ParameterFilter] = field(default_factory=list[ParameterFilter])
        """One filter per member. Empty gives every member every parameter,
        which is correct only when the members select for themselves."""

        require_total: bool = True
        """Reject a recipe that leaves a trainable parameter unclaimed."""

        drop_empty: bool = False
        """Drop a member whose filter claims nothing, instead of raising.

        Off by default because an empty filter is normally a misspelled name
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
                but does not name exactly one filter per member.

            """
            final = self.finalized()
            if not final.optimizers:
                raise ValueError("CompositeOptimizer.Config needs a member.")
            if final.select and len(final.select) != len(final.optimizers):
                raise ValueError(
                    f"select names {len(final.select)} filters for "
                    f"{len(final.optimizers)} optimizers.",
                )
            members = [member.make() for member in final.optimizers]
            filters = final.select or [everything] * len(members)
            require_total = final.require_total
            drop_empty = final.drop_empty

            def compose(model: nn.Module) -> CompositeOptimizer:
                return CompositeOptimizer(
                    _route(
                        model,
                        members,
                        filters,
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
        # The base installs the hook registries ``register_step_pre_hook`` and
        # checkpointing read, then builds groups by calling ``add_param_group``
        # once per entry -- which this class rejects. One empty placeholder
        # group satisfies its non-empty check; ``_initialized`` lets that single
        # call through and is what turns the rejection on afterwards.
        self._initialized = False
        super().__init__([{"params": []}], {})
        self._initialized = True
        # The members' OWN group dicts and state, aliased rather than copied:
        # a scheduler writing ``composite.param_groups[i]["lr"]`` must reach the
        # optimizer that will read it, and a copy would silently discard the
        # write. Replaces the placeholder.
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]
        self.state: dict[object, object] = _ChainedState(self.optimizers)

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

    class StateDict(TypedDict):
        """Every member's state, keyed by position."""

        optimizers: list[OptimizerStateDict]

    # Returns torch's ``dict[str, Any]`` rather than ``StateDict``: a TypedDict
    # is a ``Mapping``, not a ``dict``, so it cannot override ``Optimizer.state_dict``.
    @override
    def state_dict(self) -> OptimizerStateDict:
        """Return every member's state, keyed by position.

        Returns:
          state: ``{"optimizers": [<member state>, ...]}``.

        """
        state: CompositeOptimizer.StateDict = {
            "optimizers": [o.state_dict() for o in self.optimizers],
        }
        return {**state}

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`.

        Raises:
          ValueError: If the checkpoint holds a different number of members,
            which means the recipe changed and the state cannot be matched up.

        """
        saved = cast(CompositeOptimizer.StateDict, state_dict)["optimizers"]
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
    def add_param_group(self, param_group: dict[str, object]) -> None:
        """Reject: a composite cannot know which member should own the group."""
        if not self._initialized:
            super().add_param_group(param_group)
            return
        del param_group
        raise NotImplementedError(
            "Add the parameter group to one of the composite's members instead; "
            "the composite cannot know which optimizer should own it.",
        )

    @override
    def __repr__(self) -> str:
        members = ", ".join(type(o).__name__ for o in self.optimizers)
        return f"{type(self).__name__}({members})"
