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

import torch

from priml.optimizers.parameter_filter import ParameterFilter, everything


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor, nn
    from torch.nn import Parameter
    from torch.optim.optimizer import StateDict as OptimizerStateDict


class CompositeOptimizer(Optimizer):
    """Drives several optimizers as one.

    Members own state entries and group topology. The concrete ``state`` dict
    and ``param_groups`` list keep their identity and alias member values, and
    each read of either attribute rebuilds both from the members. A container
    kept from an earlier read goes stale once a member changes, so read the
    attribute again. Edit topology on members; edit state payloads and group
    settings through either owner or composite. Indexing ``state`` with a
    parameter that has no entry creates it on its member, as torch does.

    Args:
      optimizers: Members, stepped in the order given. Each parameter must
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
        """One filter per member. Empty gives a single member every parameter."""

        require_total: bool = True
        """Reject a recipe that leaves a trainable parameter unclaimed; when
        off, an unclaimed parameter is frozen."""

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
              ValueError: If no member is configured, a member is itself a
                composite, or ``select`` does not name one filter per member.

            """
            final = self.finalized()
            if not final.optimizers:
                raise ValueError("CompositeOptimizer.Config needs a member.")
            if any(isinstance(m, CompositeOptimizer.Config) for m in final.optimizers):
                raise ValueError(
                    "CompositeOptimizer.Config cannot hold a composite: a member is "
                    "built over its parameters, a composite over a model.",
                )
            filters = final.select or [everything]
            if len(filters) != len(final.optimizers):
                raise ValueError(
                    f"select names {len(final.select)} filters for "
                    f"{len(final.optimizers)} optimizers.",
                )
            return partial(
                _route,
                members=[member.make() for member in final.optimizers],
                filters=filters,
                require_total=final.require_total,
                drop_empty=final.drop_empty,
            )

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
        # Torch's cycling schedulers read ``defaults`` to learn whether every
        # group carries momentum, so it names what all members share.
        first = cast("dict[str, object]", optimizers[0].defaults)
        super().__init__(
            [{"params": []}],
            defaults={
                name: value
                for name, value in first.items()
                if all(name in optimizer.defaults for optimizer in optimizers)
            },
        )
        # The members' OWN group dicts and state, aliased rather than copied:
        # a scheduler writing ``composite.param_groups[i]["lr"]`` must reach the
        # optimizer that will read it, and a copy would silently discard the
        # write. Replaces the placeholder.
        self._groups: list[dict[str, object]] = []
        self._state: dict[object, object] = _MemberState(self.optimizers)
        self.param_groups = self._groups
        self.state: dict[object, object] = self._state
        self._initialized = True

    # Hidden from type checkers: an override returning ``object`` would make
    # them accept any attribute name, so a typo would pass basedpyright and ty.
    if not TYPE_CHECKING:

        @override
        def __getattribute__(self, name: str) -> object:
            if name in {"state", "param_groups"} and self._initialized:
                self._refresh_collections()
            return super().__getattribute__(name)

    @property
    def requires_closure(self) -> bool:
        """Report whether a member needs the loss-recomputing closure."""
        return any(getattr(o, "requires_closure", False) for o in self.optimizers)

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
            second forward that would also double-count BatchNorm stats. When
            no member takes it, it is called once here, with grad enabled as
            torch optimizers do, for the return value.

        Returns:
          loss: The closure's value -- from the last member that took it, else
            from the call here -- or None when no closure was given.

        Raises:
          ValueError: If a member has taken a parameter another member owns,
            e.g. through its own ``add_param_group``.

        """
        _reject_shared_parameters(self.optimizers)
        loss: Tensor | float | None = None
        if closure is not None and not self.requires_closure:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers:
            if closure is not None and getattr(optimizer, "requires_closure", False):
                loss = optimizer.step(closure)
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
        for pre_hook in self._optimizer_state_dict_pre_hooks.values():
            pre_hook(self)
        typed: CompositeOptimizer.StateDict = {
            "optimizers": [o.state_dict() for o in self.optimizers],
        }
        state: OptimizerStateDict = {**typed}
        for post_hook in self._optimizer_state_dict_post_hooks.values():
            replaced = post_hook(self, state)
            state = state if replaced is None else replaced
        return state

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`, all members or none.

        Raises:
          ValueError: If the checkpoint holds a different number of members,
            which means the recipe changed and the state cannot be matched up.

        """
        loaded: OptimizerStateDict = {**state_dict}
        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            replaced = pre_hook(self, loaded)
            loaded = loaded if replaced is None else replaced
        saved = cast(CompositeOptimizer.StateDict, loaded)["optimizers"]
        if len(saved) != len(self.optimizers):
            raise ValueError(
                f"Checkpoint holds {len(saved)} optimizers but this composite "
                f"has {len(self.optimizers)}; the recipe changed.",
            )
        current = [optimizer.state_dict() for optimizer in self.optimizers]
        try:
            for optimizer, member_state in zip(self.optimizers, saved, strict=True):
                optimizer.load_state_dict(member_state)
        except Exception:
            for optimizer, member_state in zip(self.optimizers, current, strict=True):
                optimizer.load_state_dict(member_state)
            raise
        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)

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
    def __getstate__(self) -> dict[str, object]:
        """Pickle the members; torch's state would drop them."""
        return {"optimizers": self.optimizers}

    @override
    def __setstate__(self, state: dict[str, object]) -> None:
        self.__init__(cast("list[Optimizer]", state["optimizers"]))

    @override
    def __repr__(self) -> str:
        members = ", ".join(type(o).__name__ for o in self.optimizers)
        return f"{type(self).__name__}({members})"

    def _refresh_collections(self) -> None:
        state: dict[object, object] = {}
        groups: list[dict[str, object]] = []
        for optimizer in self.optimizers:
            state.update(optimizer.state)
            groups.extend(optimizer.param_groups)
        self._state.clear()
        self._state.update(state)
        self._groups[:] = groups


def _route(
    model: nn.Module,
    *,
    members: Sequence[Callable[..., Optimizer]],
    filters: Sequence[ParameterFilter],
    require_total: bool,
    drop_empty: bool,
) -> CompositeOptimizer:
    """Build the composite, each member over the trainable parameters it claims."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    claimed: dict[int, int] = {}
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
            claimed[id(parameter)] = index
            group.append(parameter)
        if not group:
            if drop_empty:
                continue
            raise ValueError(f"Filter {index} claimed no parameters.")
        groups.append(group)
        kept.append(members[index])
    unclaimed = [(n, p) for n, p in named if id(p) not in claimed]
    if unclaimed and require_total:
        raise ValueError(
            f"No filter claims {len(unclaimed)} trainable parameter(s), "
            f"e.g. {unclaimed[0][0]!r}; they would never be updated.",
        )
    # No member zeroes an unclaimed gradient, so left trainable it would grow
    # every step and inflate the clip norm of the parameters that do train.
    for _, parameter in unclaimed:
        parameter.requires_grad = False
    return CompositeOptimizer(
        [member(group) for member, group in zip(kept, groups, strict=True)],
    )


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


class _MemberState(dict[object, object]):
    """Every member's state; a parameter's missing entry is made on its member."""

    def __init__(self, optimizers: Sequence[Optimizer]) -> None:
        super().__init__()
        self._optimizers = optimizers

    def __missing__(self, key: object) -> object:
        for optimizer in self._optimizers:
            for group in optimizer.param_groups:
                if any(key is p for p in cast("list[Parameter]", group["params"])):
                    entry = cast(object, optimizer.state[cast("Tensor", key)])
                    self[key] = entry
                    return entry
        raise KeyError(key)
