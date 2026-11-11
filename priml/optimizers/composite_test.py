"""Tests for CompositeOptimizer."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Protocol, TypeGuard, cast, override, runtime_checkable

import weakref

from configgle import Makes, PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.lr import apply_lr_scale, remember_initial_lrs
from priml.optimizers.muon import Muon
from priml.optimizers.newton import Newton
from priml.optimizers.parameter_filter import (
    complement,
    everything,
    excluding,
    matching,
)
from priml.optimizers.weight_programming import (
    UpdateOutput,
    WeightProgramming,
    _no_batch,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def split_model() -> nn.Module:
    """Return a model with both matrix weights and 1-D parameters."""
    return nn.Sequential(nn.Conv2d(3, 4, 3, bias=False), nn.BatchNorm2d(4))


def split_optimizer(model: nn.Module) -> CompositeOptimizer:
    """Route the model's matrices to Muon and its vectors to SGD."""
    matrix = [p for p in model.parameters() if p.ndim >= 2]
    vector = [p for p in model.parameters() if p.ndim == 1]
    return CompositeOptimizer(
        [torch.optim.SGD(vector, lr=0.1), Muon(matrix, lr=0.1)],
    )


@runtime_checkable
class _ParameterGroup(Protocol):
    def __getitem__(self, key: str, /) -> object: ...


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def parameter_group_size(group: object) -> int:
    """Read one torch optimizer group's parameter count without trusting its stub."""
    assert isinstance(group, _ParameterGroup)
    params = group["params"]
    assert _is_object_list(params)
    return len(params)


def backward(model: nn.Module) -> None:
    """Populate gradients on every parameter."""
    output = cast(Tensor, model(torch.randn(4, 3, 5, 6)))
    output.square().sum().backward()


def test_is_an_optimizer() -> None:
    # The whole point: a caller written for one optimizer takes a composite.
    assert isinstance(split_optimizer(split_model()), torch.optim.Optimizer)


def test_exposes_every_members_parameter_groups() -> None:
    model = split_model()
    optimizer = split_optimizer(model)
    owned: list[int] = []
    for group in optimizer.param_groups:
        params = cast(list[Tensor], group["params"])
        owned.extend(id(parameter) for parameter in params)
    assert sorted(owned) == sorted(id(p) for p in model.parameters())


def test_parameter_groups_are_the_members_own_dicts() -> None:
    """A scheduler writing through the composite must reach the member.

    Copying the dicts would silently discard every learning-rate update.
    """
    model = split_model()
    optimizer = split_optimizer(model)
    member = optimizer.optimizers[0]
    optimizer.param_groups[0]["lr"] = 0.999
    assert member.param_groups[0]["lr"] == 0.999


def test_step_updates_parameters_from_every_member() -> None:
    torch.manual_seed(0)
    model = split_model()
    optimizer = split_optimizer(model)
    backward(model)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    moved = [
        not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True)
    ]
    assert all(moved)


def test_zero_grad_clears_every_member() -> None:
    model = split_model()
    optimizer = split_optimizer(model)
    backward(model)
    optimizer.zero_grad()
    assert all(p.grad is None for p in model.parameters())


def test_state_dict_round_trips_every_member() -> None:
    torch.manual_seed(0)
    model = split_model()
    optimizer = split_optimizer(model)
    backward(model)
    optimizer.step()
    saved = optimizer.state_dict()

    restored = split_optimizer(model)
    restored.load_state_dict(saved)
    torch.testing.assert_close(restored.state_dict(), saved, rtol=0, atol=0)


def test_state_exposes_member_state_after_a_step() -> None:
    # A member creates its state lazily on the first step, so the view must
    # read through rather than snapshot at construction.
    torch.manual_seed(0)
    model = split_model()
    optimizer = split_optimizer(model)
    assert len(optimizer.state) == 0
    backward(model)
    optimizer.step()
    assert len(optimizer.state) > 0


@pytest.mark.parametrize("operation", ["get", "items", "values", "contains"])
def test_state_reads_observe_lazy_member_initialization(operation: str) -> None:
    parameter = nn.Parameter(torch.ones(2, 3))
    member = torch.optim.Adam([parameter], lr=0.1)
    optimizer = CompositeOptimizer([member])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state
    expected = cast(object, member.state[parameter])
    if operation == "get":
        assert state.get(parameter) is expected
    elif operation == "items":
        assert [(id(key), id(value)) for key, value in state.items()] == [
            (id(parameter), id(expected)),
        ]
    elif operation == "values":
        assert [id(value) for value in state.values()] == [id(expected)]
    else:
        assert parameter in state


def test_state_drops_removed_entries_and_follows_replacement() -> None:
    parameter = nn.Parameter(torch.ones(2, 3))
    member = torch.optim.Adam([parameter], lr=0.1)
    optimizer = CompositeOptimizer([member])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert optimizer.state[parameter] is member.state[parameter]
    member.state.clear()
    assert optimizer.state == {}
    member.state = {parameter: {"replacement": 3}}
    assert optimizer.state.get(parameter) is member.state[parameter]


def test_state_makes_a_missing_entry_on_its_member() -> None:
    # Torch's ``state`` is a defaultdict, so callers index it before a step.
    parameter = nn.Parameter(torch.ones(2))
    member = torch.optim.Adam([parameter], lr=0.1)
    optimizer = CompositeOptimizer([member])
    entry = optimizer.state[parameter]
    assert parameter in member.state
    assert entry is member.state[parameter]
    with pytest.raises(KeyError):
        _ = optimizer.state[nn.Parameter(torch.ones(2))]


def test_member_group_addition_and_load_keep_scheduling_aliases() -> None:
    model = nn.Linear(2, 3)
    assert model.bias is not None
    member = torch.optim.SGD([model.weight], lr=0.1)
    optimizer = CompositeOptimizer([member])
    member.add_param_group({"params": [model.bias]})
    remember_initial_lrs([optimizer])
    apply_lr_scale([optimizer], scale=0.25)
    assert member.param_groups[1]["lr"] == 0.025
    member.load_state_dict(member.state_dict())
    assert len(optimizer.param_groups) == 2
    assert all(
        actual is expected
        for actual, expected in zip(
            optimizer.param_groups,
            member.param_groups,
            strict=True,
        )
    )
    apply_lr_scale([optimizer], scale=0.5)
    assert all(group["lr"] == 0.05 for group in member.param_groups)
    member.param_groups.pop()
    assert len(optimizer.param_groups) == 1


def test_a_discarded_composite_is_not_kept_alive_by_its_members() -> None:
    member = torch.optim.SGD([nn.Parameter(torch.ones(2))], lr=0.1)
    composite = weakref.ref(CompositeOptimizer([member]))
    assert composite() is None


def test_step_rejects_a_parameter_a_member_took_from_its_sibling() -> None:
    """A group added on a member is stepped, so ownership is checked per step."""
    owned = nn.Parameter(torch.ones(2))
    sibling = torch.optim.SGD([nn.Parameter(torch.ones(2))], lr=1.0)
    optimizer = CompositeOptimizer([torch.optim.SGD([owned], lr=1.0), sibling])
    sibling.add_param_group({"params": [owned]})
    owned.grad = torch.ones_like(owned)
    with pytest.raises(ValueError, match="more than one optimizer"):
        optimizer.step()
    assert torch.equal(owned, torch.ones(2))


def test_static_steps_and_resume_match_independent_members_exactly() -> None:
    parameters = [
        nn.Parameter(torch.arange(6).float().reshape(2, 3) / 7),
        nn.Parameter(torch.arange(3).float() / 5),
    ]
    reference = [nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    members = [
        torch.optim.AdamW([parameters[0]], lr=0.01),
        torch.optim.SGD([parameters[1]], lr=0.03, momentum=0.7),
    ]
    controls = [
        torch.optim.AdamW([reference[0]], lr=0.01),
        torch.optim.SGD([reference[1]], lr=0.03, momentum=0.7),
    ]
    optimizer = CompositeOptimizer(members)
    for step in range(3):
        if step == 1:
            optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
            for control in controls:
                control.load_state_dict(deepcopy(control.state_dict()))
        for actual, expected in zip(parameters, reference, strict=True):
            actual.grad = torch.full_like(actual, fill_value=(step + 1) / 3)
            expected.grad = actual.grad.clone()
        optimizer.step()
        for control in controls:
            control.step()
        for actual, expected in zip(parameters, reference, strict=True):
            assert torch.equal(actual, expected)
        for member, control in zip(members, controls, strict=True):
            torch.testing.assert_close(
                member.state_dict(),
                control.state_dict(),
                rtol=0,
                atol=0,
            )
        optimizer.zero_grad()
        assert all(parameter.grad is None for parameter in parameters)


def test_step_returns_the_closure_value() -> None:
    model = split_model()
    optimizer = split_optimizer(model)
    backward(model)
    assert optimizer.step(lambda: 42.0) == 42.0


def test_closure_is_evaluated_once() -> None:
    # Calling it here as well as in Newton would run a second forward.
    model = nn.Linear(3, 1, bias=False)
    optimizer = CompositeOptimizer([Newton(model.parameters(), lr=1.0)])
    calls: list[int] = []
    _ = optimizer.step(
        lambda: calls.append(1) or model(torch.ones(2, 3)).square().sum(),
    )
    assert len(calls) == 1


def test_closure_runs_with_grad_enabled() -> None:
    # Torch optimizers enable grad around a closure, so it may call backward().
    model = split_model()
    optimizer = split_optimizer(model)
    with torch.no_grad():
        _ = optimizer.step(lambda: backward(model) or 0.0)
    assert all(p.grad is not None for p in model.parameters())


def test_requires_closure_follows_the_members() -> None:
    # The training loop forwards a closure only to an optimizer asking for it.
    model = nn.Linear(3, 1, bias=False)
    assert CompositeOptimizer([Newton(model.parameters(), lr=1.0)]).requires_closure
    assert not split_optimizer(split_model()).requires_closure


def test_closure_reaches_a_member_that_requires_it() -> None:
    """A closure-based member must receive the closure, not a bare step().

    ``CompositeOptimizer`` IS an ``Optimizer``, so it owes members the same
    ``step(closure)`` contract torch does; Newton cannot build a Hessian
    without one and raises.
    """
    torch.manual_seed(0)
    model = nn.Linear(3, 1, bias=False)
    optimizer = CompositeOptimizer([Newton(model.parameters(), lr=1.0)])
    inputs = torch.randn(32, 3)
    labels = (inputs.sum(-1) > 0).float()

    def closure() -> Tensor:
        logits = model(inputs).squeeze(-1)
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    before = closure().item()
    for _ in range(3):
        _ = optimizer.step(closure)
    assert closure().item() < before


def test_closure_is_withheld_from_first_order_members() -> None:
    """Handing a torch optimizer a closure runs a second, wasteful forward.

    It would also double-count BatchNorm statistics, so only members declaring
    ``requires_closure`` may see it.
    """
    calls: list[int] = []
    model = split_model()
    optimizer = split_optimizer(model)
    backward(model)
    _ = optimizer.step(lambda: calls.append(1) or 0.0)
    assert len(calls) == 1


def test_rejects_no_members() -> None:
    with pytest.raises(ValueError, match="at least one optimizer"):
        _ = CompositeOptimizer([])


def test_rejects_a_parameter_shared_between_members() -> None:
    # Two members holding one parameter would apply two updates per step.
    model = split_model()
    shared = list(model.parameters())
    with pytest.raises(
        ValueError,
        match=r"^A parameter belongs to more than one optimizer in the composite, "
        r"so it would be updated twice per step\.$",
    ):
        _ = CompositeOptimizer(
            [torch.optim.SGD(shared, lr=0.1), torch.optim.AdamW(shared, lr=0.1)],
        )


def test_load_state_dict_keeps_the_groups_aliased() -> None:
    """Torch REPLACES a member's group dicts, orphaning the composite's list.

    A scheduler writing through the composite after a resume would then reach a
    dict no member reads, and the restored run would ignore its schedule.
    """
    model = split_model()
    optimizer = split_optimizer(model)
    optimizer.load_state_dict(optimizer.state_dict())
    optimizer.param_groups[0]["lr"] = 0.999
    assert optimizer.optimizers[0].param_groups[0]["lr"] == 0.999


def test_load_state_dict_rejects_a_different_member_count() -> None:
    model = split_model()
    saved = split_optimizer(model).state_dict()
    single = CompositeOptimizer([torch.optim.SGD(model.parameters(), lr=0.1)])
    with pytest.raises(ValueError, match="the recipe changed"):
        single.load_state_dict(saved)


def test_a_failed_load_leaves_every_member_unchanged() -> None:
    first, second, third = (nn.Parameter(torch.ones(2)) for _ in range(3))
    optimizer = CompositeOptimizer(
        [torch.optim.SGD([first], lr=0.1), torch.optim.SGD([second, third], lr=0.1)],
    )
    regrouped = CompositeOptimizer(
        [
            torch.optim.SGD([first], lr=0.5),
            torch.optim.SGD([{"params": [second]}, {"params": [third]}], lr=0.1),
        ],
    )
    with pytest.raises(ValueError, match="different number of parameter groups"):
        optimizer.load_state_dict(regrouped.state_dict())
    assert optimizer.param_groups[0]["lr"] == 0.1


def test_checkpoint_hooks_registered_on_the_composite_run() -> None:
    optimizer = split_optimizer(split_model())
    replacement = optimizer.state_dict()
    optimizer.param_groups[0]["lr"] = 0.5
    calls: list[str] = []
    _ = optimizer.register_state_dict_pre_hook(lambda _: calls.append("save"))
    _ = optimizer.register_state_dict_post_hook(lambda _, saved: {**saved, "tag": 1})
    _ = optimizer.register_load_state_dict_pre_hook(lambda _, __: replacement)
    _ = optimizer.register_load_state_dict_post_hook(lambda _: calls.append("load"))
    saved = optimizer.state_dict()
    optimizer.load_state_dict(saved)
    assert "tag" in saved
    assert calls == ["save", "load"]
    assert optimizer.param_groups[0]["lr"] == 0.1


def test_deepcopy_keeps_the_members() -> None:
    # Torch pickles only defaults, state and param_groups, which drops them.
    model = split_model()
    optimizer = split_optimizer(model)
    backward(model)
    optimizer.step()
    copied = deepcopy(optimizer)
    assert repr(copied) == repr(optimizer)
    torch.testing.assert_close(
        copied.state_dict(),
        optimizer.state_dict(),
        rtol=0,
        atol=0,
    )


def test_defaults_hold_what_every_member_shares() -> None:
    # Torch's cycling schedulers read ``defaults`` to see if momentum exists.
    sgd = torch.optim.SGD([nn.Parameter(torch.ones(2))], lr=0.1, momentum=0.9)
    _ = torch.optim.lr_scheduler.CyclicLR(
        CompositeOptimizer([sgd]),
        base_lr=0.01,
        max_lr=0.1,
    )
    adamw = torch.optim.AdamW([nn.Parameter(torch.ones(2))], lr=0.1)
    assert "momentum" not in CompositeOptimizer([sgd, adamw]).defaults


def test_add_param_group_is_rejected() -> None:
    optimizer = split_optimizer(split_model())
    with pytest.raises(NotImplementedError, match="one of the composite's members"):
        optimizer.add_param_group({"params": []})


def test_repr_names_the_members() -> None:
    assert repr(split_optimizer(split_model())) == "CompositeOptimizer(SGD, Muon)"


def test_config_builds_over_the_models_parameters() -> None:
    model = nn.Linear(4, 4, bias=False)
    config = CompositeOptimizer.Config()
    config.optimizers = [Muon.Config()]
    optimizer = config.make()(model)
    assert isinstance(optimizer.optimizers[0], Muon)


def test_config_rejects_no_members() -> None:
    with pytest.raises(ValueError, match="needs a member"):
        _ = CompositeOptimizer.Config().make()


def test_config_routes_each_member_to_its_own_filter() -> None:
    """The whole point: one recipe, disjoint groups, one optimizer out."""
    model = split_model()
    on_muon = excluding(Muon.eligible_tensor, "head")
    config = CompositeOptimizer.Config()
    config.optimizers = [PartialConfig(torch.optim.SGD, lr=0.1), Muon.Config()]
    config.select = [complement(on_muon), on_muon]
    optimizer = config.make()(model)
    matrix = cast(list[Tensor], optimizer.optimizers[1].param_groups[0]["params"])
    assert [p.ndim for p in matrix] == [4]
    owned: list[int] = []
    for group in optimizer.param_groups:
        params = cast(list[Tensor], group["params"])
        owned.extend(id(parameter) for parameter in params)
    assert sorted(owned) == sorted(id(p) for p in model.parameters())


def test_config_rejects_a_filter_count_mismatch() -> None:
    config = CompositeOptimizer.Config()
    config.optimizers = [Muon.Config()]
    config.select = [everything, everything]
    with pytest.raises(ValueError, match="select names 2 filters"):
        _ = config.make()


def test_config_needs_a_filter_per_member_when_there_are_several() -> None:
    config = CompositeOptimizer.Config()
    config.optimizers = [Muon.Config(), Muon.Config()]
    with pytest.raises(ValueError, match="select names 0 filters for 2 optimizers"):
        _ = config.make()


def test_config_rejects_a_nested_composite() -> None:
    config = CompositeOptimizer.Config()
    config.optimizers = [CompositeOptimizer.Config()]
    with pytest.raises(ValueError, match="cannot hold a composite"):
        _ = config.make()


def test_config_freezes_a_parameter_no_filter_claims() -> None:
    # A gradient no member zeroes would grow every step and inflate clipping.
    model = nn.Linear(2, 3)
    config = CompositeOptimizer.Config()
    config.optimizers = [PartialConfig(torch.optim.SGD, lr=0.1)]
    config.select = [matching("weight")]
    config.require_total = False
    _ = config.make()(model)
    assert [p.requires_grad for p in model.parameters()] == [True, False]


def test_config_rejects_an_unclaimed_parameter() -> None:
    """A parameter nobody claims would silently never be updated."""
    config = CompositeOptimizer.Config()
    config.optimizers = [Muon.Config()]
    config.select = [Muon.eligible_tensor]
    with pytest.raises(
        ValueError,
        match=r"No filter claims 2 trainable parameter\(s\), e.g. '1.weight'; they would never be updated\.",
    ):
        _ = config.make()(split_model())


def test_config_rejects_two_filters_claiming_one_parameter() -> None:
    config = CompositeOptimizer.Config()
    config.optimizers = [Muon.Config(), Muon.Config()]
    config.select = [everything, everything]
    with pytest.raises(
        ValueError,
        match=r"Parameter '0.weight' is claimed by filter 0 and 1; it would be updated twice per step\.",
    ):
        _ = config.make()(split_model())


def test_config_rejects_a_filter_that_claims_nothing() -> None:
    config = CompositeOptimizer.Config()
    config.optimizers = [PartialConfig(torch.optim.SGD, lr=0.1), Muon.Config()]
    config.select = [complement(matching("ghost")), matching("ghost")]
    with pytest.raises(ValueError, match="Filter 1 claimed no parameters"):
        _ = config.make()(split_model())


def test_drop_empty_removes_a_member_whose_filter_claims_nothing() -> None:
    """An ablation naming a mechanism the model lacks keeps its siblings' rates."""
    config = CompositeOptimizer.Config()
    config.optimizers = [PartialConfig(torch.optim.SGD, lr=0.1), Muon.Config()]
    config.select = [complement(matching("ghost")), matching("ghost")]
    config.drop_empty = True
    optimizer = config.make()(split_model())
    assert [type(o).__name__ for o in optimizer.optimizers] == ["SGD"]


def test_drop_empty_keeps_members_after_an_empty_filter() -> None:
    config = CompositeOptimizer.Config()
    config.optimizers = [
        PartialConfig(torch.optim.SGD, lr=0.1),
        PartialConfig(torch.optim.AdamW, lr=0.1),
        PartialConfig(torch.optim.AdamW, lr=0.1),
    ]
    config.select = [matching("0."), matching("ghost"), complement(matching("0."))]
    config.drop_empty = True

    optimizer = config.make()(split_model())

    assert [type(member).__name__ for member in optimizer.optimizers] == [
        "SGD",
        "AdamW",
    ]
    assert [parameter_group_size(group) for group in optimizer.param_groups] == [1, 2]


def test_a_filter_receives_the_parameter_name() -> None:
    model = split_model()
    seen: list[str] = []

    class _Recording:
        def __call__(self, name: str, parameter: nn.Parameter) -> bool:
            del parameter
            seen.append(name)
            return True

    config = CompositeOptimizer.Config()
    config.optimizers = [PartialConfig(torch.optim.SGD, lr=0.1)]
    config.select = [_Recording()]
    _ = config.make()(model)
    assert seen == [name for name, _ in model.named_parameters()]


def test_state_iterates_over_every_members_keys() -> None:
    torch.manual_seed(0)
    model = split_model()
    optimizer = split_optimizer(model)
    backward(model)
    optimizer.step()
    state = optimizer.state
    # Plain SGD keeps no per-parameter state; Muon holds a momentum buffer.
    keys = list(state)
    assert {id(k) for k in keys} == {id(p) for p in model.parameters() if p.ndim >= 2}
    first = keys[0]
    assert isinstance(state[first], dict)


class _EditProgram(WeightProgramming):
    """Adds a fixed delta to each owned parameter."""

    class Config(Makes["Callable[..., _EditProgram]"], WeightProgramming.Config):
        """Fixed-delta programmer."""

    @override
    def program_weights(
        self,
        model: nn.Module,
        *,
        next_batch: Callable[[], dict[str, object]],
    ) -> UpdateOutput:
        del model, next_batch
        for group in self.param_groups:
            for param in cast("list[nn.Parameter]", group["params"]):
                with torch.no_grad():
                    param.add_(0.1)
        return UpdateOutput(updated=True, metrics={"edits": 1.0})


class _ObservingProgram(WeightProgramming):
    """Records requires_grad and epoch state during lifecycle."""

    class Config(
        Makes["Callable[..., _ObservingProgram]"],
        WeightProgramming.Config,
    ):
        """Observing programmer."""

    @override
    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        config: WeightProgramming.Config,
    ) -> None:
        super().__init__(params, config=config)
        self.observed_requires_grad: list[bool] = []
        self.epoch_end_count: int = 0

    @override
    def program_weights(
        self,
        model: nn.Module,
        *,
        next_batch: Callable[[], dict[str, object]],
    ) -> UpdateOutput:
        del model, next_batch
        self.observed_requires_grad = [
            p.requires_grad
            for group in self.param_groups
            for p in cast("list[nn.Parameter]", group["params"])
        ]
        for group in self.param_groups:
            for param in cast("list[nn.Parameter]", group["params"]):
                with torch.no_grad():
                    param.add_(0.01)
        return UpdateOutput(updated=True, metrics={"observed": 1.0})

    @override
    def on_epoch_end(self) -> None:
        self.epoch_end_count += 1


class _FailingProgram(WeightProgramming):
    """Raises during program_weights for failure-restoration tests."""

    class Config(
        Makes["Callable[..., _FailingProgram]"],
        WeightProgramming.Config,
    ):
        """Failing programmer."""

    @override
    def program_weights(
        self,
        model: nn.Module,
        *,
        next_batch: Callable[[], dict[str, object]],
    ) -> UpdateOutput:
        raise RuntimeError("Calibration exploded")


class _SequencedProgram(WeightProgramming):
    """Return incrementing call counts per step."""

    class Config(
        Makes["Callable[..., _SequencedProgram]"],
        WeightProgramming.Config,
    ):
        """Sequenced programmer."""

    @override
    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        config: WeightProgramming.Config,
    ) -> None:
        super().__init__(params, config=config)
        for group in self.param_groups:
            group["call_count"] = 0

    @override
    def program_weights(
        self,
        model: nn.Module,
        *,
        next_batch: Callable[[], dict[str, object]],
    ) -> UpdateOutput:
        del model, next_batch
        for group in self.param_groups:
            call_count = cast(int, group["call_count"]) + 1
            group["call_count"] = call_count
            for param in cast("list[nn.Parameter]", group["params"]):
                with torch.no_grad():
                    param.add_(0.01 * call_count)
        return UpdateOutput(
            updated=True,
            metrics={"call": float(cast(int, self.param_groups[0]["call_count"]))},
        )


class _AccumulatingProgram(WeightProgramming):
    """Stores running count in Optimizer.state for continuation testing."""

    class Config(
        Makes["Callable[..., _AccumulatingProgram]"],
        WeightProgramming.Config,
    ):
        """Accumulating programmer."""

    @override
    def program_weights(
        self,
        model: nn.Module,
        *,
        next_batch: Callable[[], dict[str, object]],
    ) -> UpdateOutput:
        del model, next_batch
        for group in self.param_groups:
            for param in cast("list[nn.Parameter]", group["params"]):
                state = cast("dict[str, Tensor]", self.state[param])
                if "count" not in state:
                    state["count"] = torch.tensor(0)
                state["count"] = state["count"] + 1
                with torch.no_grad():
                    param.add_(state["count"] * 0.01)
        return UpdateOutput(updated=True)


class _TensorMetricProgram(WeightProgramming):
    """Emits a fresh tensor metric each step for lifetime testing."""

    class Config(
        Makes["Callable[..., _TensorMetricProgram]"],
        WeightProgramming.Config,
    ):
        """Tensor-metric programmer."""

    @override
    def program_weights(
        self,
        model: nn.Module,
        *,
        next_batch: Callable[[], dict[str, object]],
    ) -> UpdateOutput:
        del model, next_batch
        return UpdateOutput(updated=True, metrics={"value": torch.tensor(1.0)})


class _NoChangeProgram(WeightProgramming):
    """Returns updated=False without modifying parameters."""

    class Config(
        Makes["Callable[..., _NoChangeProgram]"],
        WeightProgramming.Config,
    ):
        """No-change programmer."""

    @override
    def program_weights(
        self,
        model: nn.Module,
        *,
        next_batch: Callable[[], dict[str, object]],
    ) -> UpdateOutput:
        del model, next_batch
        return UpdateOutput(updated=False)


def test_mixed_recipe_routes_the_non_scorer_complement_to_sgd() -> None:
    model = nn.Module()
    model.add_module("body", nn.Linear(3, 4))
    model.add_module("embedding", nn.Embedding(5, 3))
    model.add_module("scorer", nn.Linear(4, 2))
    model.register_parameter(
        "frozen",
        nn.Parameter(torch.ones(6), requires_grad=False),
    )
    sgd = PartialConfig(torch.optim.SGD)
    sgd.lr = 0.001
    config = CompositeOptimizer.Config()
    config.optimizers = [sgd, Muon.Config(), _EditProgram.Config()]
    on_muon = excluding(Muon.eligible_tensor, "scorer.", "embedding")
    config.select = [
        excluding(complement(on_muon), "scorer."),
        on_muon,
        matching("scorer."),
    ]

    optimizer = config.make()(model)

    parameter_names = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    routed_names: list[list[str]] = []
    for member in optimizer.optimizers:
        names: list[str] = []
        for group in member.param_groups:
            parameters = cast(list[Tensor], group["params"])
            names.extend(parameter_names[id(parameter)] for parameter in parameters)
        routed_names.append(names)
    assert routed_names == [
        ["body.bias", "embedding.weight"],
        ["body.weight"],
        ["scorer.weight", "scorer.bias"],
    ]
    assert not model.get_parameter("frozen").requires_grad


def test_has_programming_false_for_gradient_only() -> None:
    optimizer = split_optimizer(split_model())
    assert not optimizer.has_programming


def test_has_programming_true_with_a_programmer() -> None:
    param = nn.Parameter(torch.randn(2, 3))
    programmer = _EditProgram.Config().make()([param])
    optimizer = CompositeOptimizer([programmer])
    assert optimizer.has_programming


def test_requires_gradients_true_for_gradient_only() -> None:
    optimizer = split_optimizer(split_model())
    assert optimizer.requires_gradients


def test_requires_gradients_true_for_mixed() -> None:
    param_a = nn.Parameter(torch.randn(2, 3))
    param_b = nn.Parameter(torch.randn(4))
    programmer = _EditProgram.Config().make()([param_a])
    sgd = torch.optim.SGD([param_b], lr=0.1)
    optimizer = CompositeOptimizer([sgd, programmer])
    assert optimizer.requires_gradients


def test_requires_gradients_false_for_programmers_only() -> None:
    param = nn.Parameter(torch.randn(2, 3))
    programmer = _EditProgram.Config().make()([param])
    optimizer = CompositeOptimizer([programmer])
    assert not optimizer.requires_gradients


def test_programming_enables_programmer_step() -> None:
    model = nn.Linear(3, 4, bias=False)
    programmer = _EditProgram.Config().make()(list(model.parameters()))
    optimizer = CompositeOptimizer([programmer])
    before = model.weight.detach().clone()
    with optimizer.programming(model, next_batch=_no_batch):
        optimizer.step()
    assert not torch.equal(model.weight.data, before)


def test_programming_binds_multiple_programmers() -> None:
    model = nn.Sequential(nn.Linear(3, 4, bias=False), nn.Linear(4, 2, bias=False))
    params_0 = list(model[0].parameters())
    params_1 = list(model[1].parameters())
    prog_0 = _ObservingProgram.Config().make()(params_0)
    prog_1 = _ObservingProgram.Config().make()(params_1)
    optimizer = CompositeOptimizer([prog_0, prog_1])
    with optimizer.programming(model, next_batch=_no_batch):
        optimizer.step()
    assert prog_0.output.updated
    assert prog_1.output.updated


def test_programming_restores_on_exit() -> None:
    model = nn.Linear(3, 4, bias=False)
    programmer = _EditProgram.Config().make()(list(model.parameters()))
    optimizer = CompositeOptimizer([programmer])
    with optimizer.programming(model, next_batch=_no_batch):
        optimizer.step()
    with pytest.raises(RuntimeError, match="requires a bound model"):
        optimizer.step()


def test_programming_restores_on_failure() -> None:
    model = nn.Linear(3, 4, bias=False)
    programmer = _FailingProgram.Config().make()(list(model.parameters()))
    optimizer = CompositeOptimizer([programmer])
    with (
        pytest.raises(RuntimeError, match="Calibration exploded"),
        optimizer.programming(model, next_batch=_no_batch),
    ):
        optimizer.step()
    with pytest.raises(RuntimeError, match="requires a bound model"):
        optimizer.step()


def test_programming_output_namespaced() -> None:
    model = nn.Sequential(nn.Linear(3, 4, bias=False), nn.Linear(4, 2, bias=False))
    sgd = torch.optim.SGD(list(model[0].parameters()), lr=0.1)
    programmer = _EditProgram.Config().make()(list(model[1].parameters()))
    optimizer = CompositeOptimizer([sgd, programmer])
    with optimizer.programming(model, next_batch=_no_batch):
        model(torch.randn(2, 3)).sum().backward()
        optimizer.step()
    output = optimizer.programming_output
    assert output.updated
    assert output.metrics == {"programming/1/edits": 1.0}


def test_programming_output_empty_without_programmers() -> None:
    optimizer = split_optimizer(split_model())
    assert optimizer.programming_output == UpdateOutput()


def test_on_epoch_end_forwards_to_programmers() -> None:
    param_a = nn.Parameter(torch.randn(2, 3))
    param_b = nn.Parameter(torch.randn(4, 5))
    prog_a = _ObservingProgram.Config().make()([param_a])
    prog_b = _ObservingProgram.Config().make()([param_b])
    sgd = torch.optim.SGD([nn.Parameter(torch.randn(2))], lr=0.1)
    optimizer = CompositeOptimizer([prog_a, sgd, prog_b])
    optimizer.on_epoch_end()
    assert prog_a.epoch_end_count == 1
    assert prog_b.epoch_end_count == 1
    optimizer.on_epoch_end()
    assert prog_a.epoch_end_count == 2


def test_mixed_gradient_and_programmer_step() -> None:
    """SGD uses backward gradients; programmer edits directly."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4, bias=False), nn.Linear(4, 2, bias=False))
    sgd_params = list(model[0].parameters())
    prog_params = list(model[1].parameters())
    sgd = torch.optim.SGD(sgd_params, lr=0.01)
    programmer = _EditProgram.Config().make()(prog_params)
    optimizer = CompositeOptimizer([sgd, programmer])

    sgd_before = sgd_params[0].detach().clone()
    prog_before = prog_params[0].detach().clone()

    with optimizer.programming(model, next_batch=_no_batch):
        output = model(torch.randn(2, 3))
        output.sum().backward()
        assert sgd_params[0].grad is not None
        assert prog_params[0].grad is None
        optimizer.step()

    assert not torch.equal(sgd_params[0].data, sgd_before)
    expected_prog = prog_before + 0.1
    assert torch.equal(prog_params[0].data, expected_prog)


def test_sibling_gradients_preserved() -> None:
    """Programmer step does not disturb sibling optimizer gradients."""
    model = nn.Sequential(nn.Linear(3, 4, bias=False), nn.Linear(4, 2, bias=False))
    sgd_params = list(model[0].parameters())
    prog_params = list(model[1].parameters())
    sgd = torch.optim.SGD(sgd_params, lr=0.01)
    programmer = _EditProgram.Config().make()(prog_params)
    optimizer = CompositeOptimizer([sgd, programmer])

    with optimizer.programming(model, next_batch=_no_batch):
        output = model(torch.randn(2, 3))
        output.sum().backward()
        assert sgd_params[0].grad is not None
        grad_before = sgd_params[0].grad.clone()
        optimizer.step()

    assert sgd_params[0].grad is not None
    assert torch.equal(sgd_params[0].grad, grad_before)


def test_pure_program_without_backward() -> None:
    """A composite with only programmers needs no forward/backward."""
    model = nn.Linear(3, 4, bias=False)
    programmer = _EditProgram.Config().make()(list(model.parameters()))
    optimizer = CompositeOptimizer([programmer])
    before = model.weight.detach().clone()

    with optimizer.programming(model, next_batch=_no_batch):
        optimizer.step()

    expected = before + 0.1
    assert torch.equal(model.weight.data, expected)


def test_programming_output_aggregates_multiple_steps() -> None:
    """Two steps in one binding retain last-wins metrics."""
    model = nn.Linear(3, 4, bias=False)
    programmer = _SequencedProgram.Config().make()(list(model.parameters()))
    optimizer = CompositeOptimizer([programmer])

    with optimizer.programming(model, next_batch=_no_batch):
        optimizer.step()
        optimizer.step()

    output = optimizer.programming_output
    assert output.updated
    assert output.metrics == {"programming/0/call": 2.0}


def test_programming_output_empty_in_fresh_binding() -> None:
    """A binding without steps produces an empty output."""
    model = nn.Linear(3, 4, bias=False)
    programmer = _EditProgram.Config().make()(list(model.parameters()))
    optimizer = CompositeOptimizer([programmer])

    with optimizer.programming(model, next_batch=_no_batch):
        optimizer.step()
    assert optimizer.programming_output.updated

    with optimizer.programming(model, next_batch=_no_batch):
        pass
    assert optimizer.programming_output == UpdateOutput()


def test_programming_output_includes_ordinary_member_fan_out() -> None:
    """programming_output.updated reflects ordinary member execution."""
    model = nn.Linear(3, 4)
    assert model.bias is not None
    sgd = torch.optim.SGD([model.weight], lr=0.1)
    programmer = _NoChangeProgram.Config().make()([model.bias])
    optimizer = CompositeOptimizer([sgd, programmer])

    assert not optimizer.programming_output.updated

    model.weight.grad = torch.ones_like(model.weight)
    with optimizer.programming(model, next_batch=_no_batch):
        optimizer.step()
        assert optimizer.programming_output.updated

    with optimizer.programming(model, next_batch=_no_batch):
        assert not optimizer.programming_output.updated


def test_programming_output_keeps_both_member_namespaces() -> None:
    """Repeated steps retain independently namespaced member metrics."""
    model = nn.Sequential(nn.Linear(3, 4, bias=False), nn.Linear(4, 2, bias=False))
    prog_a = _SequencedProgram.Config().make()(list(model[0].parameters()))
    prog_b = _SequencedProgram.Config().make()(list(model[1].parameters()))
    optimizer = CompositeOptimizer([prog_a, prog_b])

    with optimizer.programming(model, next_batch=_no_batch):
        optimizer.step()
        optimizer.step()

    output = optimizer.programming_output
    assert output.updated
    assert output.metrics == {
        "programming/0/call": 2.0,
        "programming/1/call": 2.0,
    }


def test_programmer_add_param_group_through_composite() -> None:
    """A programmer member accepts add_param_group; composite reflects it."""
    model = nn.Sequential(nn.Linear(3, 4, bias=False), nn.Linear(4, 2, bias=False))
    programmer = _EditProgram.Config().make()(list(model[0].parameters()))
    optimizer = CompositeOptimizer([programmer])

    new_param = nn.Parameter(torch.randn(2, 5))
    cast(WeightProgramming, optimizer.optimizers[0]).add_param_group(
        {"params": [new_param]},
    )
    assert not new_param.requires_grad
    assert len(optimizer.param_groups) == 2


def test_deepcopy_composite_preserves_programmer_state() -> None:
    """Deepcopy preserves programmer state: exact numerical continuation."""
    model = nn.Linear(3, 4, bias=False)
    programmer = _SequencedProgram.Config().make()(list(model.parameters()))
    optimizer = CompositeOptimizer([programmer])

    before = model.weight.detach().clone()
    with optimizer.programming(model, next_batch=_no_batch):
        optimizer.step()
    # Step 1: 0.01 * call_count(1) = 0.01.
    assert torch.equal(model.weight.data, before + 0.01)

    after_step1 = model.weight.detach().clone()
    model_copy, optimizer_copy = deepcopy((model, optimizer))

    with optimizer_copy.programming(model_copy, next_batch=_no_batch):
        optimizer_copy.step()
    # Step 2 on copy: 0.01 * call_count(2) = 0.02, proving count preserved.
    assert torch.equal(model_copy.weight.data, after_step1 + 0.02)
    # Original unchanged.
    assert torch.equal(model.weight.data, after_step1)


@pytest.mark.parametrize(
    ("source_program", "target_program"),
    [(True, False), (False, True), (True, True)],
)
def test_changed_programmer_identity_is_rejected_before_member_load(
    source_program: bool,
    target_program: bool,
) -> None:
    source_model = nn.Linear(3, 4)
    target_model = nn.Linear(3, 4)
    assert source_model.bias is not None
    assert target_model.bias is not None
    source_member = (
        _AccumulatingProgram.Config().make()([source_model.bias])
        if source_program
        else torch.optim.SGD([source_model.bias], lr=0.25)
    )
    target_member = (
        _EditProgram.Config().make()([target_model.bias])
        if target_program
        else torch.optim.SGD([target_model.bias], lr=0.25)
    )
    source = CompositeOptimizer(
        [torch.optim.SGD([source_model.weight], lr=0.5), source_member],
    )
    ordinary = torch.optim.SGD([target_model.weight], lr=0.125)
    target = CompositeOptimizer([ordinary, target_member])
    before = deepcopy(target.state_dict())
    calls: list[str] = []
    _ = ordinary.register_load_state_dict_pre_hook(
        lambda _, __: calls.append("ordinary"),
    )

    with pytest.raises(ValueError, match="programmer"):
        target.load_state_dict(source.state_dict())

    assert calls == []
    assert target.state_dict() == before


def test_stateful_programmer_resumes_through_composite() -> None:
    """Per-parameter state survives save/load; next step continues."""
    model = nn.Linear(2, 3, bias=False)
    programmer = _AccumulatingProgram.Config().make()(list(model.parameters()))
    composite = CompositeOptimizer([programmer])

    with composite.programming(model, next_batch=_no_batch):
        composite.step()
        composite.step()

    saved = composite.state_dict()

    model2 = nn.Linear(2, 3, bias=False)
    programmer2 = _AccumulatingProgram.Config().make()(list(model2.parameters()))
    composite2 = CompositeOptimizer([programmer2])
    composite2.load_state_dict(saved)

    param = model2.weight
    count = cast(Tensor, programmer2.state[param]["count"])
    assert count.item() == 2

    before = param.detach().clone()
    with composite2.programming(model2, next_batch=_no_batch):
        composite2.step()
    count_after = cast(Tensor, programmer2.state[param]["count"])
    assert count_after.item() == 3
    # Resumed count=3: delta = 3 * 0.01 = 0.03, not 1 * 0.01 if state lost.
    assert torch.equal(param.data, before + 0.03)


def test_nested_programming_rejection_preserves_outer_output() -> None:
    """Rejected nested programming() must not destroy the outer output."""
    model = nn.Linear(3, 4, bias=False)
    programmer = _EditProgram.Config().make()(list(model.parameters()))
    optimizer = CompositeOptimizer([programmer])

    with optimizer.programming(model, next_batch=_no_batch):
        optimizer.step()
        assert optimizer.programming_output.updated

        with (
            pytest.raises(RuntimeError, match="already bound"),
            optimizer.programming(model, next_batch=_no_batch),
        ):
            pass

        assert optimizer.programming_output.updated
        assert optimizer.programming_output.metrics == {"programming/0/edits": 1.0}


def test_composite_closure_differentiates_programmer_weights() -> None:
    """Fallback closure differentiates programmer-owned weights."""
    model = nn.Linear(3, 4, bias=False)
    programmer = _EditProgram.Config().make()(list(model.parameters()))
    optimizer = CompositeOptimizer([programmer])
    media = torch.ones(2, 3)
    expected = model(media).square().sum().detach()

    def closure() -> Tensor:
        loss = model(media).square().sum()
        loss.backward()
        return loss

    with optimizer.programming(model, next_batch=_no_batch), torch.no_grad():
        result = optimizer.step(closure)
    assert isinstance(result, Tensor)
    assert torch.equal(result, expected)
    assert model.weight.grad is not None
    assert not model.weight.requires_grad


def test_composite_closure_failure_refreezes_programmer_weights() -> None:
    """Failed fallback closure refreezes programmer-owned weights."""
    model = nn.Linear(3, 4, bias=False)
    programmer = _EditProgram.Config().make()(list(model.parameters()))
    optimizer = CompositeOptimizer([programmer])

    def closure() -> Tensor:
        model(torch.ones(2, 3)).sum().backward()
        raise RuntimeError("closure failed")

    with (
        pytest.raises(RuntimeError, match="closure failed"),
        optimizer.programming(model, next_batch=_no_batch),
    ):
        optimizer.step(closure)
    assert not model.weight.requires_grad


def test_mixed_closure_preserves_programmer_gradient_isolation() -> None:
    """Mixed fallback closure does not restore programmer eligibility."""
    model = nn.Linear(3, 4)
    assert model.bias is not None
    sgd = torch.optim.SGD([model.weight], lr=0.1)
    programmer = _EditProgram.Config().make()([model.bias])
    optimizer = CompositeOptimizer([sgd, programmer])
    media = torch.randn(2, 3)

    def closure() -> Tensor:
        loss = model(media).sum()
        loss.backward()
        return loss

    with optimizer.programming(model, next_batch=_no_batch), torch.no_grad():
        _ = optimizer.step(closure)
    assert model.weight.grad is not None
    assert model.bias.grad is None
    assert not model.bias.requires_grad


def test_pure_programming_closure_runs_once() -> None:
    """Fallback closure runs exactly once across two programmers."""
    model = nn.Sequential(nn.Linear(3, 4, bias=False), nn.Linear(4, 2, bias=False))
    prog_a = _EditProgram.Config().make()(list(model[0].parameters()))
    prog_b = _EditProgram.Config().make()(list(model[1].parameters()))
    optimizer = CompositeOptimizer([prog_a, prog_b])
    calls: list[int] = []

    def closure() -> Tensor:
        calls.append(1)
        loss = model(torch.ones(2, 3)).square().sum()
        loss.backward()
        return loss

    with optimizer.programming(model, next_batch=_no_batch), torch.no_grad():
        _ = optimizer.step(closure)
    assert len(calls) == 1
    for p in model.parameters():
        assert p.grad is not None
        assert not p.requires_grad


def test_programming_output_does_not_retain_superseded_metric_tensors() -> None:
    """Superseded metric tensors become collectible after last-wins replacement."""
    model = nn.Linear(3, 4, bias=False)
    programmer = _TensorMetricProgram.Config().make()(list(model.parameters()))
    optimizer = CompositeOptimizer([programmer])

    with optimizer.programming(model, next_batch=_no_batch):
        optimizer.step()
        ref = weakref.ref(optimizer.programming_output.metrics["programming/0/value"])
        optimizer.step()

    assert ref() is None


def test_deepcopy_composite_preserves_autograd_eligibility() -> None:
    """Deepcopy preserves gradient eligibility: autograd works in the copy."""
    model = nn.Linear(3, 4, bias=False)
    programmer = _ObservingProgram.Config().make()(list(model.parameters()))
    optimizer = CompositeOptimizer([programmer])
    assert not model.weight.requires_grad

    model_copy, optimizer_copy = deepcopy((model, optimizer))
    prog_copy = cast(_ObservingProgram, optimizer_copy.optimizers[0])

    before = model_copy.weight.detach().clone()
    with optimizer_copy.programming(model_copy, next_batch=_no_batch):
        optimizer_copy.step()
    # requires_grad restored during program_weights in the copy.
    assert prog_copy.observed_requires_grad == [True]
    assert not torch.equal(model_copy.weight.data, before)
    assert not model_copy.weight.requires_grad
    # Original unchanged.
    assert not model.weight.requires_grad


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
