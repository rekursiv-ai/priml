"""Tests for CompositeOptimizer."""

from __future__ import annotations

from copy import deepcopy
from typing import Protocol, TypeGuard, cast, runtime_checkable

import gc
import weakref

from configgle import PartialConfig
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
    _ = gc.collect()
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


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
