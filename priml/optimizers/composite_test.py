"""Tests for CompositeOptimizer."""

from __future__ import annotations

from configgle import PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.optimizers.composite import (
    CompositeOptimizer,
    complement,
    everything,
    excluding,
)
from priml.optimizers.muon import Muon
from priml.optimizers.newton import Newton


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


def backward(model: nn.Module) -> None:
    """Populate gradients on every parameter."""
    model(torch.randn(8, 3, 8, 8)).sum().backward()


def test_is_an_optimizer() -> None:
    # The whole point: a caller written for one optimizer takes a composite.
    assert isinstance(split_optimizer(split_model()), torch.optim.Optimizer)


def test_exposes_every_members_parameter_groups() -> None:
    model = split_model()
    optimizer = split_optimizer(model)
    owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
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
    for member, other in zip(
        optimizer.optimizers,
        restored.optimizers,
        strict=True,
    ):
        assert member.state_dict()["param_groups"] == other.state_dict()["param_groups"]


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


def test_step_returns_the_closure_value() -> None:
    model = split_model()
    optimizer = split_optimizer(model)
    backward(model)
    assert optimizer.step(lambda: 42.0) == 42.0


def test_closure_is_evaluated_once() -> None:
    # Forwarding it to each member would recompute the loss per member.
    calls: list[int] = []
    model = split_model()
    optimizer = split_optimizer(model)
    backward(model)
    _ = optimizer.step(lambda: calls.append(1) or 0.0)
    assert len(calls) == 1


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
    with pytest.raises(ValueError, match="more than one optimizer"):
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
    assert isinstance(optimizer, CompositeOptimizer)
    assert isinstance(optimizer.optimizers[0], Muon)


def test_config_rejects_no_members() -> None:
    with pytest.raises(ValueError, match="needs a member"):
        _ = CompositeOptimizer.Config().make()


def test_config_routes_each_member_to_its_own_selector() -> None:
    """The whole point: one recipe, disjoint groups, one optimizer out."""
    model = split_model()
    on_muon = excluding(Muon.eligible_tensor, "head")
    config = CompositeOptimizer.Config()
    config.optimizers = [PartialConfig(torch.optim.SGD, lr=0.1), Muon.Config()]
    config.select = [complement(on_muon), on_muon]
    optimizer = config.make()(model)
    matrix = optimizer.optimizers[1].param_groups[0]["params"]
    assert [p.ndim for p in matrix] == [4]
    owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert sorted(owned) == sorted(id(p) for p in model.parameters())


def test_config_rejects_a_selector_count_mismatch() -> None:
    config = CompositeOptimizer.Config()
    config.optimizers = [Muon.Config()]
    config.select = [everything, everything]
    with pytest.raises(ValueError, match="select names 2 selectors"):
        _ = config.make()


def test_config_rejects_an_unclaimed_parameter() -> None:
    """A parameter nobody claims would silently never be updated."""
    config = CompositeOptimizer.Config()
    config.optimizers = [Muon.Config()]
    config.select = [Muon.eligible_tensor]
    with pytest.raises(ValueError, match="No selector claims"):
        _ = config.make()(split_model())


def test_config_rejects_two_selectors_claiming_one_parameter() -> None:
    config = CompositeOptimizer.Config()
    config.optimizers = [Muon.Config(), Muon.Config()]
    config.select = [everything, everything]
    with pytest.raises(ValueError, match="claimed by selector"):
        _ = config.make()(split_model())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
