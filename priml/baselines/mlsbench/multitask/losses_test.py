"""Tests for the loss combinations, against their closed forms."""

from __future__ import annotations

import math

from torch import nn

import pytest
import torch

from priml.baselines.mlsbench.multitask.losses import (
    DynamicWeightAverage,
    PCGrad,
    SumLoss,
    UncertaintyLoss,
    graph_leaves,
)


def test_sum_adds_the_two_losses() -> None:
    loss = SumLoss.Config().make()(torch.tensor(1.5), torch.tensor(0.25), 0, 3)
    assert loss.item() == 1.75


def test_uncertainty_weights_by_precision_and_penalizes_log_variance() -> None:
    module = UncertaintyLoss.Config().make()
    assert [n for n, _ in module.named_parameters()] == ["log_vars"]
    with torch.no_grad():
        module.log_vars.copy_(torch.tensor([0.0, math.log(2.0)]))
    loss = module(torch.tensor(3.0), torch.tensor(4.0), 0, 3)
    assert loss.item() == pytest.approx(3.0 + 4.0 / 2 + math.log(2.0))
    loss.backward()
    assert module.log_vars.grad is not None
    assert module.log_vars.grad.tolist() == pytest.approx([1 - 3.0, 1 - 2.0])


def test_dwa_weights_equally_in_the_first_epoch_and_by_loss_ratio_after() -> None:
    module = DynamicWeightAverage.Config().make()
    first = module(torch.tensor(2.0), torch.tensor(4.0), 0, 3)
    assert first.item() == 6.0
    fine, coarse = torch.tensor(1.0), torch.tensor(4.0)
    second = module(fine, coarse, 1, 3)
    ratios = torch.tensor([1.0 / 2.0, 4.0 / 4.0]) / 2.0
    weights = 2 * torch.softmax(ratios, dim=0)
    assert second.item() == pytest.approx(
        (weights * torch.tensor([1.0, 4.0])).sum().item(),
    )
    assert module.prev_losses is not None
    assert module.prev_losses.tolist() == [1.0, 4.0]
    restored = DynamicWeightAverage.Config().make()
    restored.load_state_dict(module.state_dict())
    assert restored.prev_losses is not None
    assert restored.prev_losses.tolist() == [1.0, 4.0]


def test_dwa_resets_to_equal_weights_whenever_the_epoch_is_zero() -> None:
    module = DynamicWeightAverage.Config().make()
    _ = module(torch.tensor(2.0), torch.tensor(4.0), 0, 3)
    assert module(torch.tensor(1.0), torch.tensor(1.0), 0, 3).item() == 2.0


def test_dwa_moves_a_restored_history_to_the_losses_device() -> None:
    dwa = DynamicWeightAverage.Config().make()
    dwa.set_extra_state(torch.tensor([1.0, 2.0]))
    meta = torch.device("meta")
    loss = dwa(torch.tensor(1.0, device=meta), torch.tensor(2.0, device=meta), 1, 3)
    assert loss.device == meta


def test_pcgrad_leaves_agreeing_gradients_summed() -> None:
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    module = PCGrad.Config().make()
    module(((w - 3) ** 2).sum(), (w**2).sum() * 0 + w.sum() * -1, 0, 3).backward()
    assert w.grad is not None
    assert w.grad.tolist() == [-4.0 - 1.0, -2.0 - 1.0]


def test_pcgrad_projects_conflicting_gradients_off_each_other() -> None:
    w = torch.tensor([1.0, 0.0], requires_grad=True)
    module = PCGrad.Config().make()
    # Gradients (1, 1) and (-1, 0): the dot product is -1.
    module((w[0] + w[1]) * 1, -w[0], 0, 3).backward()
    g0, g1 = torch.tensor([1.0, 1.0]), torch.tensor([-1.0, 0.0])
    p0 = g0 - (-1 / (1 + module.eps)) * g1
    p1 = g1 - (-1 / (2 + module.eps)) * g0
    assert w.grad is not None
    assert torch.equal(w.grad, p0 + p1)


def test_pcgrad_keeps_the_leaves_of_its_first_call() -> None:
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)
    module = PCGrad.Config().make()
    module(a * 2, a * 3, 0, 3).backward()
    module(a * b, b, 0, 3).backward()
    assert module.shared == [a]
    assert b.grad is None


def test_pcgrad_sums_losses_that_reach_no_leaf() -> None:
    module = PCGrad.Config().make()
    x = torch.tensor(2.0, requires_grad=True)
    loss = module(torch.tensor(1.0) * 1, x * 3, 0, 3)
    assert loss.item() == 1.0 + 6.0
    assert module.shared == []


def test_graph_leaves_finds_every_parameter_once_in_walk_order() -> None:
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 4))
    x = torch.randn(5, 2)
    h = net(x)
    loss = (h + net(x)).sum()
    leaves = graph_leaves(loss)
    assert sorted(id(p) for p in leaves) == sorted(id(p) for p in net.parameters())
    assert leaves[0] is net[2].bias


@pytest.mark.parametrize(
    ("config", "params"),
    [
        (SumLoss.Config(), 0),
        (UncertaintyLoss.Config(), 2),
        (DynamicWeightAverage.Config(), 0),
        (PCGrad.Config(), 0),
    ],
    ids=["sum", "uncertainty", "dwa", "pcgrad"],
)
def test_every_combination_costs_a_few_scalar_ops_and_owns_its_parameters(
    config: SumLoss.Config
    | UncertaintyLoss.Config
    | DynamicWeightAverage.Config
    | PCGrad.Config,
    params: int,
) -> None:
    cost = config.cost(batch_size=7, dtype=None)
    assert cost.params == params == sum(p.numel() for p in config.make().parameters())
    assert cost == config.cost(batch_size=128, dtype=None)  # O(1) in the batch.
    assert cost["flops", "primal", "matmul"].sum() == 0


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
