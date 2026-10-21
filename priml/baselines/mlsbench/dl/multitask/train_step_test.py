"""Tests for the multi-task step: optimizer order, the update, eval and checkpoints."""

from __future__ import annotations

from typing import Final, cast

from configgle import Makeable, PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.mlsbench.dl.multitask.losses import (
    DynamicWeightAverage,
    PCGrad,
    UncertaintyLoss,
)
from priml.baselines.mlsbench.dl.multitask.model import TwoHeadResNet
from priml.baselines.mlsbench.dl.multitask.train_step import MultiTaskStep
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.train_step import cross_entropy
from priml.train.parallelism import NoParallel


CLASSES: Final = 3
COARSE: Final = 2


def step(mtl_loss: Makeable[nn.Module] | None = None) -> MultiTaskStep:
    """Build a tiny step: widths 2, 4, 5, one block per stage, SGD."""
    config = MultiTaskStep.Config()
    assert isinstance(config.model, TwoHeadResNet.Config)
    config.model.channels_out = COARSE
    trunk = config.model.trunk
    assert isinstance(trunk, ResNet.Config)
    trunk.channels_in = 3
    trunk.channels_out = CLASSES
    trunk.channels_hidden = (2, 4, 5)
    trunk.blocks_per_stage = (1, 1, 1)
    sgd = config.optimizer = PartialConfig(torch.optim.SGD)
    sgd.lr = 0.1
    config.loss = PartialConfig(cross_entropy)
    if mtl_loss is not None:
        config.mtl_loss = mtl_loss
    config.parallelism = NoParallel.Config(device="cpu")
    config.train_budget_epochs = 4
    return config.make()


def batch() -> dict[str, Tensor]:
    """Two 6x7 images with fine and coarse labels."""
    return {
        "media": torch.arange(2 * 3 * 6 * 7.0).reshape(2, 3, 6, 7) / 100,
        "label": torch.tensor([0, 2]),
        "label_coarse": torch.tensor([1, 0]),
    }


def log_vars(made: MultiTaskStep) -> Tensor:
    """Return the uncertainty combination's learned log-variances."""
    assert isinstance(made.mtl_loss, UncertaintyLoss)
    return made.mtl_loss.log_vars


def prev_losses(made: MultiTaskStep) -> Tensor:
    """Return the weight-average combination's remembered losses."""
    assert isinstance(made.mtl_loss, DynamicWeightAverage)
    assert made.mtl_loss.prev_losses is not None
    return made.mtl_loss.prev_losses


def test_an_unbounded_budget_is_refused_by_the_vision_check() -> None:
    config = MultiTaskStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    with pytest.raises(ValueError, match="train_budget_epochs must be finite"):
        config.make()


def test_the_optimizer_names_parameters_by_their_owner() -> None:
    made = step(UncertaintyLoss.Config())
    names = [name for name, _ in made.optimized.named_parameters()]
    assert names == [
        *(f"model.{name}" for name, _ in made.model.named_parameters()),
        "mtl_loss.log_vars",
    ]


@pytest.mark.parametrize(
    "mtl_loss",
    [UncertaintyLoss.Config(), DynamicWeightAverage.Config(), PCGrad.Config()],
    ids=["uncertainty", "dwa", "pcgrad"],
)
def test_the_reported_loss_is_the_sum_of_the_two_heads(
    mtl_loss: Makeable[nn.Module],
) -> None:
    made = step(mtl_loss)
    assert isinstance(made.model, TwoHeadResNet)
    before = {k: v.clone() for k, v in made.model.state_dict().items()}
    result = made.train_step(**batch())
    made.model.load_state_dict(before)
    made.model.train()
    with torch.no_grad():
        fine, coarse = made.model(batch()["media"])
    expected = torch.nn.functional.cross_entropy(
        fine,
        batch()["label"],
    ) + torch.nn.functional.cross_entropy(coarse, batch()["label_coarse"])
    assert torch.equal(result["loss"], expected)


def test_the_combinations_parameters_follow_the_networks() -> None:
    made = step(UncertaintyLoss.Config())
    params = cast(list[Tensor], made.optimizer.param_groups[0]["params"])
    assert [id(p) for p in params] == [
        *(id(p) for p in made.model.parameters()),
        id(log_vars(made)),
    ]


def test_an_update_moves_the_network_and_the_combination() -> None:
    made = step(UncertaintyLoss.Config())
    before = log_vars(made).detach().clone()
    result = made.train_step(**batch())
    assert result["loss"].shape == ()
    assert result["model"].shape == (2, CLASSES + COARSE)
    assert not torch.equal(log_vars(made), before)
    assert made.global_step == 1


def test_the_combination_is_told_the_epoch() -> None:
    made = step(DynamicWeightAverage.Config())
    epochs: list[int] = []
    inner = made.mtl_loss

    def record_epoch(module: nn.Module, args: tuple[object, ...]) -> None:
        del module
        epochs.append(cast(int, args[2]))

    made.mtl_loss.register_forward_pre_hook(record_epoch)
    for epoch in (0, 1):
        made.timer_epoch.global_count = epoch
        _ = made.train_step(**batch())
    assert epochs == [0, 1]
    assert made.mtl_loss is inner
    assert prev_losses(made).shape == (2,)


def test_eval_scores_the_fine_head_and_returns_both() -> None:
    made = step()
    result = made.eval_loss(**batch())
    fine = result["model"][:, :CLASSES]
    expected = torch.nn.functional.cross_entropy(fine, batch()["label"])
    assert torch.equal(result["loss"], expected)
    assert result["model"].shape == (2, CLASSES + COARSE)


def test_a_checkpoint_restores_the_combinations_state() -> None:
    made = step(DynamicWeightAverage.Config())
    _ = made.train_step(**batch())
    state = made.state_dict()
    fresh = step(DynamicWeightAverage.Config())
    fresh.load_state_dict(state)
    assert torch.equal(prev_losses(fresh), prev_losses(made))
    state.pop("mtl_loss")
    with pytest.raises(KeyError, match="mtl_loss"):
        step(DynamicWeightAverage.Config()).load_state_dict(state)
    unrestored = step(DynamicWeightAverage.Config())
    unrestored.load_state_dict(state, strict=False)
    assert isinstance(unrestored.mtl_loss, DynamicWeightAverage)
    assert unrestored.mtl_loss.prev_losses is None


def test_pcgrad_leaves_the_coarse_head_unchanged() -> None:
    """Upstream's PCGrad takes leaves from the fine loss only, so the coarse head never trains."""
    made = step(PCGrad.Config())
    assert isinstance(made.model, TwoHeadResNet)
    before = {k: v.clone() for k, v in made.model.head.state_dict().items()}
    fine = made.model.trunk.get_submodule("head")
    assert isinstance(fine, nn.Linear)
    trunk = fine.weight.detach().clone()
    _ = made.train_step(**batch())
    for name, value in made.model.head.state_dict().items():
        assert torch.equal(value, before[name]), name
    assert not torch.equal(fine.weight, trunk)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
