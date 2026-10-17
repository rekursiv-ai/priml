"""Tests for the multi-task step: optimizer order, the update, eval and checkpoints."""

from __future__ import annotations

from typing import Final, cast

from configgle import Makeable, PartialConfig
from torch import Tensor, nn

import torch

from priml.baselines.mlsbench.multitask.losses import (
    DynamicWeightAverage,
    UncertaintyLoss,
)
from priml.baselines.mlsbench.multitask.model import TwoHeadResNet
from priml.baselines.mlsbench.multitask.train_step import MultiTaskStep
from priml.baselines.mlsbench.pytorch_vision.train_step import cross_entropy
from priml.train.parallelism import NoParallel


CLASSES: Final = 3
COARSE: Final = 2


def step(mtl_loss: Makeable[nn.Module] | None = None) -> MultiTaskStep:
    """Build a tiny step: widths 2, 4, 5, one block per stage, SGD."""
    config = MultiTaskStep.Config()
    assert isinstance(config.model, TwoHeadResNet.Config)
    config.model.channels_out = COARSE
    config.model.trunk.channels_in = 3
    config.model.trunk.channels_out = CLASSES
    config.model.trunk.channels_hidden = (2, 4, 5)
    config.model.trunk.blocks_per_stage = (1, 1, 1)
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
    made.timer_epoch.global_count = 0
    first = made.train_step(**batch())
    remembered = prev_losses(made).clone()
    made.timer_epoch.global_count = 1
    second = made.train_step(**batch())
    # Epoch 0 weights both losses by one; epoch 1 weights by their ratios.
    assert torch.equal(first["loss"], remembered.sum())
    assert not torch.equal(second["loss"], prev_losses(made).sum())


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
    unrestored = step(DynamicWeightAverage.Config())
    unrestored.load_state_dict(state)
    assert isinstance(unrestored.mtl_loss, DynamicWeightAverage)
    assert unrestored.mtl_loss.prev_losses is None


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
