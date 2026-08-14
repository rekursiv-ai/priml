"""Tests for the rollout cadence."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from priml.baselines.craftax.data import CraftaxRollouts
from priml.data.custom_types import DatasetProtocol
from priml.train.custom_types import TrainStepProtocol


def _rollouts(**overrides: int) -> CraftaxRollouts:
    config = CraftaxRollouts.Config()
    config.updates_per_epoch = 3
    config.eval_batches = 2
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def test_it_satisfies_the_dataset_protocol() -> None:
    assert isinstance(_rollouts(), DatasetProtocol)


def test_an_epoch_is_as_long_as_configured() -> None:
    assert len(list(_rollouts().train_dataloader())) == 3


def test_an_evaluation_runs_the_configured_passes() -> None:
    assert len(list(_rollouts().eval_dataloader())) == 2


def test_each_tick_carries_a_unit_weight() -> None:
    # The loop weights an evaluation by ``valid_count``; one per pass makes
    # the score an average over passes rather than over environments.
    for batch in _rollouts().eval_dataloader():
        assert batch["valid_count"] == 1


def test_the_step_can_be_bound() -> None:
    rollouts = _rollouts()
    marker = object()
    rollouts.bind_step(cast("TrainStepProtocol", marker))
    assert rollouts._step is marker


def test_an_evaluation_batch_carries_the_bound_policy() -> None:
    # This is the whole point of ``bind_step``: the score is a property of the
    # network, so the metric has to be handed the one that is training.
    rollouts = _rollouts()
    step = _Step()
    rollouts.bind_step(cast("TrainStepProtocol", step))
    for batch in rollouts.eval_dataloader():
        assert batch["policy"] is step.model


def test_an_unbound_dataset_yields_no_policy() -> None:
    for batch in _rollouts().eval_dataloader():
        assert "policy" not in batch


def test_training_batches_carry_no_policy() -> None:
    # The step already owns its network; sending it back would invite a
    # training path that reads the model from the batch instead.
    rollouts = _rollouts()
    rollouts.bind_step(cast("TrainStepProtocol", _Step()))
    for batch in rollouts.train_dataloader():
        assert "policy" not in batch


class _Step:
    """A stand-in training step carrying only the field the dataset reads.

    Cast at each call site rather than implementing the whole training-step
    protocol: ``bind_step`` stores what it is given and the dataset reads one
    attribute off it, so the rest of the protocol is not exercised here.
    """

    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1)


def test_it_carries_no_resumable_state() -> None:
    # The environment holds everything a resume needs, so a checkpoint here
    # would be a second, conflicting copy.
    rollouts = _rollouts()
    assert rollouts.state_dict() == {}
    rollouts.load_state_dict({"anything": 1})


def test_a_fresh_iterator_is_returned_each_epoch() -> None:
    rollouts = _rollouts()
    assert len(list(rollouts.train_dataloader())) == 3
    assert len(list(rollouts.train_dataloader())) == 3


@pytest.mark.parametrize("field", ["updates_per_epoch", "eval_batches"])
def test_an_empty_cadence_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        _rollouts(**{field: 0})


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
