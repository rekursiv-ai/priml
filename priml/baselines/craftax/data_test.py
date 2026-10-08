"""Tests for the cadence: endless ticks, the checkpoint, and the evaluation batches."""

from __future__ import annotations

from typing import cast

import itertools

import numpy as np
import pytest

from priml.baselines.craftax.data import CraftaxRollouts
from priml.baselines.craftax.evaluation import Played
from priml.baselines.craftax.game.state import LOG_DTYPE
from priml.train.custom_types import TrainStepProtocol


def _rollouts() -> CraftaxRollouts:
    return CraftaxRollouts.Config().make()


def test_the_cadence_ticks_once_per_training_step_without_end() -> None:
    """More than exp003's 15,258 epochs: the loop's ``max_steps`` ends a run."""
    ticks = list(itertools.islice(_rollouts().train_dataloader(), 20_000))

    assert len(ticks) == 20_000
    assert all(tick == {"valid_count": 1} for tick in ticks)
    # Fresh dicts, so a consumer that edits one edits no other tick.
    assert ticks[0] is not ticks[1]


def test_a_checkpoint_carries_the_pass_count_alone() -> None:
    rollouts = _rollouts()
    rollouts.timer_epoch.global_count = 3
    resumed = _rollouts()
    resumed.load_state_dict(rollouts.state_dict())

    assert list(rollouts.state_dict()) == ["timer_epoch"]
    assert resumed.timer_epoch.global_count == 3


class _Evaluator:
    def __init__(self, played: Played) -> None:
        self.played = played
        self.closed = False

    def play(self) -> Played:
        return self.played

    def close(self) -> None:
        self.closed = True


class _Step:
    """Hands out one evaluator per call, as a train step's ``make_evaluator`` does."""

    def __init__(self) -> None:
        self.made: list[_Evaluator] = []

    def make_evaluator(self) -> _Evaluator:
        evaluator = _Evaluator(
            Played(logs=np.zeros(1, dtype=LOG_DTYPE), rollouts=1, gameplay_seconds=0.0),
        )
        self.made.append(evaluator)
        return evaluator


def test_an_eval_is_one_fresh_evaluation_played_and_closed() -> None:
    rollouts = _rollouts()
    step = _Step()
    rollouts.bind_step(cast(TrainStepProtocol, step))

    loader = rollouts.eval_dataloader()
    assert step.made == []
    batches = list(loader)

    assert len(batches) == len(step.made) == 1
    assert batches[0]["played"] is step.made[0].played
    assert batches[0]["metric_only"] is True
    assert step.made[0].closed


def test_evaluation_without_a_bound_step_fails_loudly() -> None:
    with pytest.raises(TypeError, match="make_evaluator"):
        _rollouts().eval_dataloader()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
