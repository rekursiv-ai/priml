"""Training is a function of its config: tiny CPU runs through the reference slots.

The GPU goldens pin the bits. These pin what makes them meaningful on any
host: two fresh runs agree bit for bit, and an evaluation between epochs moves
nothing the training reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from priml.baselines.craftax.testing import tiny_train_step
from priml.baselines.craftax.train_step import AgentWindows
from priml.optimizers.fused_muon import FusedMuon


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.craftax.train_step import CraftaxTrainStep


def _masters(step: CraftaxTrainStep) -> list[Tensor]:
    optimizer = step.optimizer
    assert isinstance(optimizer, FusedMuon)
    return [master.clone() for master in optimizer.master_weights]


# Rollouts of 2 steps, one window of every agent: the epoch's parts are the train
# step's tests'. The evaluation plays one rollout from its reset, as ``play`` begins.
def _train(*, evaluate_between: bool) -> list[Tensor]:
    """Run two of the tiny pipeline's epochs; return the masters."""
    config = tiny_train_step()
    config.rollout.horizon = 2
    windows = config.learner
    assert isinstance(windows, AgentWindows.Config)
    windows.minibatch_size = 2 * config.env.num_envs
    torch.manual_seed(0)
    step = config.make()
    try:
        step.train_step()
        if evaluate_between:
            evaluation = step.make_evaluator()
            try:
                evaluation.reset()
                evaluation.collect()
            finally:
                evaluation.close()
        step.train_step()
        return _masters(step)
    finally:
        step.close()


def _same(first: list[Tensor], second: list[Tensor]) -> bool:
    return all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_two_fresh_runs_train_identically() -> None:
    assert _same(_train(evaluate_between=False), _train(evaluate_between=False))


def test_an_evaluation_between_epochs_changes_nothing() -> None:
    assert _same(_train(evaluate_between=False), _train(evaluate_between=True))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
