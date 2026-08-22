"""Tests for the Q-learning training step."""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import copy
import math

import pytest
import torch

from priml.baselines.craftax.pqn_train_step import CraftaxPQNTrainStep
from priml.train.custom_types import TrainStepOutput, TrainStepProtocol
from priml.train.parallelism import NoParallel


pytestmark = pytest.mark.compute_training


def _config(**overrides: object) -> CraftaxPQNTrainStep.Config:
    config = CraftaxPQNTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.env.device = "cpu"
    config.env.num_envs = 2
    config.env.optimistic_reset_ratio = 1
    # A 3x3 view, not the benchmark's 9x11: these test the UPDATE, and a
    # 8,268-wide observation makes the first layer dominate every one.
    config.env.view = (3, 3)
    config.rollout_steps = 2
    config.num_epochs = 1
    config.num_minibatches = 1
    config.total_train_steps = 10
    config.model.hidden_size = 4
    for name, value in overrides.items():
        setattr(config, name, value)
    if "seed" in overrides:
        config.env.seed = int(config.seed)
    return config


def _step() -> CraftaxPQNTrainStep:
    return cast("CraftaxPQNTrainStep", _config().make())  # pyright: ignore[reportUnnecessaryCast] -- ty reads `Makes[...].make()` as @Todo and needs the cast


def _metrics(result: TrainStepOutput) -> dict[str, float | torch.Tensor]:
    """Read the optional diagnostics a completed update always carries."""
    return result.get("metrics", {})


def test_it_satisfies_the_training_step_protocol() -> None:
    assert isinstance(_step(), TrainStepProtocol)


def test_the_network_is_sized_from_the_environment() -> None:
    step = _step()
    assert step.model.encoder.in_features == step.env.observation_size
    assert step.model.head.out_features == step.env.num_actions


def test_a_step_optimizes_and_reports_its_diagnostics() -> None:
    result = _step().train_step()
    assert math.isfinite(float(result["loss"]))
    for name in ("q_loss", "q_mean", "grad_norm", "learning_rate", "epsilon"):
        assert math.isfinite(float(_metrics(result)[name])), name


def test_each_epoch_visits_every_minibatch() -> None:
    step = _config(num_epochs=2, num_minibatches=2).make()
    with patch.object(step.optimizer, "step", wraps=step.optimizer.step) as optimizer:
        step.train_step()
    assert optimizer.call_count == 4


def test_a_step_changes_the_network() -> None:
    step = _step()
    before = step.model.encoder.weight.detach().clone()
    step.train_step()
    assert not torch.equal(before, step.model.encoder.weight.detach())


def test_it_trains_without_a_target_network() -> None:
    """The whole claim: one network, regressed against its own values.

    A second network would show up here as a second copy of the weights, so
    this is checkable rather than merely stated.
    """
    step = _step()
    modules = {name for name, _ in step.model.named_modules() if name}
    assert not any("target" in name for name in modules)


def test_it_keeps_no_replay_buffer() -> None:
    # Every rollout is discarded after its update; nothing accumulates.
    step = _step()
    step.train_step()
    step.train_step()
    saved = step.state_dict()
    assert not any("buffer" in key or "replay" in key for key in saved)


def test_targets_are_built_once_before_optimizing() -> None:
    """Recomputing them per epoch would chase a value already moved.

    The rollout carries its targets, so all optimization passes
    regress toward the same numbers.
    """
    step = _step()
    rollout = step.collect()
    before = rollout.target.clone()
    list(rollout.minibatches(count=2))
    assert torch.equal(before, rollout.target)


def test_a_rollout_records_both_recurrent_tensors() -> None:
    step = _step()
    rollout = step.collect()
    assert rollout.hidden.shape == (2, 4)
    assert rollout.cell.shape == (2, 4)
    assert rollout.previous_action.shape == (2, 2)


def test_exploration_decays_across_the_run() -> None:
    step = _config(total_train_steps=10, epsilon_decay_fraction=1.0).make()
    rates: list[float] = []
    for _ in range(3):
        rates.append(step.epsilon)
        step.train_step()
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] < rates[0]


def test_full_exploration_ignores_the_networks_preference() -> None:
    # At epsilon 1 every action is random, which is what makes the first
    # updates informative rather than a self-fulfilling greedy loop.
    config = _config(epsilon_start=1.0, epsilon_finish=1.0)
    config.env.num_envs = 4
    step = config.make()
    rollout = step.collect()
    # A 43-action space over eight slots: an argmax-only policy would repeat
    # itself far more than this.
    assert len(set(rollout.action.flatten().tolist())) > 4


def test_no_exploration_takes_the_greedy_action() -> None:
    step = _config(epsilon_start=0.0, epsilon_finish=0.0).make()
    rollout = step.collect()
    assert torch.equal(rollout.action, rollout.q_value.argmax(dim=-1))


def test_collection_does_not_update_the_running_statistics() -> None:
    """A rollout is inference.

    Folding it into the normalization would count every observation twice per
    update -- once acting, once learning.
    """
    step = _step()
    before = int(step.model.normalize.steps)
    step.collect()
    assert int(step.model.normalize.steps) == before


def test_the_same_seed_reproduces_the_same_update() -> None:
    def run() -> float:
        return float(_config(seed=7).make().train_step()["loss"])

    assert run() == run()


def test_evaluation_does_not_change_the_network() -> None:
    step = _step()
    before = step.model.encoder.weight.detach().clone()
    result = step.eval_loss()
    assert math.isfinite(float(result["loss"]))
    assert torch.equal(before, step.model.encoder.weight.detach())


def test_action_values_can_be_read_for_arbitrary_observations() -> None:
    step = _step()
    values = step.call_eval(observation=torch.zeros(3, step.env.observation_size))
    assert values.shape == (3, step.env.num_actions)
    assert not values.requires_grad


def test_a_checkpoint_resumes_an_identical_run() -> None:
    step = _config(seed=3).make()
    step.train_step()
    saved = copy.deepcopy(step.state_dict())
    expected = float(step.train_step()["loss"])

    resumed = _config(seed=3).make()
    resumed.load_state_dict(saved)
    assert float(resumed.train_step()["loss"]) == expected


def test_a_checkpoint_restores_both_recurrent_tensors() -> None:
    step = _config(seed=3).make()
    step.train_step()
    saved = copy.deepcopy(step.state_dict())
    resumed = _config(seed=3).make()
    resumed.load_state_dict(saved)
    assert torch.equal(resumed._state[0], step._state[0])
    assert torch.equal(resumed._state[1], step._state[1])


def test_a_checkpoint_restores_the_normalization_statistics() -> None:
    # They are buffers, not parameters, so a state dict that missed them
    # would resume with a network normalizing by nothing.
    step = _config(seed=3).make()
    step.train_step()
    saved = copy.deepcopy(step.state_dict())
    resumed = _config(seed=3).make()
    resumed.load_state_dict(saved)
    assert torch.equal(
        resumed.model.normalize.running_mean,
        step.model.normalize.running_mean,
    )


def test_finished_episodes_are_summarized_once() -> None:
    step = _step()
    step.env.state.player_health[:] = 0.0
    metrics = _metrics(step.train_step())
    assert float(metrics["episodes"]) > 0.0
    assert float(_metrics(step.train_step())["episodes"]) < float(metrics["episodes"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rollout_steps", 0),
        ("num_epochs", 0),
        ("num_minibatches", 0),
        ("total_train_steps", 0),
        ("discount", 1.5),
        ("trace_decay", -0.1),
        ("epsilon_decay_fraction", 0.0),
    ],
)
def test_an_invalid_setting_is_refused(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=r"positive|at least one|between zero|in \(0"):
        _config(**{field: value}).make()


def test_minibatches_that_do_not_divide_the_workers_are_refused() -> None:
    with pytest.raises(ValueError, match="divide"):
        _config(num_minibatches=3).make()


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
