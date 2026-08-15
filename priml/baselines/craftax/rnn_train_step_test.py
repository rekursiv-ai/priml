"""Tests for the recurrent PPO training step."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from priml.baselines.craftax.rnn_train_step import CraftaxRNNTrainStep
from priml.train.custom_types import TrainStepOutput, TrainStepProtocol


def _config(**overrides: object) -> CraftaxRNNTrainStep.Config:
    config = CraftaxRNNTrainStep.Config()
    config.device = "cpu"
    config.env.device = "cpu"
    config.env.num_envs = 4
    config.env.optimistic_reset_ratio = 1
    # A 3x3 view, not the benchmark's 9x11: these test the UPDATE, and a
    # 8,268-wide observation makes the first layer dominate every one.
    config.env.view = (3, 3)
    config.rollout_steps = 4
    config.num_epochs = 2
    config.num_minibatches = 2
    config.total_train_steps = 10
    config.model.hidden_size = 16
    for name, value in overrides.items():
        setattr(config, name, value)
    if "seed" in overrides:
        config.env.seed = int(config.seed)
    return config


def _step() -> CraftaxRNNTrainStep:
    return _config().make()


def _metrics(result: TrainStepOutput) -> dict[str, float | torch.Tensor]:
    """Read the optional diagnostics a completed update always carries."""
    return result.get("metrics", {})


def test_it_satisfies_the_training_step_protocol() -> None:
    assert isinstance(_step(), TrainStepProtocol)


def test_the_network_is_sized_from_the_environment() -> None:
    step = _step()
    assert step.model.embed[0].in_features == step.env.observation_size
    assert step.model.actor[-1].out_features == step.env.num_actions


def test_one_step_consumes_the_declared_interactions() -> None:
    assert _step().steps_per_update == 4 * 4


def test_a_step_optimizes_and_reports_its_diagnostics() -> None:
    result = _step().train_step()
    assert math.isfinite(float(result["loss"]))
    for name in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
        "learning_rate",
        "explained_variance",
        "episodes",
    ):
        assert math.isfinite(float(_metrics(result)[name])), name


def test_a_step_changes_the_policy() -> None:
    step = _step()
    before = step.model.embed[0].weight.detach().clone()
    step.train_step()
    assert not torch.equal(before, step.model.embed[0].weight.detach())


def test_a_rollout_records_the_state_it_began_from() -> None:
    # That one vector is the whole replay record: unlike the transformer,
    # there is no per-layer cache to rebuild.
    step = _step()
    rollout = step.collect()
    assert rollout.initial_state.shape == (4, 16)
    assert rollout.observation.shape == (4, 4, step.env.observation_size)
    assert rollout.previous_done.shape == (4, 4)


def test_memory_carries_across_updates() -> None:
    step = _step()
    step.train_step()
    assert float(step._state.abs().max()) > 0.0


def test_minibatches_split_workers_and_never_time() -> None:
    """Whole trajectories move together; cutting time would sever history."""
    step = _step()
    rollout = step.collect()
    minibatches = list(rollout.minibatches(count=2))
    assert len(minibatches) == 2
    for minibatch in minibatches:
        # 4 steps x 2 workers, with the state each worker began from.
        assert minibatch["observation"].shape[:2] == (4, 2)
        assert minibatch["initial_state"].shape == (2, 16)


def test_a_replayed_trajectory_reproduces_the_rollout_exactly() -> None:
    """The recorded state replays the recurrence with no drift at all.

    Exact, not approximate: a GRU run over the same inputs from the same
    state performs identical operations in identical order. The transformer's
    equivalent test tolerates a difference because its training windows see
    more context than the rollout did; here there is nothing to differ.
    """
    step = _config(num_minibatches=1).make()
    rollout = step.collect()
    minibatch = next(rollout.minibatches(count=1))
    with torch.no_grad():
        _, logits, value = step.model.sequence(
            minibatch["initial_state"],
            minibatch["observation"],
            minibatch["previous_done"],
        )
    log_prob = logits.log_softmax(-1).gather(
        -1,
        minibatch["action"][..., None].long(),
    )[..., 0]
    assert torch.equal(value, minibatch["value"])
    assert torch.equal(log_prob, minibatch["log_prob"])


def test_the_learning_rate_anneals_toward_zero() -> None:
    step = _config(total_train_steps=4).make()
    rates: list[float] = []
    for _ in range(3):
        step.train_step()
        rates.append(float(step.optimizer.param_groups[0]["lr"]))
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] < rates[0]


def test_annealing_can_be_switched_off() -> None:
    step = _config(anneal_learning_rate=False, learning_rate=1e-3).make()
    step.train_step()
    assert step.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_the_same_seed_reproduces_the_same_update() -> None:
    def run() -> float:
        return float(_config(seed=7).make().train_step()["loss"])

    assert run() == run()


def test_evaluation_does_not_change_the_policy() -> None:
    step = _step()
    before = step.model.embed[0].weight.detach().clone()
    result = step.eval_loss()
    assert math.isfinite(float(result["loss"]))
    assert torch.equal(before, step.model.embed[0].weight.detach())


def test_action_logits_can_be_read_for_arbitrary_observations() -> None:
    step = _step()
    logits = step.call_eval(observation=torch.zeros(3, step.env.observation_size))
    assert logits.shape == (3, step.env.num_actions)
    assert not logits.requires_grad


def test_a_checkpoint_resumes_an_identical_run() -> None:
    step = _config(seed=3).make()
    step.train_step()
    saved = copy.deepcopy(step.state_dict())
    expected = float(step.train_step()["loss"])

    resumed = _config(seed=3).make()
    resumed.load_state_dict(saved)
    assert float(resumed.train_step()["loss"]) == expected


def test_a_checkpoint_restores_the_recurrent_state() -> None:
    # Without it a resumed run acts from an empty memory its weights do not
    # expect, and the first rollout after every resume is off-policy.
    step = _config(seed=3).make()
    step.train_step()
    saved = copy.deepcopy(step.state_dict())
    resumed = _config(seed=3).make()
    resumed.load_state_dict(saved)
    assert torch.equal(resumed._state, step._state)


def test_finished_episodes_are_summarized_once() -> None:
    step = _step()
    step.env.state.player_health[:] = 0.0
    metrics = _metrics(step.train_step())
    assert float(metrics["episodes"]) > 0.0
    assert math.isfinite(float(metrics["episode_return"]))
    assert float(_metrics(step.train_step())["episodes"]) < float(metrics["episodes"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rollout_steps", 0),
        ("num_epochs", 0),
        ("num_minibatches", 0),
        ("total_train_steps", 0),
        ("clip_epsilon", 0.0),
        ("discount", 1.5),
        ("trace_decay", -0.1),
    ],
)
def test_an_invalid_setting_is_refused(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=r"positive|at least one|between zero and one"):
        _config(**{field: value}).make()


def test_minibatches_that_do_not_divide_the_workers_are_refused() -> None:
    with pytest.raises(ValueError, match="divide"):
        _config(num_minibatches=3).make()


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
