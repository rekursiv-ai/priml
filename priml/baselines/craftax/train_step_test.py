"""Tests for the PPO training step."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from priml.baselines.craftax.train_step import CraftaxTrainStep
from priml.testing.fixtures import torch_compiler_isolation
from priml.train.custom_types import TrainStepOutput, TrainStepProtocol


def _config(**overrides: object) -> CraftaxTrainStep.Config:
    config = CraftaxTrainStep.Config()
    config.device = "cpu"
    config.env.device = "cpu"
    config.env.num_envs = 4
    config.rollout_steps = 4
    config.num_epochs = 2
    config.num_minibatches = 2
    config.total_train_steps = 10
    config.model.hidden_size = 32
    config.model.num_layers = 1
    for name, value in overrides.items():
        setattr(config, name, value)
    # The world and the policy draw from separate streams, so a reproducible
    # run has to pin both.
    if "seed" in overrides:
        config.env.seed = int(config.seed)
    return config


def _step() -> CraftaxTrainStep:
    return _config().make()


def test_it_satisfies_the_training_step_protocol() -> None:
    assert isinstance(_step(), TrainStepProtocol)


def test_the_network_is_sized_from_the_environment() -> None:
    # An experiment that changes the environment must not have to remember to
    # resize the network by hand.
    step = _step()
    assert step.model.policy[0].in_features == step.env.observation_size
    assert step.model.policy[-1].out_features == step.env.num_actions


def test_one_step_consumes_the_declared_interactions() -> None:
    step = _step()
    assert step.steps_per_update == 4 * 4


def test_a_step_optimizes_and_reports_its_diagnostics() -> None:
    result = _step().train_step()
    assert math.isfinite(float(result["loss"]))
    metrics = _metrics(result)
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
        assert math.isfinite(float(metrics[name])), name


def test_a_step_changes_the_policy() -> None:
    step = _step()
    before = step.model.policy[0].weight.detach().clone()
    step.train_step()
    assert not torch.equal(before, step.model.policy[0].weight.detach())


def test_the_step_counter_advances() -> None:
    step = _step()
    step.train_step()
    step.train_step()
    assert step.global_step == 2


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


def test_a_rollout_has_the_declared_shape() -> None:
    step = _step()
    rollout = step.collect()
    assert rollout.observation.shape == (4, 4, step.env.observation_size)
    assert rollout.action.shape == (4, 4)
    assert rollout.advantage.shape == (4, 4)
    assert rollout.target.shape == (4, 4)


def test_a_rollout_is_collected_without_gradients() -> None:
    # Backpropagating through the collection would tie the policy to its own
    # sampling, which the clipped objective already accounts for.
    rollout = _step().collect()
    assert not rollout.observation.requires_grad
    assert not rollout.log_prob.requires_grad


def test_minibatches_partition_the_rollout_exactly() -> None:
    step = _step()
    rollout = step.collect()
    seen = [minibatch["action"].shape[0] for minibatch in rollout.minibatches(count=4)]
    assert sum(seen) == 4 * 4
    assert len(seen) == 4


def test_minibatches_are_shuffled() -> None:
    rollout = _step().collect()
    first = next(
        rollout.minibatches(count=1, generator=torch.Generator().manual_seed(0))
    )
    second = next(
        rollout.minibatches(count=1, generator=torch.Generator().manual_seed(1)),
    )
    assert not torch.equal(first["action"], second["action"])


def test_the_same_seed_reproduces_the_same_update() -> None:
    def run() -> float:
        step = _config(seed=7).make()
        return float(step.train_step()["loss"])

    assert run() == run()


def test_evaluation_does_not_change_the_policy() -> None:
    step = _step()
    before = step.model.policy[0].weight.detach().clone()
    result = step.eval_loss()
    assert math.isfinite(float(result["loss"]))
    assert torch.equal(before, step.model.policy[0].weight.detach())


def test_action_logits_can_be_read_for_arbitrary_observations() -> None:
    step = _step()
    logits = step.call_eval(observation=torch.zeros(3, step.env.observation_size))
    assert logits.shape == (3, step.env.num_actions)
    assert not logits.requires_grad


def test_a_checkpoint_resumes_an_identical_run() -> None:
    step = _config(seed=3).make()
    step.train_step()
    # Deep-copied because the live step keeps mutating these tensors: a
    # shallow snapshot would be rewritten by the very update it is meant to
    # be compared against.
    saved = copy.deepcopy(step.state_dict())

    expected = float(step.train_step()["loss"])

    resumed = _config(seed=3).make()
    resumed.load_state_dict(saved)
    assert float(resumed.train_step()["loss"]) == expected


def test_finished_episodes_are_summarized_once() -> None:
    step = _step()
    step.env.state.player_health[:] = 0.0
    metrics = _metrics(step.train_step())
    assert float(metrics["episodes"]) > 0.0
    assert math.isfinite(float(metrics["episode_return"]))
    # The bank is cleared, so the next update reports only its own episodes.
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


@pytest.mark.slow
def test_compiling_agrees_with_eager_to_float32_rounding() -> None:
    """Compiling changes the last bits, and nothing above them.

    Measured, not assumed: at this width the two paths differ by about 4e-9,
    because inductor fuses the first matmul into a different reduction order.
    So the compiled run is the SAME experiment -- but it is not bit-for-bit,
    which is why the golden pins ``compile=False`` rather than defaulting it.
    """

    def loss(*, compiled: bool) -> float:
        with torch_compiler_isolation():
            return float(_config(seed=5, compile=compiled).make().train_step()["loss"])

    eager = loss(compiled=False)
    assert loss(compiled=True) == pytest.approx(eager, abs=1e-6)


def _metrics(result: TrainStepOutput) -> dict[str, float | torch.Tensor]:
    """Read the optional diagnostics a completed update always carries."""
    return result.get("metrics", {})


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
