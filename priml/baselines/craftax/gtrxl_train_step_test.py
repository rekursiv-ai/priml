"""Tests for the recurrent PPO training step."""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import copy
import math

import pytest
import torch

from priml.baselines.craftax.gtrxl_train_step import CraftaxGTrXLTrainStep
from priml.train.custom_types import TrainStepOutput, TrainStepProtocol
from priml.train.parallelism import NoParallel


pytestmark = pytest.mark.compute_training


def _config(**overrides: object) -> CraftaxGTrXLTrainStep.Config:
    config = CraftaxGTrXLTrainStep.Config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.env.device = "cpu"
    config.env.num_envs = 2
    config.rollout_steps = 4
    config.gradient_window = 2
    config.num_epochs = 1
    config.num_minibatches = 1
    config.total_train_steps = 10
    config.model.embed_dim = 4
    config.model.num_heads = 1
    config.model.num_layers = 1
    config.model.qkv_dim = 4
    config.model.hidden_size = 4
    config.model.memory_length = 2
    for name, value in overrides.items():
        setattr(config, name, value)
    # The world and the policy draw from separate streams, so a reproducible
    # run has to pin both.
    if "seed" in overrides:
        config.env.seed = int(config.seed)
    return config


def _step() -> CraftaxGTrXLTrainStep:
    return cast("CraftaxGTrXLTrainStep", _config().make())  # pyright: ignore[reportUnnecessaryCast] -- ty reads `Makes[...].make()` as @Todo and needs the cast


def test_it_satisfies_the_training_step_protocol() -> None:
    assert isinstance(_step(), TrainStepProtocol)


def test_the_network_is_sized_from_the_environment() -> None:
    step = _step()
    assert step.model.encoder.in_features == step.env.observation_size
    assert step.model.actor[-1].out_features == step.env.num_actions


def test_one_step_consumes_the_declared_interactions() -> None:
    assert _step().steps_per_update == 2 * 4


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


def test_each_epoch_visits_every_minibatch() -> None:
    step = _config(num_epochs=2, num_minibatches=2).make()
    with patch.object(step.optimizer, "step", wraps=step.optimizer.step) as optimizer:
        step.train_step()
    assert optimizer.call_count == 4


def test_a_step_changes_the_policy() -> None:
    step = _step()
    before = step.model.encoder.weight.detach().clone()
    step.train_step()
    assert not torch.equal(before, step.model.encoder.weight.detach())


def test_a_rollout_carries_the_memory_needed_to_replay_it() -> None:
    step = _step()
    rollout = step.collect()
    assert rollout.observation.shape == (4, 2, step.env.observation_size)
    # One row per layer per step: exactly what each layer read, which is what
    # lets a gradient window rebuild the cache it began with.
    assert rollout.layer_input.shape == (4, 2, 1, 4)
    assert rollout.valid_length.shape == (4, 2)


def test_memory_carries_across_updates() -> None:
    # A policy that forgot at every update boundary would never learn to use
    # anything longer than one rollout.
    step = _step()
    step.train_step()
    assert int(step._valid_length.min()) > 0


def test_a_terminal_transition_is_recorded_before_it_clears_the_memory() -> None:
    step = _step()
    step.env.state.player_health[:] = 0.0
    rollout = step.collect()
    # Step 0 ends every episode, so step 1 opens fresh with nothing to look
    # back at.
    assert rollout.valid_length[1].tolist() == [0, 0]


def test_minibatches_split_workers_and_never_time() -> None:
    """Whole trajectories move together; splitting time would sever history.

    Each minibatch holds ``envs / count`` whole trajectories, cut into
    windows that are folded into the batch axis -- so a window's rows are
    still contiguous in time.
    """
    step = _step()
    memory = step._memory.clone()
    rollout = step.collect()
    minibatches = list(
        rollout.minibatches(
            initial_memory=memory,
            count=2,
            window=2,
            memory_length=2,
        ),
    )
    assert len(minibatches) == 2
    for minibatch in minibatches:
        assert minibatch["observation"].shape == (
            2,
            2,
            step.env.observation_size,
        )
        assert minibatch["memory"].shape == (2, 2, 1, 4)
        assert minibatch["valid_length"].shape == (2,)


def test_a_replayed_window_reproduces_the_rollout_it_came_from() -> None:
    """The rebuilt memory is the one the rollout actually had.

    Checked on the FIRST window, where the two paths must agree exactly: a
    later window's queries attend their whole window on top of the cache,
    which is a longer context than the rollout had (see the next test).
    """
    step = _config(num_minibatches=1).make()
    memory = step._memory.clone()
    rollout = step.collect()
    minibatch = next(
        rollout.minibatches(
            initial_memory=memory,
            count=1,
            window=2,
            memory_length=2,
        ),
    )
    with torch.no_grad():
        logits, value = step.model.sequence(
            minibatch["memory"],
            minibatch["valid_length"],
            minibatch["observation"],
            minibatch["previous_done"],
        )
    log_prob = logits.log_softmax(-1).gather(
        -1,
        minibatch["action"][..., None].long(),
    )[..., 0]
    assert torch.allclose(value[:, :2], minibatch["value"][:, :2], atol=1e-6)
    assert torch.allclose(log_prob[:, :2], minibatch["log_prob"][:, :2], atol=1e-6)


def test_a_training_window_sees_more_context_than_the_rollout_did() -> None:
    """Not a defect: it is what a Transformer-XL training window is.

    A later query in a window attends the replayed cache AND every earlier
    step of its own window, so its context can exceed ``memory_length``,
    which the fixed-size rollout cache never can. The reference behaves the
    same way, and matching it is why the two numbers differ.
    """
    step = _config(num_minibatches=1).make()
    memory = step._memory.clone()
    rollout = step.collect()
    minibatch = next(
        rollout.minibatches(
            initial_memory=memory,
            count=1,
            window=2,
            memory_length=2,
        ),
    )
    with torch.no_grad():
        _, value = step.model.sequence(
            minibatch["memory"],
            minibatch["valid_length"],
            minibatch["observation"],
            minibatch["previous_done"],
        )
    # The second window's last step is where the extra context accumulates.
    assert not torch.allclose(value[-1, 2:], minibatch["value"][-1, 2:], atol=1e-5)


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
    before = step.model.encoder.weight.detach().clone()
    result = step.eval_loss()
    assert math.isfinite(float(result["loss"]))
    assert torch.equal(before, step.model.encoder.weight.detach())


def test_action_logits_can_be_read_for_arbitrary_observations() -> None:
    step = _step()
    logits = step.call_eval(observation=torch.zeros(3, step.env.observation_size))
    assert logits.shape == (3, step.env.num_actions)
    assert not logits.requires_grad


def test_a_checkpoint_resumes_an_identical_run() -> None:
    step = _config(seed=3).make()
    step.train_step()
    # Deep-copied because the live step keeps mutating these tensors.
    saved = copy.deepcopy(step.state_dict())
    expected = float(step.train_step()["loss"])

    resumed = _config(seed=3).make()
    resumed.load_state_dict(saved)
    assert float(resumed.train_step()["loss"]) == expected


def test_a_checkpoint_restores_the_memory() -> None:
    # Without it a resumed run would act with an empty cache for the next
    # ``memory_length`` steps while its weights expect a full one.
    step = _config(seed=3).make()
    step.train_step()
    saved = copy.deepcopy(step.state_dict())
    resumed = _config(seed=3).make()
    resumed.load_state_dict(saved)
    assert torch.equal(resumed._memory, step._memory)
    assert torch.equal(resumed._valid_length, step._valid_length)


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
        ("gradient_window", 0),
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


def test_a_window_that_does_not_divide_the_rollout_is_refused() -> None:
    # A ragged final window would silently receive gradients over a shorter
    # context than every other one.
    with pytest.raises(ValueError, match="divide"):
        _config(gradient_window=3).make()


def test_minibatches_that_do_not_divide_the_workers_are_refused() -> None:
    with pytest.raises(ValueError, match="divide"):
        _config(num_minibatches=3).make()


def _metrics(result: TrainStepOutput) -> dict[str, float | torch.Tensor]:
    """Read the optional diagnostics a completed update always carries."""
    return result.get("metrics", {})


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
