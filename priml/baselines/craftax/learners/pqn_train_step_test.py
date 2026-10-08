"""Tests for the Q-learning train step, at exp006's recipe and a tiny geometry.

The geometry: 12 environments in 2 buffers, rollouts of 2 steps, minibatches
of 3 trajectories, a network 5 wide; the recipe's 4 passes of 4 minibatches
and its schedules stay. Most tests train on ``testing.FakeEnv``'s random
transitions, which need no game; those marked ``compute_training`` step the
real game by exp006's rules. The golden runs inside ``host_agnostic_pipeline``,
so its bits hold on every CPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Final, cast, override
from unittest.mock import patch

import io
import math
import re

from configgle import Fig

import pytest
import torch

from priml.baselines.craftax.env import CraftaxEnv
from priml.baselines.craftax.experiments import exp006
from priml.baselines.craftax.game.state import OBS_SIZE, new_stats
from priml.baselines.craftax.learners.pqn_train_step import (
    CraftaxPQNTrainLoop,
    CraftaxPQNTrainStep,
)
from priml.baselines.craftax.policies.pqn import EpsilonGreedy
from priml.baselines.craftax.rollout import PhiloxSampler, TorchPhiloxSampler
from priml.baselines.craftax.testing import (
    FakeEnv,
    assert_golden,
    digest,
    fp32,
    host_agnostic_pipeline,
    tiny_env,
)
from priml.lib.codec import from_plain
from priml.math.advantage import q_lambda_targets
from priml.train.tracker import (
    AsyncTracker,
    FileTracker,
    TrackerList,
    WandbTracker,
)


if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from numpy.typing import NDArray
    from torch import Tensor

    import numpy as np

    from priml.baselines.craftax.learners.pqn_train_step import QRollout

ENVS: Final = 12
HORIZON: Final = 2
UPDATES: Final = 2
"""The tiny run's budget, which the schedules span."""

ACTIONS: Final = 43


class _Scripted:
    """``FakeEnv``'s random transitions, behind the surface the step trains on."""

    class Config(Fig["_Scripted"]):
        """The environments' count; ``FakeEnv`` writes the packed observation."""

        num_envs: int = ENVS
        """Environments, in 2 buffers."""

        @property
        def observation_size(self) -> int:
            """Floats per observation: the packed view's."""
            return OBS_SIZE

    def __init__(self, config: Config) -> None:
        inner = self._inner = FakeEnv.make(
            num_envs=config.num_envs,
            num_buffers=2,
            seed=0,
        )
        self.observations = inner.observations
        self.action_mask = inner.action_mask
        self.rewards = inner.rewards
        self.terminals = inner.terminals
        self.actions = inner.actions
        self.num_envs = inner.num_envs
        self.num_buffers = inner.num_buffers
        self.stats: NDArray[np.void] = new_stats(config.num_envs)

    def buffer_slice(self, buffer: int) -> slice:
        return self._inner.buffer_slice(buffer)

    def step_buffer(self, buffer: int) -> None:
        self._inner.step_buffer(buffer)

    def reset(self) -> None:
        """Keep the first observations: the transitions are random from the start."""

    def prepare_rollout(self) -> None:
        """Do nothing: there is no practice to restore."""

    def state_dict(self) -> dict[str, Tensor]:
        return {
            **self._buffers(),
            **{
                f"generator{buffer}": generator.get_state()
                for buffer, generator in enumerate(self._inner.generators)
            },
        }

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        for name, live in self._buffers().items():
            live.copy_(state[name])
        for buffer, generator in enumerate(self._inner.generators):
            generator.set_state(state[f"generator{buffer}"])

    def close(self) -> None:
        """Do nothing: no thread steps the transitions."""

    def _buffers(self) -> dict[str, Tensor]:
        return {
            "observations": self.observations,
            "action_mask": self.action_mask,
            "rewards": self.rewards,
            "terminals": self.terminals,
            "actions": self.actions,
        }


class _Recording(CraftaxPQNTrainStep):
    """Keep every update's rollout and a copy of its targets, as a probe would read them."""

    def __init__(self, config: CraftaxPQNTrainStep.Config) -> None:
        super().__init__(config)
        self.rollouts: list[QRollout] = []
        self.targets: list[Tensor] = []

    @override
    def collect(self) -> QRollout:
        rollout = super().collect()
        self.rollouts.append(rollout)
        self.targets.append(rollout.targets.clone())
        return rollout


def _config(*, game: bool = False) -> CraftaxPQNTrainStep.Config:
    """Return exp006's step on the CPU at the tiny geometry; on the real game if ``game``."""
    config = exp006().step
    if game:
        env = tiny_env()
        parent = config.env
        assert isinstance(parent, CraftaxEnv.Config)
        env.rules = parent.rules
        env.restart = parent.restart
        env.num_envs = ENVS
        config.env = env
    else:
        config.env = _Scripted.Config()
    config.model.channels_hidden = 5
    config.rollout.horizon = HORIZON
    sampler = config.sampler
    assert isinstance(sampler, EpsilonGreedy.Config)
    sampler.sampler = TorchPhiloxSampler.Config().update(
        sampler.sampler,
        skip_missing=True,
    )
    config.parallelism.device = "cpu"
    config.evaluation.num_episodes = 1
    config.train_budget_steps = UPDATES
    return config


@contextmanager
def _step(
    config: CraftaxPQNTrainStep.Config | None = None,
    *,
    seed: int = 0,
) -> Generator[_Recording]:
    """Build the recording step from torch seed ``seed``, and close it after."""
    torch.manual_seed(seed)
    step = _Recording((config or _config()).copy_tree().finalize())
    try:
        yield step
    finally:
        step.close()


def test_the_network_is_sized_from_the_environment() -> None:
    with _step() as step:
        assert step.model.encoder.in_features == OBS_SIZE
        assert step.model.head.out_features == ACTIONS


def test_a_step_optimizes_and_reports_its_diagnostics() -> None:
    with _step() as step:
        result = step.train_step()
        assert math.isfinite(float(result["loss"]))
        assert result["model"].shape == (ENVS // 4, HORIZON, ACTIONS)
        metrics = result.get("metrics", {})
        for name in (
            "q_loss",
            "q_mean",
            "grad_norm",
            "learning_rate",
            "epsilon",
            "explained_variance",
            "transitions_per_second",
            "rollout_seconds",
            "learner_seconds",
        ):
            assert math.isfinite(float(metrics[name])), name
        assert metrics["agent_steps"] == ENVS * HORIZON
        assert step.global_step == 1
        rollout, learner = metrics["rollout_seconds"], metrics["learner_seconds"]
        assert float(rollout) > 0.0
        assert float(learner) > 0.0
        assert float(metrics["transitions_per_second"]) == pytest.approx(
            ENVS * HORIZON / (float(rollout) + float(learner)),
        )


def test_each_update_sets_the_optimizers_rate_from_the_linear_schedule() -> None:
    with _step() as step:
        for update in range(UPDATES):
            metrics = step.train_step().get("metrics", {})
            rate = 3e-4 * (1 - update / UPDATES)
            assert float(metrics["learning_rate"]) == pytest.approx(rate)
            assert step.optimizer.param_groups[0]["lr"] == pytest.approx(rate)


def test_each_pass_visits_every_minibatch() -> None:
    with _step() as step:
        optimizer = step.optimizer
        with patch.object(optimizer, "step", wraps=optimizer.step) as stepped:
            step.train_step()
        assert stepped.call_count == 4 * 4


def test_a_step_changes_the_network() -> None:
    with _step() as step:
        before = step.model.encoder.weight.detach().clone()
        step.train_step()
        assert not torch.equal(before, step.model.encoder.weight.detach())


def test_it_trains_without_a_target_network() -> None:
    """The whole claim: one network, regressed against its own values.

    A second network would show up as a second copy of the weights, so this is
    checkable rather than merely stated.
    """
    with _step() as step:
        modules = {name for name, _ in step.model.named_modules() if name}
        assert not any("target" in name for name in modules)
        params = cast("list[Tensor]", step.optimizer.param_groups[0]["params"])
        assert [id(p) for p in params] == [id(p) for p in step.model.parameters()]


def test_it_keeps_no_replay_buffer() -> None:
    # Every rollout is rewritten by the next; a checkpoint keeps none of it.
    with _step() as step:
        step.train_step()
        step.train_step()
        saved = step.state_dict()
        assert sorted(saved["rollout"]) == ["carry", "draws"]
        assert not any("buffer" in key or "replay" in key for key in saved)


def test_the_targets_are_built_once_before_optimizing() -> None:
    """Recomputing them per pass would chase a value already moved.

    The rollout carries its targets, so all 16 minibatches regress toward the
    numbers ``collect`` built.
    """
    with _step() as step:
        step.train_step()
        (rollout,) = step.rollouts
        assert torch.equal(step.targets[0], rollout.targets)


def test_a_subclass_reads_each_update_before_it_learns() -> None:
    """``collect`` is the probe's hook: the values, rewards and terminals per update.

    The values are the collecting weights': the next update's rollout, collected
    after the learner moved them, values its observations by the moved weights.
    """
    with _step() as step:
        step.train_step()
        step.train_step()
        first, second = step.rollouts
        taken = first.q_values.gather(-1, first.actions[..., None])[..., 0]
        assert taken.shape == first.rewards.shape == first.dones.shape
        assert (first.bootstrap.shape, first.targets.shape) == (
            (ENVS, ACTIONS),
            (HORIZON, ENVS),
        )
        assert not torch.equal(first.q_values, second.q_values)


def test_each_transition_reads_the_reward_and_terminal_of_the_next_row() -> None:
    """Row ``t`` stores what arrived with observation ``t``; the env holds the last."""
    with _step() as step:
        rollout = step.collect()
        storage = step.rollout.slots[0]
        assert torch.equal(rollout.rewards[:-1], storage.rewards[1:])
        assert torch.equal(rollout.rewards[-1], step.env.rewards)
        assert torch.equal(rollout.dones[:-1], storage.terminals[1:] != 0)
        assert torch.equal(rollout.dones[-1], step.env.terminals != 0)
        assert bool(rollout.dones.any())


def test_the_targets_are_the_reference_recursion_of_the_rollouts_own_values() -> None:
    with _step() as step:
        rollout = step.collect()
        assert bool(rollout.dones.any())
        expected = _reference_targets(
            rewards=rollout.rewards,
            q_values=rollout.q_values,
            bootstrap=rollout.bootstrap,
            dones=rollout.dones.float(),
            gamma=0.99,
            trace_decay=0.5,
        )
        assert torch.equal(rollout.targets, expected)


@pytest.mark.parametrize("steps", [1, 2, 6])
def test_q_lambda_targets_are_the_reference_recursion(steps: int) -> None:
    """The value-based target, given identical rewards and Q-values.

    Parameterized over length because the reference special-cases its last
    step and scans the rest, so one length exercises only one of its paths.
    """
    generator = torch.Generator().manual_seed(steps)
    rewards = torch.randn(steps, 3, generator=generator)
    dones = (torch.rand(steps, 3, generator=generator) < 0.25).float()
    q_values = torch.randn(steps + 1, 3, 4, generator=generator)
    targets = q_lambda_targets(
        rewards=rewards,
        q_values=q_values,
        dones=dones,
        discount=0.99,
        trace_decay=0.5,
    )
    assert torch.equal(
        targets,
        _reference_targets(
            rewards=rewards,
            q_values=q_values[:-1],
            bootstrap=q_values[-1],
            dones=dones,
            gamma=0.99,
            trace_decay=0.5,
        ),
    )


def test_the_rescored_values_are_the_ones_the_actor_computed() -> None:
    """The rollout's stored values are the actor's greedy values, step by step.

    Exactly equal, with no tolerance: the actor scored each buffer's 6 rows
    and the rescore all 12, and inside ``host_agnostic_pipeline`` every float
    op runs in float64 and rounds once to fp32, so a product's row count moves
    no bit. On the GPU the kernels' row counts may round differently; there
    the CUDA test allows their tolerance.
    """
    with host_agnostic_pipeline(), _step() as step:
        rollout = step.collect()
        assert bool(rollout.dones.any())
        assert torch.equal(rollout.q_values.amax(dim=-1), step.rollout.slots[0].values)


def test_the_bootstrap_is_the_value_the_actor_reads_next() -> None:
    """Q(obs_H), from the rescore's final carry and the env's buffers, is the next step's.

    The next rollout's actor steps that observation from the carry it left,
    after the action that led there.
    """
    with host_agnostic_pipeline(), _step() as step:
        bootstrap = step.collect().bootstrap.clone()
        following = step.collect()
        assert torch.equal(bootstrap.amax(dim=-1), step.rollout.slots[0].values[0])
        assert torch.equal(bootstrap, following.q_values[0])


def test_exploration_decays_across_the_run() -> None:
    with _step() as step:
        rates = [
            float(step.train_step().get("metrics", {})["epsilon"])
            for _ in range(UPDATES)
        ]
        assert rates == [1.0, pytest.approx(0.005)]


def test_collection_does_not_update_the_running_statistics() -> None:
    """A rollout is inference.

    Folding it into the normalization would count every observation twice per
    update -- once acting, once learning.
    """
    with _step() as step:
        step.collect()
        assert int(step.model.normalize.steps) == 0
        step.train_step()
        # The learner's 16 minibatches each fold in their window, once.
        assert int(step.model.normalize.steps) == 4 * 4
        assert not step.model.training


def test_the_loss_is_half_the_mean_squared_error_of_the_taken_actions() -> None:
    """The loss is purejaxql's, ``0.5 * mean((Q - y) ** 2)``, over every transition.

    Scored after a collect, so the targets are the ones it built: the same
    weights, the same rollout and the env's same buffers.
    """
    with _step() as step:
        rollout = step.collect()
        result = step.train_loss()
        chosen = result["model"].gather(-1, rollout.actions.T[..., None])[..., 0]
        expected = 0.5 * ((chosen - rollout.targets.T) ** 2).mean()
        assert torch.equal(result["loss"], expected)


def test_the_same_seed_reproduces_the_same_update() -> None:
    def run() -> float:
        with _step() as step:
            return float(step.train_step()["loss"])

    assert run() == run()


def test_scoring_changes_nothing() -> None:
    with _step() as step:
        step.train_step()
        before = {
            name: value.clone() for name, value in step.model.state_dict().items()
        }
        result = step.eval_loss()
        assert math.isfinite(float(result["loss"]))
        assert result["model"].shape == (ENVS, HORIZON, ACTIONS)
        for name, value in step.model.state_dict().items():
            assert torch.equal(before[name], value), name


def test_action_values_can_be_read_for_arbitrary_observations() -> None:
    with _step() as step:
        observation = torch.zeros(3, OBS_SIZE)
        values = step.call_eval(observation=observation)
        assert values.shape == (3, ACTIONS)
        assert not values.requires_grad
        assert torch.equal(step.call_eval(observation), values)


def test_a_checkpoint_resumes_where_the_run_left_off() -> None:
    """A step from another init, loaded, collects the next rollout bit for bit.

    That rollout reads all a checkpoint restores but the optimizer, compared
    after: the weights and their running normalization, which are buffers, not
    parameters; the environments, the carries, the streams, and the update
    count, which sets the exploration rate.
    """
    with _step() as step:
        step.train_step()
        buffer = io.BytesIO()
        torch.save(step.state_dict(), buffer)
        expected = step.collect()
        moments = _moments(step)

    with _step(seed=1) as resumed:
        buffer.seek(0)
        resumed.load_state_dict(
            from_plain(
                cast("object", torch.load(buffer, weights_only=True)),
                dict[str, object],
            ),
        )
        assert resumed.global_step == 1
        rollout = resumed.collect()
        for ours, theirs in (
            (rollout.actions, expected.actions),
            (rollout.q_values, expected.q_values),
            (rollout.bootstrap, expected.bootstrap),
            (rollout.rewards, expected.rewards),
            (rollout.dones, expected.dones),
            (rollout.targets, expected.targets),
        ):
            assert torch.equal(ours, theirs)
        assert _moments(resumed).keys() == moments.keys()
        for key, value in _moments(resumed).items():
            assert torch.equal(value, moments[key]), key
        # The episode logs restored, the next update reports from them.
        assert math.isfinite(float(resumed.train_step()["loss"]))


def test_the_loops_tick_passes_through() -> None:
    """The data comes from the environments; the loop's batch is only its tick."""
    with _step() as step:
        batch: dict[str, object] = {"tick": 3}
        assert step.preprocess_batch(batch) is batch


def test_an_update_reports_the_episodes_finished_since_the_last() -> None:
    """Each update differences the env's running log sums against the last it read."""
    with _step() as step:
        logs = step.env.stats["log"]
        logs["n"][:2] = 2.0
        logs["perf"][:2] = (0.5, 1.5)
        first = step.train_step().get("metrics", {})
        second = step.train_step().get("metrics", {})
    assert (first["env/n"], first["env/perf"]) == (4.0, 0.5)
    assert not any(name.startswith("env/") for name in second)


def test_a_single_finished_episode_is_reported() -> None:
    with _step() as step:
        step.env.stats["log"]["n"][:1] = 1.0
        metrics = step.train_step().get("metrics", {})
    assert metrics["env/n"] == 1.0


def test_a_spent_run_refuses_another_update() -> None:
    with _step() as step:
        for _ in range(UPDATES):
            step.train_step()
        with pytest.raises(RuntimeError, match="spent"):
            step.train_step()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            "rollout.num_slots",
            2,
            (
                "each rollout is learned from with the weights that collected it: "
                "num_slots must be 1, not 2"
            ),
        ),
        ("num_epochs", 0, "num_epochs and num_minibatches must be positive"),
        ("num_minibatches", 0, "num_epochs and num_minibatches must be positive"),
        ("num_minibatches", 5, "num_minibatches must divide the environments"),
        ("discount", 1.5, "discount must be in [0, 1], not 1.5"),
        ("trace_decay", -0.1, "trace_decay must be in [0, 1], not -0.1"),
        ("trace_decay", math.nan, "trace_decay must be in [0, 1], not nan"),
        ("max_grad_norm", 0.0, "max_grad_norm must be positive and finite, not 0.0"),
        (
            "max_grad_norm",
            math.inf,
            "max_grad_norm must be positive and finite, not inf",
        ),
        (
            "train_budget_steps",
            math.inf,
            "train_budget_steps is the update count; it must be finite and positive",
        ),
        (
            "train_budget_steps",
            0,
            "train_budget_steps is the update count; it must be finite and positive",
        ),
    ],
)
def test_an_invalid_setting_is_refused(path: str, value: float, message: str) -> None:
    config = _config()
    _set(config, path, value)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        config.make()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("num_epochs", 1),
        ("num_minibatches", 1),
        ("discount", 0.0),
        ("discount", 1.0),
        ("trace_decay", 0.0),
        ("trace_decay", 1.0),
        ("train_budget_steps", 1),
    ],
)
def test_a_setting_at_its_bound_is_accepted(path: str, value: float) -> None:
    config = _config()
    _set(config, path, value)
    with _step(config) as step:
        assert step.global_step == 0


def test_the_loop_names_its_wandb_run_after_the_experiment() -> None:
    """Unnamed, W&B makes a name up, and a board of seeds cannot be told apart."""
    config = CraftaxPQNTrainLoop()
    config.experiment_name = "exp006_s43"
    dashboard = WandbTracker.Config()
    wrapper = AsyncTracker.Config()
    wrapper.tracker = dashboard
    trackers = config.tracker = TrackerList.Config()
    trackers.trackers = {"metrics": FileTracker.Config(), "wandb": wrapper}
    finalized = config.copy_tree().finalize()
    assert isinstance(finalized.tracker, TrackerList.Config)
    named = finalized.tracker.trackers["wandb"]
    assert isinstance(named, AsyncTracker.Config)
    assert isinstance(named.tracker, WandbTracker.Config)
    assert named.tracker.name == "exp006_s43"
    dashboard.name = "kept"
    kept = config.copy_tree().finalize().tracker
    assert isinstance(kept, TrackerList.Config)
    wrapped = kept.trackers["wandb"]
    assert isinstance(wrapped, AsyncTracker.Config)
    assert isinstance(wrapped.tracker, WandbTracker.Config)
    assert wrapped.tracker.name == "kept"
    config.tracker = FileTracker.Config()
    assert isinstance(config.copy_tree().finalize().tracker, FileTracker.Config)


@pytest.mark.compute_training
def test_finished_episodes_are_reported_once() -> None:
    with _step(_config(game=True)) as step:
        env = step.env
        assert isinstance(env, CraftaxEnv)
        env.states["player_health"][:] = 0.0
        metrics = step.train_step().get("metrics", {})
        assert float(metrics["env/n"]) == ENVS
        later = step.train_step().get("metrics", {})
        assert float(later.get("env/n", 0.0)) < ENVS


@pytest.mark.compute_training
def test_an_evaluation_plays_greedily_and_changes_nothing_training_reads() -> None:
    def train(*, evaluate: bool) -> dict[str, Tensor]:
        with _step(_config(game=True)) as step:
            step.train_step()
            if evaluate:
                evaluation = step.make_evaluator()
                try:
                    played = evaluation.play()
                finally:
                    evaluation.close()
                assert played.rollouts > 0
            step.train_step()
            return {
                name: value.clone() for name, value in step.model.state_dict().items()
            }

    plain, evaluated = train(evaluate=False), train(evaluate=True)
    assert plain.keys() == evaluated.keys()
    for name, value in plain.items():
        assert torch.equal(value, evaluated[name]), name


@pytest.mark.compute_training
def test_the_tiny_steps_updates_match_their_golden() -> None:
    """exp006's step at test size from torch seed 0, every update frozen on every host.

    The whole update on the real game by exp006's rules: the epsilon-greedy
    rollout, the rescore and its targets, the 16 minibatches, RAdam behind the
    clip, the linear rate, the running normalization. Each update's targets,
    rate, exploration and loss, then the final weights and statistics.
    """
    lines: list[str] = []
    with host_agnostic_pipeline(), _step(_config(game=True)) as step:
        for update in range(1, UPDATES + 1):
            result = step.train_step()
            metrics = result.get("metrics", {})
            lines += [
                f"update {update:04d} targets {digest(step.targets[-1])}",
                f"update {update:04d} learning_rate {fp32(metrics['learning_rate'])}",
                f"update {update:04d} epsilon {fp32(metrics['epsilon'])}",
                f"update {update:04d} loss {fp32(result['loss'])}",
            ]
        lines += [
            f"final {name} {digest(value)}"
            for name, value in step.model.state_dict().items()
        ]
    assert_golden(test_file=__file__, name="train_step_exp006_tiny", lines=lines)


@pytest.mark.gpu_triton
def test_exp006s_step_trains_and_evaluates_on_the_gpu() -> None:
    """exp006's step at the tiny geometry on CUDA: Philox's kernel, captured step graphs.

    The rescore runs the 12 rows at once where each captured step ran its
    buffer's 6, and the GPU's products round by their row counts, so the two
    agree within fp32's default tolerance rather than bit for bit.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    config = _config(game=True)
    config.parallelism.device = "cuda"
    sampler = config.sampler
    assert isinstance(sampler, EpsilonGreedy.Config)
    sampler.sampler = PhiloxSampler.Config().update(
        sampler.sampler,
        skip_missing=True,
    )
    with _step(config) as step:
        result = step.train_step()
        assert math.isfinite(float(result["loss"]))
        rollout = step.collect()
        torch.testing.assert_close(
            rollout.q_values.amax(dim=-1),
            step.rollout.slots[0].values,
        )
        values = step.call_eval(torch.zeros(3, step.env.observations.shape[1]))
        assert (values.device.type, values.shape) == ("cuda", (3, ACTIONS))
        evaluation = step.make_evaluator()
        try:
            assert evaluation.play().rollouts > 0
        finally:
            evaluation.close()


def _moments(step: CraftaxPQNTrainStep) -> dict[str, Tensor]:
    """Return the optimizer's per-parameter state, keyed by parameter index and name."""
    state = cast(
        "dict[int, dict[str, Tensor]]",
        step.optimizer.state_dict()["state"],
    )
    return {
        f"{index}.{name}": value
        for index, entry in state.items()
        for name, value in entry.items()
    }


def _set(config: CraftaxPQNTrainStep.Config, path: str, value: float) -> None:
    """Set the dotted ``path`` of ``config`` to ``value``."""
    *parents, name = path.split(".")
    target: object = config
    for parent in parents:
        target = cast("object", getattr(target, parent))
    setattr(target, name, value)


# ``pqn_rnn_craftax.py``'s ``_compute_targets``, transition by transition from the last:
# ``reward + gamma (1 - done) next_q`` mixed with the carried return by ``gamma
# lambda``, then the reward alone where ``done``.
def _reference_targets(
    *,
    rewards: Tensor,
    q_values: Tensor,
    bootstrap: Tensor,
    dones: Tensor,
    gamma: float,
    trace_decay: float,
) -> Tensor:
    """Return purejaxql's Q(lambda) targets, its ``_get_target`` recursion written out."""
    lambda_returns = rewards[-1] + gamma * (1 - dones[-1]) * bootstrap.amax(dim=-1)
    targets = [lambda_returns]
    next_q = q_values[-1].amax(dim=-1)
    for step in range(rewards.shape[0] - 2, -1, -1):
        target_bootstrap = rewards[step] + gamma * (1 - dones[step]) * next_q
        delta = lambda_returns - next_q
        lambda_returns = target_bootstrap + gamma * trace_decay * delta
        lambda_returns = (1 - dones[step]) * lambda_returns + dones[step] * rewards[
            step
        ]
        next_q = q_values[step].amax(dim=-1)
        targets.append(lambda_returns)
    return torch.stack(targets[::-1])


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
