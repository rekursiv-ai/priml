"""Unit tests for the windowed PPO epoch over whole trajectories, at tiny sizes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final
from unittest.mock import patch

import copy
import math

from configgle import PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.craftax.env import WorldPool
from priml.baselines.craftax.game.state import ATN_DIM, OBS_SIZE
from priml.baselines.craftax.learners.gtrxl_train_step import TrajectoryWindows
from priml.baselines.craftax.lib.adam import ClippedAdam
from priml.baselines.craftax.policies.gtrxl import GTrXLPolicy
from priml.baselines.craftax.rollout import (
    PhiloxSampler,
    Rollout,
    RolloutStorage,
    TorchPhiloxSampler,
)
from priml.baselines.craftax.testing import (
    FakeEnv,
    host_agnostic_pipeline,
    tiny_train_step,
)
from priml.baselines.craftax.train_step import CraftaxTrainStep, LearnerRollout
from priml.loss.policy_gradient import TorchPPO
from priml.testing.bfb import assert_bfb_against_golden


if TYPE_CHECKING:
    from collections.abc import Callable


_CWD: Final = Path(__file__).resolve().parent

AGENTS: Final = 6
HORIZON: Final = 4
WINDOW: Final = 2
MEMORY: Final = 5
OBSERVATION_SIZE: Final = 9
NUM_ACTIONS: Final = 4


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("num_passes", 0),
        ("num_minibatches", 0),
        ("window", 0),
        ("discount", 1.5),
        ("trace_decay", -0.1),
        ("clip_epsilon", math.nan),
        ("value_coefficient", -1.0),
        ("entropy_coefficient", math.inf),
        ("seed", -1),
    ],
)
def test_an_invalid_setting_is_refused(name: str, value: float) -> None:
    config = TrajectoryWindows.Config()
    setattr(config, name, value)
    with pytest.raises(ValueError, match=name):
        config.make()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("bootstrap", "bootstrap"),
        ("slots", "one slot"),
        ("horizon", "divide the horizon"),
        ("envs", "split the 7 environments"),
    ],
)
def test_a_rollout_the_windows_cannot_learn_from_is_refused(
    change: str,
    message: str,
) -> None:
    config = _step_config()
    if change == "bootstrap":
        config.rollout.bootstrap = False
    elif change == "slots":
        config.rollout.num_slots = 2
    elif change == "horizon":
        config.rollout.horizon = 5
    else:
        config.env.num_envs = 7
    with pytest.raises(ValueError, match=message):
        _learner().prepare(config)


def test_the_advantage_is_gae_bootstrapped_from_the_row_after() -> None:
    """Transition ``t`` leaves observation ``t``: its reward and done are row ``t + 1``'s."""
    rollout = _rollout()
    advantages, returns = _learner().advantage(rollout)
    expected = torch.zeros(AGENTS, HORIZON)
    gae = torch.zeros(AGENTS)
    next_value = rollout.values[:, HORIZON]
    for t in reversed(range(HORIZON)):
        done = rollout.terminals[:, t + 1]
        delta = rollout.rewards[:, t + 1] + 0.999 * next_value * (1 - done)
        gae = delta - rollout.values[:, t] + 0.999 * 0.8 * (1 - done) * gae
        expected[:, t] = gae
        next_value = rollout.values[:, t]
    torch.testing.assert_close(advantages, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(
        returns,
        expected + rollout.values[:, :HORIZON],
        rtol=0,
        atol=1e-6,
    )


def test_each_window_starts_from_the_carry_the_actor_read() -> None:
    """The replay rebuilds, before any update, the memory the rollout's actor read.

    Exactly: under ``host_agnostic_pipeline`` every float op computes in
    float64 and rounds once to float32, so the actor's two-environment buffers
    and the replay's four-environment batch round alike although their GEMMs
    differ in shape. On a GPU the shapes select different TF32 kernels, which
    agree only to their rounding. Rollouts of 2 steps and windows of 1: each
    op costs 80 us in those numerics on x86.
    """
    policy = _policy(observation_size=OBS_SIZE, num_actions=ATN_DIM).make()
    env = FakeEnv.make(num_envs=4, num_buffers=2, seed=0)
    config = Rollout.Config()
    config.num_slots = 1
    config.horizon = 2
    config.bootstrap = True
    windows = TrajectoryWindows.Config()
    windows.window = 1
    with host_agnostic_pipeline():
        rollout = Rollout(
            config,
            policy=policy,
            sampler=TorchPhiloxSampler.Config().make(),
            env=env,
            device=torch.device("cpu"),
        )
        spy = _CarrySpy(policy.forward_fused)
        try:
            with patch.object(policy, "forward_fused", spy):
                storage = rollout.collect(0)
        finally:
            rollout.close()
        carries = windows.make().carries(policy, _learner_rollout(storage))
    read = [spy.seen[graph.state.data_ptr()] for graph in rollout.graphs[0]]
    assert [len(steps) for steps in read] == [config.horizon + 1] * 2
    assert torch.equal(carries[0], storage.initial_states)
    assert torch.equal(carries[1], torch.cat([steps[1] for steps in read], dim=1))
    assert carries[1][..., -1].any()


def test_the_windows_split_environments_and_never_time() -> None:
    """Each window is ``window`` consecutive steps of one environment, from its carry."""
    rollout, step, learner = _rollout(), _step(), _learner()
    carries = learner.carries(step.model, rollout)
    calls = _record_windows(step, learner, rollout)
    assert len(calls) == 2 * 2
    for agents, (observations, state, starts) in calls:
        assert observations.shape == (2 * len(agents), WINDOW, OBSERVATION_SIZE)
        for window in range(HORIZON // WINDOW):
            rows = slice(window * len(agents), (window + 1) * len(agents))
            times = slice(window * WINDOW, (window + 1) * WINDOW)
            assert torch.equal(observations[rows], rollout.observations[agents, times])
            assert torch.equal(starts[rows], rollout.terminals[agents, times])
            assert torch.equal(state[:, rows], carries[window][:, agents])


def test_the_carries_stay_those_of_the_collection_weights_in_every_pass() -> None:
    """The second pass reads the carries the first did, though the weights have moved."""
    rollout, step = _rollout(), _step()
    learner = _learner(num_minibatches=1)
    before = [weight.clone() for weight in step.model.parameters()]
    calls = _record_windows(step, learner, rollout)
    assert [len(agents) for agents, _ in calls] == [AGENTS, AGENTS]
    first, second = (
        state[:, _windowed(agents).argsort()] for agents, (_, state, _) in calls
    )
    assert torch.equal(first, second)
    moved = zip(before, step.model.parameters(), strict=True)
    assert not all(torch.equal(old, new) for old, new in moved)


def test_an_epoch_steps_once_per_minibatch_of_every_pass() -> None:
    step, learner = _step(), _learner()
    learner.begin_epoch(step, 0)
    with patch.object(step.optimizer, "step", wraps=step.optimizer.step) as stepped:
        losses, metrics = learner(step, _rollout())
    assert stepped.call_count == 2 * 2
    assert losses.shape == (len(TorchPPO.Config.LOSS_NAMES),)
    assert bool(losses.isfinite().all())
    assert metrics == {}


def test_the_loss_scores_the_first_environments_without_updating() -> None:
    step, learner = _step(), _learner()
    before = [weight.clone() for weight in step.model.parameters()]
    with torch.no_grad():
        total, losses = learner.loss(step, _rollout())
    names = TorchPPO.Config.LOSS_NAMES
    assert total == losses[names.index("total_loss")]
    after = step.model.parameters()
    assert all(torch.equal(old, new) for old, new in zip(before, after, strict=True))


def test_each_epoch_draws_its_own_shuffles_from_the_seed() -> None:
    """Keyed by the seed and the epoch, so a resumed run draws what it would have."""
    step, learner = _step(), _learner()
    learner.begin_epoch(step, 3)
    assert learner.order is not None
    third = learner.order.clone()
    assert torch.equal(third[0].sort().values, torch.arange(AGENTS))
    learner.begin_epoch(step, 4)
    assert not torch.equal(learner.order, third)
    learner.begin_epoch(step, 3)
    assert torch.equal(learner.order, third)


def test_the_windows_keep_no_state_and_refuse_anothers() -> None:
    learner = _learner()
    assert learner.state_dict() == {}
    learner.load_state_dict({})
    with pytest.raises(ValueError, match="no state"):
        learner.load_state_dict({"order": torch.zeros(2)})


def test_one_epoch_matches_its_bfb_golden() -> None:
    """The replay, the advantages, one minibatch of every window and Adam's step.

    One pass of one minibatch, not the recipe's four of eight: in host-agnostic
    numerics each Adam step is a third of the test on x86, and the passes' and
    minibatches' loops are the unit tests'.
    """
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="gtrxl_windows_tiny",
        build_module=_policy().make,
        build_input=lambda: _rollout_fields(seed=1),
        run=_learn,
    )


def test_a_step_learning_from_trajectory_windows_resumes_exactly() -> None:
    """Epoch 1's checkpoint, loaded into the step after epoch 2: epoch 2 runs again.

    The checkpoint holds the rollout's carry, so the resumed actor reads the
    memory the uninterrupted one did, and the learner replays it alike.
    Rollouts of one window, learned in one minibatch, from a pool of one
    world: the windows' carries, the passes and the minibatches are the tests'
    above.
    """
    config = _tiny_step_config()
    config.rollout.horizon = WINDOW
    config.train_budget_steps = 2
    learner = config.learner
    assert isinstance(learner, TrajectoryWindows.Config)
    learner.num_passes = learner.num_minibatches = 1
    pool = config.env.restart
    assert isinstance(pool, WorldPool.Config)
    pool.num_worlds = 1
    _assert_resumes_exactly(config)


@pytest.mark.gpu_triton
def test_captured_epochs_resume_exactly_on_the_gpu() -> None:
    """The captured rollout steps and learner epoch keep the memory and the bits."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    config = _tiny_step_config()
    config.parallelism.device = "cuda"
    config.sampler = PhiloxSampler.Config()
    config.train_budget_steps = 4
    _assert_resumes_exactly(config)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Step:
    """The parts of a train step the windows read."""

    model: GTrXLPolicy
    optimizer: torch.optim.Optimizer
    device: torch.device


class _CarrySpy:
    """Wrap a policy's ``forward_fused``, keeping a copy of every carry it reads."""

    def __init__(self, forward: Callable[..., tuple[Tensor, Tensor]]) -> None:
        self.forward = forward
        self.seen: dict[int, list[Tensor]] = {}

    def __call__(
        self,
        observations: Tensor,
        state: Tensor,
        *args: object,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor]:
        self.seen.setdefault(state.data_ptr(), []).append(state.clone())
        return self.forward(observations, state, *args, **kwargs)


class _WindowSpy:
    """Wrap a policy's ``forward_sequence``, keeping each window's inputs."""

    def __init__(self, forward: Callable[..., tuple[Tensor, Tensor, Tensor]]) -> None:
        self.forward = forward
        self.seen: list[tuple[Tensor, Tensor, Tensor]] = []

    def __call__(
        self,
        observations: Tensor,
        state: Tensor,
        episode_start: Tensor,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.seen.append((observations, state, episode_start))
        return self.forward(observations, state, episode_start, **kwargs)


# One layer and one-layer towers, not the recipe's two: Adam's steps over each parameter
# are most of an epoch in host-agnostic numerics, and ``gtrxl_test`` stacks the layers.
def _policy(
    *,
    observation_size: int = OBSERVATION_SIZE,
    num_actions: int = NUM_ACTIONS,
) -> GTrXLPolicy.Config:
    """Return the policy at test size: 1 layer of 6, 2 heads of 3, 5 rows, towers of 1 x 7."""
    config = GTrXLPolicy.Config()
    config.observation_size = observation_size
    config.num_actions = num_actions
    config.num_layers = config.decoder.num_layers = 1
    config.channels_hidden = 6
    config.memory_length = MEMORY
    config.block.heads = 2
    config.block.channels_head = 3
    config.decoder.channels_hidden = 7
    return config


def _learner(*, num_minibatches: int = 2, num_passes: int = 2) -> TrajectoryWindows:
    """Return passes, two by default, of windows of 2 over the tiny step, prepared."""
    config = TrajectoryWindows.Config()
    config.num_passes = num_passes
    config.num_minibatches = num_minibatches
    config.window = WINDOW
    learner = config.make()
    learner.prepare(_step_config())
    return learner


def _tiny_step_config() -> CraftaxTrainStep.Config:
    """Return the GTrXL recipe on the tiny CPU pipeline: 4 environments, windows of 2."""
    config = tiny_train_step()
    config.model = _policy(observation_size=OBS_SIZE, num_actions=ATN_DIM)
    config.optimizer = PartialConfig(ClippedAdam, lr=1e-3, eps=1e-5, max_grad_norm=1.0)
    learner = config.learner = TrajectoryWindows.Config()
    learner.num_passes = 2
    learner.num_minibatches = 2
    learner.window = WINDOW
    config.rollout.num_slots = 1
    config.rollout.bootstrap = True
    config.reward_clip = math.inf
    return config


# Loaded into the step that moved on, the checkpoint must overwrite all it changed.
def _assert_resumes_exactly(config: CraftaxTrainStep.Config) -> None:
    """Train an epoch, checkpoint, train on; reload and check the rest replays bit for bit."""
    epochs = int(config.train_budget_steps) - 1
    torch.manual_seed(0)
    step = config.make()
    try:
        step.train_step()
        # A copy, not a file: ``train_step_test`` reads a step's state back through
        # ``torch.load``, whose unpickler took a sixth of this test on x86.
        saved = copy.deepcopy(step.state_dict())
        expected = [step.train_step()["model"] for _ in range(epochs)]
        expected += [p.detach().clone() for p in step.model.parameters()]
        step.load_state_dict(saved)
        actual = [step.train_step()["model"] for _ in range(epochs)]
        actual += list(step.model.parameters())
    finally:
        step.close()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert bool(torch.isfinite(torch.stack(actual[:epochs])).all())


def _step_config() -> CraftaxTrainStep.Config:
    """Return a step's recipe of the tiny geometry: one slot of 4 steps and a bootstrap row."""
    config = CraftaxTrainStep.Config()
    config.env.num_envs = AGENTS
    config.rollout.horizon = HORIZON
    config.rollout.num_slots = 1
    config.rollout.bootstrap = True
    return config


def _step() -> _Step:
    """Return the tiny policy and its Adam, as a train step holds them."""
    torch.manual_seed(0)
    model = _policy().make()
    optimizer = ClippedAdam(model.parameters(), lr=1e-2, eps=1e-5, max_grad_norm=1.0)
    return _Step(model=model, optimizer=optimizer, device=torch.device("cpu"))


def _rollout_fields(*, seed: int) -> dict[str, Tensor]:
    """Draw a rollout's tensors, its step-0 carry two rows deep for some agents."""
    generator = torch.Generator().manual_seed(seed)
    rows = HORIZON + 1
    carry = torch.randn(MEMORY, AGENTS, 6 + 1, generator=generator)
    written = torch.arange(MEMORY)[:, None] >= MEMORY - 2 * (torch.arange(AGENTS) % 2)
    carry = torch.where(written[..., None], carry, 0.0)
    carry[..., -1] = written.float()
    shape = (AGENTS, rows)
    return {
        "observations": torch.randn(*shape, OBSERVATION_SIZE, generator=generator),
        "actions": torch.randint(NUM_ACTIONS, shape, generator=generator).float(),
        "logprobs": torch.randn(*shape, generator=generator) - NUM_ACTIONS**0.5,
        "rewards": torch.randn(*shape, generator=generator),
        "terminals": (torch.rand(*shape, generator=generator) < 0.3).float(),
        "values": torch.randn(*shape, generator=generator),
        "action_mask": torch.ones(*shape, NUM_ACTIONS),
        "initial_states": carry,
        "branch_starts": torch.zeros(AGENTS, dtype=torch.uint8),
    }


def _rollout(*, seed: int = 0) -> LearnerRollout:
    return _as_rollout(_rollout_fields(seed=seed))


def _as_rollout(tensors: dict[str, Tensor]) -> LearnerRollout:
    """Return :func:`_rollout_fields`' tensors as the learner's rollout."""
    return LearnerRollout(
        observations=tensors["observations"],
        actions=tensors["actions"],
        logprobs=tensors["logprobs"],
        rewards=tensors["rewards"],
        terminals=tensors["terminals"],
        values=tensors["values"],
        action_mask=tensors["action_mask"],
        initial_states=tensors["initial_states"],
        branch_starts=tensors["branch_starts"],
    )


def _learner_rollout(storage: RolloutStorage) -> LearnerRollout:
    """Return a slot in the learner's layout, its rewards as stored."""
    return LearnerRollout.from_time_major(
        storage.observations,
        storage.actions,
        storage.logprobs,
        storage.rewards,
        storage.terminals,
        storage.values,
        storage.action_mask,
        storage.initial_states,
        storage.branch_starts,
        reward_scale=1.0,
        reward_clip=math.inf,
    )


def _windowed(agents: Tensor) -> Tensor:
    """Return each window row's ``window * AGENTS + environment``, as the windows fold them."""
    windows = torch.arange(HORIZON // WINDOW)
    return (windows[:, None] * AGENTS + agents[None, :]).flatten()


def _record_windows(
    step: _Step,
    learner: TrajectoryWindows,
    rollout: LearnerRollout,
) -> list[tuple[Tensor, tuple[Tensor, Tensor, Tensor]]]:
    """Run one epoch; return each minibatch's environments and window inputs."""
    learner.begin_epoch(step, 0)
    assert learner.order is not None
    agents = list(learner.order.reshape(-1, learner.rows))
    spy = _WindowSpy(step.model.forward_sequence)
    with patch.object(step.model, "forward_sequence", spy):
        learner(step, rollout)
    return list(zip(agents, spy.seen, strict=True))


def _learn(module: nn.Module, inputs: dict[str, Tensor]) -> Tensor:
    """Run one epoch of the windows on the tiny policy; return its mean terms."""
    assert isinstance(module, GTrXLPolicy)
    optimizer = ClippedAdam(module.parameters(), lr=1e-2, eps=1e-5, max_grad_norm=1.0)
    step = _Step(model=module, optimizer=optimizer, device=torch.device("cpu"))
    learner = _learner(num_minibatches=1, num_passes=1)
    learner.begin_epoch(step, 0)
    losses, _ = learner(step, _as_rollout(inputs))
    return losses


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
