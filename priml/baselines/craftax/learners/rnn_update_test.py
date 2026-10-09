"""Unit tests for the recurrent PPO epoch over shuffled whole trajectories, at tiny sizes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, cast

import io
import math

from configgle import PartialConfig

import pytest
import torch

from priml.baselines.craftax.experiments import exp005
from priml.baselines.craftax.game.state import ATN_DIM, OBS_SIZE
from priml.baselines.craftax.learners.rnn_update import ShuffledTrajectories
from priml.baselines.craftax.learners.update import shuffles
from priml.baselines.craftax.lib.adam import ClippedAdam
from priml.baselines.craftax.model import FeasibilityLoss, Policy
from priml.baselines.craftax.policies.rnn import ActorCriticRNN
from priml.baselines.craftax.rollout import (
    PhiloxSampler,
    Rollout,
    TorchPhiloxSampler,
)
from priml.baselines.craftax.testing import (
    FakeEnv,
    assert_golden,
    digest,
    fp32,
    host_agnostic_pipeline,
    packed_observations,
    portable_uniform,
    tiny_env,
    tiny_policy,
    tiny_train_step,
)
from priml.baselines.craftax.train_step import (
    CraftaxTrainStep,
    LearnerRollout,
)
from priml.lib.codec import from_plain
from priml.loss.policy_gradient import TorchPPO


if TYPE_CHECKING:
    from torch import Tensor

    from priml.train.custom_types import TrainStepOutput


AGENTS: Final = 6
HORIZON: Final = 3
OBSERVATION: Final = 7
WIDTH: Final = 5
ACTIONS: Final = 3

_TOTAL: Final = TorchPPO.Config.LOSS_NAMES.index("total_loss")


def test_a_minibatch_scores_the_references_objective_over_whole_trajectories() -> None:
    """``ppo_rnn.py``'s ``_loss_fn``: the GRU replayed from each agent's carry, then PPO.

    The reference here steps the policy one observation at a time, the
    rollout's way, zeroing the carry by hand where an episode starts; some
    ratios and values sit outside the clip. Every term is compared, the
    diagnostics with the loss.
    """
    torch.manual_seed(0)
    policy = _tiny_policy().make()
    rollout = _rollout()
    agents = torch.tensor([4, 1])
    generator = torch.Generator().manual_seed(1)
    advantages = torch.randn(2, HORIZON, generator=generator)
    returns = torch.randn(2, HORIZON, generator=generator)
    update = ShuffledTrajectories(ShuffledTrajectories.Config())
    total, losses = update.score_trajectories(
        policy,
        rollout,
        agents=agents,
        advantages=advantages,
        returns=returns,
    )
    total.backward()
    actual = [p.grad.clone() for p in policy.parameters() if p.grad is not None]
    policy.zero_grad()
    expected = _reference_trajectory_terms(
        policy,
        rollout,
        agents=agents,
        advantages=advantages,
        returns=returns,
    )
    expected[_TOTAL].backward()
    torch.testing.assert_close(total, expected[_TOTAL], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(losses, expected.detach(), rtol=1e-5, atol=1e-6)
    assert len(actual) == len(list(policy.parameters()))
    for a, parameter in zip(actual, policy.parameters(), strict=True):
        assert parameter.grad is not None
        torch.testing.assert_close(a, parameter.grad, rtol=1e-5, atol=1e-7)
    assert losses[_TOTAL] == total


def test_a_policys_own_loss_joins_the_objective_scored_on_the_actions_taken() -> None:
    """A MinGRU with a feasibility head keeps a carry too; its loss reads the actions."""
    torch.manual_seed(0)
    config = tiny_policy(dtype=torch.float32)
    feasibility = config.auxiliary = FeasibilityLoss.Config()
    feasibility.coefficient = 0.1
    policy = config.make()
    generator = torch.Generator().manual_seed(2)
    rows = HORIZON + 1
    rollout = replace(
        _rollout(),
        observations=packed_observations(config, batch=AGENTS, time=rows, seed=3),
        actions=torch.randint(ATN_DIM, (AGENTS, rows), generator=generator).float(),
        action_mask=torch.ones(AGENTS, rows, ATN_DIM),
        initial_states=torch.randn_like(policy.initial_state(AGENTS)),
    )
    agents = torch.tensor([4, 1])
    update = ShuffledTrajectories(ShuffledTrajectories.Config())
    total, losses = update.score_trajectories(
        policy,
        rollout,
        agents=agents,
        advantages=torch.randn(2, HORIZON, generator=generator),
        returns=torch.randn(2, HORIZON, generator=generator),
    )
    with torch.no_grad():
        auxiliary = policy.forward_sequence(
            rollout.observations[agents, :HORIZON],
            rollout.initial_states[:, agents],
            rollout.terminals[agents, :HORIZON],
            actions=rollout.actions[agents, :HORIZON],
        )[2]
    assert float(auxiliary) > 0
    torch.testing.assert_close(total.detach(), losses[_TOTAL] + auxiliary)


def test_each_agents_advantages_are_its_rollouts_gae_from_the_row_after() -> None:
    """``ppo_rnn.py``'s ``_calculate_gae``, one row per agent.

    Row ``t`` stores what arrived with observation ``t``, so the reward and
    done of the transition that leaves it are row ``t + 1``'s; the bootstrap
    value is the row past the horizon.
    """
    rollout = _rollout()
    update = ShuffledTrajectories(ShuffledTrajectories.Config())
    advantages, returns = update.trajectory_advantages(rollout)
    expected = torch.zeros(AGENTS, HORIZON)
    gae = torch.zeros(AGENTS)
    next_value = rollout.values[:, HORIZON]
    for t in reversed(range(HORIZON)):
        done = rollout.terminals[:, t + 1]
        delta = rollout.rewards[:, t + 1] + 0.99 * next_value * (1 - done)
        delta = delta - rollout.values[:, t]
        gae = delta + 0.99 * 0.8 * (1 - done) * gae
        expected[:, t] = gae
        next_value = rollout.values[:, t]
    assert torch.equal(advantages, expected)
    assert torch.equal(returns, expected + rollout.values[:, :HORIZON])


def test_an_epoch_steps_once_per_minibatch_from_the_weights_before_it() -> None:
    """Three minibatches of two whole trajectories, learned in order; their mean terms.

    The reference replays each minibatch by hand, as the first test does, and
    takes the same SGD step after it, so each minibatch is scored with the
    weights the ones before it moved. The episodes that start inside the
    window reset the carry: replayed without the reset, the window differs.
    """
    torch.manual_seed(0)
    config = _tiny_policy()
    policy = config.make()
    reference = config.make()
    reference.load_state_dict(policy.state_dict())
    rollout = _rollout()
    minibatches = torch.tensor([[4, 1], [3, 0], [5, 2]])
    assert bool(rollout.terminals[minibatches, :HORIZON].any())
    update = ShuffledTrajectories(ShuffledTrajectories.Config())
    losses = update.learn(
        policy,
        torch.optim.SGD(policy.parameters(), lr=0.5),
        rollout,
        minibatches,
    )
    advantages, returns = update.trajectory_advantages(rollout)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.5)
    terms: list[Tensor] = []
    for agents in minibatches:
        optimizer.zero_grad()
        expected = _reference_trajectory_terms(
            reference,
            rollout,
            agents=agents,
            advantages=advantages[agents],
            returns=returns[agents],
        )
        expected[_TOTAL].backward()
        optimizer.step()
        terms.append(expected.detach())
    torch.testing.assert_close(
        losses,
        torch.stack(terms).mean(dim=0),
        rtol=1e-5,
        atol=1e-6,
    )
    for actual, expected in zip(
        policy.parameters(),
        reference.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    with torch.no_grad():
        reset = _replay_by_hand(reference, rollout, minibatches.flatten())
        carried = _replay_by_hand(
            reference,
            rollout,
            minibatches.flatten(),
            reset=False,
        )
    assert not torch.allclose(reset, carried)


def test_prepare_cuts_the_agents_into_minibatches_of_whole_trajectories() -> None:
    """The tiny step's 4 agents of 4 steps, in two minibatches: 2 agents, 8 transitions."""
    config = ShuffledTrajectories.Config()
    config.num_minibatches = 2
    update = config.make()
    update.prepare(_tiny_config())
    assert (update.rows, update.size) == (2, 2 * 4)


def test_the_learner_replays_what_the_actor_scored() -> None:
    """From the stored carries, the learner's values are the rollout's own, bit for bit.

    The rollout steps each buffer's two agents one observation at a time and
    the learner replays all four agents' trajectories at once; inside the
    host-agnostic pipeline both round alike.
    """
    torch.manual_seed(0)
    policy = _fake_env_policy().make()
    with host_agnostic_pipeline():
        learner = _collected(policy, envs=4)
        with torch.no_grad():
            decoded, _, _ = policy.forward_sequence(
                learner.observations[:, :HORIZON],
                learner.initial_states,
                learner.terminals[:, :HORIZON],
            )
    # The rollout starts mid-episode: its carries are not zero.
    assert bool(learner.initial_states.any())
    assert torch.equal(decoded[..., -1], learner.values[:, :HORIZON])


def test_an_epoch_learns_each_shuffled_agent_as_the_reference_replays_it() -> None:
    """Two passes over 4 agents' windows collected on ``FakeEnv``, 1 per minibatch.

    The windows start mid-episode, from nonzero carries, and an episode starts
    inside them. The reference replays each minibatch by hand from its agent's
    stored carry, logits masked as sampled, and takes the same SGD step after
    it, so each is scored with the weights the ones before it moved.
    """
    torch.manual_seed(0)
    config = _fake_env_policy()
    policy, reference = config.make(), config.make()
    reference.load_state_dict(policy.state_dict())
    rollout = _collected(policy, envs=4)
    assert bool(rollout.initial_states.any())
    assert bool(rollout.terminals[:, 1:HORIZON].any())
    learner = _learner()
    step = _Step(
        config=_window(),
        device=torch.device("cpu"),
        model=policy,
        optimizer=torch.optim.SGD(policy.parameters(), lr=0.5),
    )
    learner.begin_epoch(step, 1)
    losses, metrics = learner(step, rollout)
    order = shuffles(seed=0, epoch=1, transitions=4, passes=2)
    assert learner.order is not None
    assert torch.equal(learner.order, order)
    advantages, returns = learner.trajectory_advantages(rollout)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.5)
    terms: list[Tensor] = []
    for agents in order.reshape(-1, 1):
        optimizer.zero_grad()
        expected = _reference_trajectory_terms(
            reference,
            rollout,
            agents=agents,
            advantages=advantages[agents],
            returns=returns[agents],
        )
        expected[_TOTAL].backward()
        optimizer.step()
        terms.append(expected.detach())
    assert metrics == {}
    torch.testing.assert_close(
        losses,
        torch.stack(terms).mean(dim=0),
        rtol=1e-5,
        atol=1e-6,
    )
    for actual, expected in zip(
        policy.parameters(),
        reference.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_the_loss_scores_the_first_agents_window_unshuffled() -> None:
    """Agent 0's whole window from its stored carry, with no optimizer step."""
    torch.manual_seed(0)
    policy = _fake_env_policy().make()
    rollout = _collected(policy, envs=4)
    learner = _learner()
    step = _Step(
        config=_window(),
        device=torch.device("cpu"),
        model=policy,
        optimizer=torch.optim.SGD(policy.parameters(), lr=0.5),
    )
    before = [p.detach().clone() for p in policy.parameters()]
    total, losses = learner.loss(step, rollout)
    advantages, returns = learner.trajectory_advantages(rollout)
    expected = _reference_trajectory_terms(
        policy,
        rollout,
        agents=torch.tensor([0]),
        advantages=advantages[:1],
        returns=returns[:1],
    )
    torch.testing.assert_close(total, expected[_TOTAL], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(losses, expected.detach(), rtol=1e-5, atol=1e-6)
    assert all(
        torch.equal(a, b) for a, b in zip(before, policy.parameters(), strict=True)
    )


def test_minibatches_that_do_not_split_the_agents_are_refused() -> None:
    """The tiny step's 16 transitions split into 8, but its 4 agents do not."""
    config = _tiny_config()
    learner = config.learner
    assert isinstance(learner, ShuffledTrajectories.Config)
    learner.num_minibatches = 8
    with pytest.raises(ValueError, match="8 minibatches must split the 4 agents"):
        config.make()


def test_a_rollout_without_the_bootstrap_row_is_refused_at_construction() -> None:
    config = _tiny_config()
    config.rollout.bootstrap = False
    with pytest.raises(ValueError, match="bootstrap"):
        config.make()


def test_a_step_learning_from_shuffled_trajectories_resumes_exactly() -> None:
    """Two epochs straight against a checkpoint after the first and one more.

    Each epoch's shuffles are of the four agents, two passes of two
    minibatches of two, over rollouts of 2 steps and the bootstrap row.
    """
    config = _tiny_config()
    config.rollout.horizon = 2
    config.train_budget_steps = 2
    straight = _make(config)
    try:
        learner = straight.learner
        assert isinstance(learner, ShuffledTrajectories)
        straight.train_step()
        assert learner.order is not None
        assert torch.equal(
            learner.order,
            shuffles(seed=0, epoch=0, transitions=4, passes=2),
        )
        saved = io.BytesIO()
        torch.save(straight.state_dict(), saved)
        expected = [straight.train_step()["model"]]
        expected += [p.detach().clone() for p in straight.model.parameters()]
        # The rollout keeps its bootstrap row, and the GRU's carry moved on.
        assert straight.rollout.slots[0].observations.shape[0] == 3
        assert bool(straight.rollout.slots[0].initial_states.any())
    finally:
        straight.close()
    resumed = _make(config)
    try:
        resumed.load_state_dict(
            from_plain(
                cast(
                    "object",
                    torch.load(io.BytesIO(saved.getvalue()), weights_only=True),
                ),
                dict[str, object],
            ),
        )
        actual = [resumed.train_step()["model"]]
        actual += list(resumed.model.parameters())
    finally:
        resumed.close()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert bool(torch.isfinite(actual[0]).all())


@pytest.mark.gpu_triton
def test_captured_epochs_resume_exactly_on_the_gpu() -> None:
    """The captured learner replays the GRU over the shuffles drawn between its replays."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    config = _tiny_config()
    config.parallelism.device = "cuda"
    config.sampler = PhiloxSampler.Config()
    config.train_budget_steps = 4
    straight = _make(config)
    try:
        straight.train_step()
        saved = io.BytesIO()
        torch.save(straight.state_dict(), saved)
        expected = [straight.train_step()["model"] for _ in range(3)]
        expected += [p.detach().clone() for p in straight.model.parameters()]
    finally:
        straight.close()
    resumed = _make(config)
    try:
        resumed.load_state_dict(
            from_plain(
                cast(
                    "object",
                    torch.load(io.BytesIO(saved.getvalue()), weights_only=True),
                ),
                dict[str, object],
            ),
        )
        actual = [resumed.train_step()["model"] for _ in range(3)]
        actual += list(resumed.model.parameters())
    finally:
        resumed.close()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert bool(torch.isfinite(torch.stack(actual[:3])).all())


def test_exp005s_epoch_matches_its_golden() -> None:
    """exp005 at test size from portable seed-73 weights, an epoch, frozen.

    exp005's recipe -- its GRU, shuffled trajectories, Adam and its rules --
    on a GRU of 8 with heads of one layer in 4 environments of horizon 2,
    minibatches of 2 agents: the epoch's rate, mean losses and weights. One
    pass, not the recipe's four, as for exp003's golden; the heads' second
    layers and the init's distributions are ``rnn_test``'s. Built natively,
    as the portable draws overwrite every
    weight the build draws: in host-agnostic numerics the build cost a
    third of the test.
    """
    config = exp005().step
    env = tiny_env()
    env.rules = config.env.rules
    env.restart = config.env.restart
    config.env = env
    model = config.model
    assert isinstance(model, ActorCriticRNN.Config)
    model.channels_hidden = 8
    model.num_layers = 1
    learner = config.learner
    assert isinstance(learner, ShuffledTrajectories.Config)
    learner.num_minibatches = 2
    learner.num_passes = 1
    config.parallelism.device = "cpu"
    config.sampler = TorchPhiloxSampler.Config().update(
        config.sampler,
        skip_missing=True,
    )
    config.rollout.horizon = 2
    config.train_budget_steps = 1
    step = _make(config)
    assert isinstance(step.model, ActorCriticRNN)
    with host_agnostic_pipeline():
        try:
            _fill_portable(step.model)
            lines = _epoch_entries(step, step.train_step())
        finally:
            step.close()
    assert_golden(test_file=__file__, name="train_step_exp005_tiny", lines=lines)


# ``testing.fill_portable`` fills MinGRU's stages alone; the GRU's cell and heads would
# keep their init, whose ``randn`` differs by host (SLEEF's ``log`` under AVX2).
def _fill_portable(model: ActorCriticRNN) -> None:
    """Overwrite every weight with ``U(+-1/sqrt(fan_in))`` drawn the same on any host."""
    generator = torch.Generator().manual_seed(73)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                portable_uniform(
                    *parameter.shape,
                    bound=parameter.shape[-1] ** -0.5,
                    generator=generator,
                ),
            )


def _epoch_entries(step: CraftaxTrainStep, result: TrainStepOutput) -> list[str]:
    """Digest one epoch: its rate, its mean loss terms, then every weight."""
    prefix = f"epoch {step.global_step:04d}"
    metrics = result.get("metrics", {})
    lines = [f"{prefix} learning_rate {fp32(metrics['learning_rate'])}"]
    lines += [
        f"{prefix} {name} {fp32(value)}"
        for name, value in zip(TorchPPO.Config.LOSS_NAMES, result["model"], strict=True)
    ]
    return lines + [
        f"{prefix} weight {name} {digest(value)}"
        for name, value in step.model.named_parameters()
    ]


# Returns the ``TorchPPO.Config.LOSS_NAMES`` terms, the total differentiable: the
# reference's three terms and total, then the port's diagnostics of the log-ratio.
def _reference_terms(
    decoded: Tensor,
    *,
    actions: Tensor,
    old_logprobs: Tensor,
    old_values: Tensor,
    advantages: Tensor,
    targets: Tensor,
) -> Tensor:
    """``_loss_fn`` of Craftax_Baselines' ``ppo_rnn.py``, in torch, its epsilon fp32's."""
    log_probs = decoded[:, :-1].log_softmax(dim=-1)
    value = decoded[:, -1]
    log_prob = log_probs.gather(-1, actions[:, None])[:, 0]
    value_pred_clipped = old_values + (value - old_values).clip(-0.2, 0.2)
    value_loss = (
        0.5
        * torch.maximum(
            (value - targets).square(),
            (value_pred_clipped - targets).square(),
        ).mean()
    )
    log_ratio = log_prob - old_logprobs
    ratio = log_ratio.exp()
    gae = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + torch.finfo(torch.float32).eps
    )
    loss_actor = -torch.minimum(ratio * gae, ratio.clip(0.8, 1.2) * gae).mean()
    entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
    total = loss_actor + 0.5 * value_loss - 0.01 * entropy
    return torch.stack(
        (
            loss_actor,
            value_loss,
            entropy,
            total,
            (-log_ratio).mean(),
            ((ratio - 1) - log_ratio).mean(),
            ((ratio - 1).abs() > 0.2).float().mean(),
            ratio.mean(),
        ),
    )


def _reference_trajectory_terms(
    policy: ActorCriticRNN,
    rollout: LearnerRollout,
    *,
    agents: Tensor,
    advantages: Tensor,
    returns: Tensor,
) -> Tensor:
    """Return :func:`_reference_terms` of ``agents``' trajectories, replayed by hand."""
    decoded = _replay_by_hand(policy, rollout, agents)
    # The sampler's mask, as the learner applies it; an all-legal one changes nothing.
    logits = torch.where(
        rollout.action_mask[agents, :HORIZON].flatten(0, 1) != 0,
        decoded[:, :-1],
        TorchPPO.Config.MASKED_LOGIT,
    )
    return _reference_terms(
        torch.cat((logits, decoded[:, -1:]), dim=-1),
        actions=rollout.actions[agents, :HORIZON].flatten().long(),
        old_logprobs=rollout.logprobs[agents, :HORIZON].flatten(),
        old_values=rollout.values[agents, :HORIZON].flatten(),
        advantages=advantages.flatten(),
        targets=returns.flatten(),
    )


# With ``reset`` the carry is zeroed by hand where an episode starts, and the policy's
# own reset is never asked for.
def _replay_by_hand(
    policy: ActorCriticRNN,
    rollout: LearnerRollout,
    agents: Tensor,
    *,
    reset: bool = True,
) -> Tensor:
    """Step ``agents`` through the window one observation at a time; flat fused rows."""
    carry = rollout.initial_states[:, agents]
    steps: list[Tensor] = []
    for step in range(HORIZON):
        if reset:
            carry = carry * (1 - rollout.terminals[agents, step])[None, :, None]
        decoded, carry = policy.forward_fused(
            rollout.observations[agents, step],
            carry,
            None,
        )
        steps.append(decoded)
    return torch.stack(steps, dim=1).flatten(0, 1)


def _rollout() -> LearnerRollout:
    """Return a random agent-major rollout with its bootstrap row and nonzero carries."""
    generator = torch.Generator().manual_seed(0)
    rows = HORIZON + 1
    return LearnerRollout(
        observations=torch.randn(AGENTS, rows, OBSERVATION, generator=generator),
        actions=torch.randint(ACTIONS, (AGENTS, rows), generator=generator).float(),
        logprobs=torch.randn(AGENTS, rows, generator=generator) - ACTIONS**0.5,
        rewards=torch.randn(AGENTS, rows, generator=generator),
        terminals=(torch.rand(AGENTS, rows, generator=generator) < 0.3).float(),
        values=torch.randn(AGENTS, rows, generator=generator),
        action_mask=torch.ones(AGENTS, rows, ACTIONS),
        # One layer: ``ActorCriticRNN.initial_state`` carries the cell's one state.
        initial_states=torch.randn(1, AGENTS, WIDTH, generator=generator),
        branch_starts=torch.zeros(AGENTS, dtype=torch.uint8),
    )


def _tiny_policy() -> ActorCriticRNN.Config:
    """Return the policy at test size: 7 inputs, a GRU of 5, heads of 2 layers, 3 actions."""
    config = ActorCriticRNN.Config()
    config.observation_size = OBSERVATION
    config.channels_hidden = WIDTH
    config.num_actions = ACTIONS
    return config


@dataclass(frozen=True, slots=True, kw_only=True)
class _Step:
    """What the epoch reads of a train step, around a policy the test built."""

    config: CraftaxTrainStep.Config
    device: torch.device
    model: Policy
    optimizer: torch.optim.Optimizer


def _fake_env_policy() -> ActorCriticRNN.Config:
    """Return :func:`_tiny_policy` at the fake env's widths: packed observations, every action."""
    config = _tiny_policy()
    config.observation_size = OBS_SIZE
    config.num_actions = ATN_DIM
    return config


# It starts mid-episode, where a rollout before it would have left the env and the
# carries: the env stepped a horizon, the carries drawn. It keeps its bootstrap row.
def _collected(policy: Policy, *, envs: int) -> LearnerRollout:
    """Return a window ``policy`` collected on ``FakeEnv``, agent-major."""
    config = Rollout.Config()
    config.num_slots = 1
    config.horizon = HORIZON
    config.bootstrap = True
    env = FakeEnv.make(num_envs=envs, num_buffers=2, seed=0)
    for buffer in range(env.num_buffers):
        for _ in range(HORIZON):
            env.step_buffer(buffer)
    rollout = Rollout(
        config,
        policy=policy,
        sampler=TorchPhiloxSampler.Config().make(),
        env=env,
        device=torch.device("cpu"),
    )
    try:
        generator = torch.Generator().manual_seed(1)
        for graph in rollout.graphs[0]:
            graph.state.copy_(torch.randn(graph.state.shape, generator=generator))
        storage = rollout.collect(0)
    finally:
        rollout.close()
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


def _learner() -> ShuffledTrajectories:
    """Return exp005's epoch at two passes of four minibatches, one agent each."""
    config = ShuffledTrajectories.Config()
    config.num_passes = 2
    config.num_minibatches = 4
    learner = config.make()
    learner.prepare(_window())
    return learner


def _window() -> CraftaxTrainStep.Config:
    """Return a step recipe whose rollout is 4 agents' collected windows."""
    config = CraftaxTrainStep.Config()
    config.env.num_envs = 4
    config.rollout.horizon = HORIZON
    config.rollout.bootstrap = True
    return config


def _tiny_config() -> CraftaxTrainStep.Config:
    """Return exp005's learner on exp000's tiny CPU pipeline: two passes of two minibatches."""
    config = tiny_train_step()
    model = config.model = ActorCriticRNN.Config()
    model.observation_size = OBS_SIZE
    model.channels_hidden = 8
    config.optimizer = PartialConfig(ClippedAdam, lr=1e-3, eps=1e-5, max_grad_norm=1.0)
    learner = config.learner = ShuffledTrajectories.Config()
    learner.num_passes = 2
    learner.num_minibatches = 2
    config.rollout.num_slots = 1
    config.rollout.bootstrap = True
    config.reward_clip = math.inf
    return config


def _make(config: CraftaxTrainStep.Config) -> CraftaxTrainStep:
    torch.manual_seed(0)
    return config.make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
