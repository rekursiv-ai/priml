"""Unit tests for the standard PPO epoch over shuffled transitions, at tiny sizes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, cast

import io
import math

from configgle import PartialConfig

import pytest
import torch

from priml.baselines.craftax.game.state import OBS_SIZE
from priml.baselines.craftax.learners.update import ShuffledTransitions, shuffles
from priml.baselines.craftax.lib.adam import ClippedAdam
from priml.baselines.craftax.model import Policy
from priml.baselines.craftax.policies.actor_critic import ActorCritic
from priml.baselines.craftax.rollout import (
    PhiloxSampler,
    Rollout,
    TorchPhiloxSampler,
)
from priml.baselines.craftax.testing import FakeEnv, tiny_train_step
from priml.baselines.craftax.train_step import (
    CraftaxTrainStep,
    LearnerRollout,
)
from priml.lib.codec import from_plain
from priml.loss.policy_gradient import TorchPPO


if TYPE_CHECKING:
    from torch import Tensor


ENVS: Final = 2
HORIZON: Final = 3
"""The collected window: 2 environments of 3 steps, 6 transitions."""


def test_the_advantage_is_the_references_gae_bootstrapped_from_the_row_after() -> None:
    """``ppo.py``'s ``_calculate_gae``, transition ``t`` leaving observation ``t``.

    The rollout stores beside observation ``t`` what arrived with it, so the
    reward and done of the transition that leaves it are row ``t + 1``'s, and
    the bootstrap value is the row past the horizon.
    """
    agents, horizon = 3, 5
    rollout = _rollout(agents=agents, rows=horizon + 1)
    advantages, returns = ShuffledTransitions(ShuffledTransitions.Config()).advantage(
        rollout,
    )
    expected = torch.zeros(agents, horizon)
    gae = torch.zeros(agents)
    next_value = rollout.values[:, horizon]
    for t in reversed(range(horizon)):
        done = rollout.terminals[:, t + 1]
        delta = rollout.rewards[:, t + 1] + 0.99 * next_value * (1 - done)
        delta = delta - rollout.values[:, t]
        gae = delta + 0.99 * 0.8 * (1 - done) * gae
        expected[:, t] = gae
        next_value = rollout.values[:, t]
    assert torch.equal(advantages, expected.reshape(-1))
    assert torch.equal(returns, (expected + rollout.values[:, :horizon]).reshape(-1))


def test_a_minibatch_scores_the_references_clipped_objective() -> None:
    """``ppo.py``'s ``_loss_fn`` as written there, loss and gradients alike.

    Some ratios and values sit outside the clip, so both clips are exercised.
    """
    torch.manual_seed(0)
    config = ActorCritic.Config()
    config.observation_size = 6
    config.channels_hidden = 4
    config.num_layers = 1
    config.num_actions = 3
    policy = config.make()
    rollout = _rollout(agents=4, rows=3, observation_size=6, num_actions=3)
    agents = torch.arange(4).repeat_interleave(2)
    times = torch.arange(2).repeat(4)
    generator = torch.Generator().manual_seed(1)
    advantages = torch.randn(8, generator=generator)
    returns = torch.randn(8, generator=generator)
    update = ShuffledTransitions(ShuffledTransitions.Config())
    total, losses = update.score(
        policy,
        rollout,
        agents=agents,
        times=times,
        advantages=advantages,
        returns=returns,
    )
    total.backward()
    actual = [p.grad.clone() for p in policy.parameters() if p.grad is not None]
    policy.zero_grad()
    expected = _reference_loss(
        policy(rollout.observations[agents, times]),
        actions=rollout.actions[agents, times].long(),
        old_logprobs=rollout.logprobs[agents, times],
        old_values=rollout.values[agents, times],
        advantages=advantages,
        targets=returns,
    )
    expected.backward()
    torch.testing.assert_close(total, expected, rtol=1e-6, atol=1e-7)
    for a, parameter in zip(actual, policy.parameters(), strict=True):
        assert parameter.grad is not None
        torch.testing.assert_close(a, parameter.grad, rtol=1e-5, atol=1e-7)
    assert losses[TorchPPO.Config.LOSS_NAMES.index("total_loss")] == total


def test_a_policy_with_a_carry_is_refused() -> None:
    """A transition scored alone would read its carry from the rollout's first step."""
    config = ActorCritic.Config()
    config.observation_size = 6
    config.channels_hidden = 4
    config.num_layers = 1
    config.num_actions = 3
    # Two layers of 5 for 4 agents: a carry, where the actor-critic's is empty.
    rollout = replace(_rollout(agents=4, rows=3), initial_states=torch.zeros(2, 4, 5))
    update = ShuffledTransitions(ShuffledTransitions.Config())
    with pytest.raises(ValueError, match="from no carry"):
        update.score(
            config.make(),
            rollout,
            agents=torch.arange(4),
            times=torch.zeros(4, dtype=torch.int64),
            advantages=torch.zeros(4),
            returns=torch.zeros(4),
        )


def test_each_pass_shuffles_every_transition_keyed_by_seed_and_epoch() -> None:
    first = shuffles(seed=0, epoch=0, transitions=16, passes=2)
    assert first.shape == (2, 16)
    assert first.dtype == torch.int64
    for row in first:
        assert torch.equal(row.sort().values, torch.arange(16))
    assert not torch.equal(first[0], first[1])
    assert torch.equal(shuffles(seed=0, epoch=0, transitions=16, passes=2), first)
    assert not torch.equal(shuffles(seed=0, epoch=1, transitions=16, passes=2), first)
    assert not torch.equal(shuffles(seed=1, epoch=0, transitions=16, passes=2), first)
    # Seeds that differ only above 32 bits still draw apart.
    high = shuffles(seed=2**32, epoch=0, transitions=16, passes=2)
    assert not torch.equal(high, first)


def test_an_epoch_learns_each_shuffled_minibatch_as_the_reference_replays_it() -> None:
    """Two passes of two minibatches over a window the actor collected on ``FakeEnv``.

    Epoch 1's shuffles cut the 6 transitions into minibatches of 3. The
    reference replays the epoch by hand: ``ppo.py``'s loss on each minibatch
    with the sampler's mask, then the same SGD step, so each minibatch is
    scored with the weights the ones before it moved.
    """
    torch.manual_seed(0)
    config = _tiny_policy()
    policy, reference = config.make(), config.make()
    reference.load_state_dict(policy.state_dict())
    rollout = _collected(policy)
    learner = _learner()
    step = _Step(
        config=_window(),
        device=torch.device("cpu"),
        model=policy,
        optimizer=torch.optim.SGD(policy.parameters(), lr=0.5),
    )
    learner.begin_epoch(step, 1)
    losses, metrics = learner(step, rollout)
    order = shuffles(seed=0, epoch=1, transitions=ENVS * HORIZON, passes=2)
    assert learner.order is not None
    assert torch.equal(learner.order, order)
    advantages, returns = learner.advantage(rollout)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.5)
    totals: list[Tensor] = []
    for chosen in order.reshape(-1, HORIZON):
        optimizer.zero_grad()
        expected = _masked_reference_loss(
            reference,
            rollout,
            agents=chosen // HORIZON,
            times=chosen % HORIZON,
            advantages=advantages[chosen],
            returns=returns[chosen],
        )
        expected.backward()
        optimizer.step()
        totals.append(expected.detach())
    assert metrics == {}
    total = losses[TorchPPO.Config.LOSS_NAMES.index("total_loss")]
    torch.testing.assert_close(total, torch.stack(totals).mean(), rtol=1e-5, atol=1e-6)
    for actual, expected in zip(
        policy.parameters(),
        reference.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_the_loss_scores_the_first_minibatch_unshuffled_and_learns_nothing() -> None:
    """The first 3 transitions in rollout order: environment 0's whole window."""
    torch.manual_seed(0)
    policy = _tiny_policy().make()
    rollout = _collected(policy)
    learner = _learner()
    step = _Step(
        config=_window(),
        device=torch.device("cpu"),
        model=policy,
        optimizer=torch.optim.SGD(policy.parameters(), lr=0.5),
    )
    before = [p.detach().clone() for p in policy.parameters()]
    total, losses = learner.loss(step, rollout)
    advantages, returns = learner.advantage(rollout)
    expected = _masked_reference_loss(
        policy,
        rollout,
        agents=torch.zeros(HORIZON, dtype=torch.int64),
        times=torch.arange(HORIZON),
        advantages=advantages[:HORIZON],
        returns=returns[:HORIZON],
    )
    torch.testing.assert_close(total, expected, rtol=1e-6, atol=1e-7)
    assert losses[TorchPPO.Config.LOSS_NAMES.index("total_loss")] == total.detach()
    assert all(
        torch.equal(a, b) for a, b in zip(before, policy.parameters(), strict=True)
    )


def test_begin_epoch_draws_the_epochs_shuffles_for_every_minibatch() -> None:
    """Two passes of the 16 transitions: four minibatches of 8."""
    step = _make(_tiny_config())
    try:
        learner = step.learner
        assert isinstance(learner, ShuffledTransitions)
        learner.begin_epoch(step, 3)
        assert learner.order is not None
        assert torch.equal(
            learner.order,
            shuffles(seed=0, epoch=3, transitions=16, passes=2),
        )
    finally:
        step.close()


def test_a_step_learning_from_shuffled_transitions_resumes_exactly() -> None:
    """Epoch 1's checkpoint, loaded into the step after epoch 2: epoch 2 runs again.

    Rollouts of 2 steps and the bootstrap row; the load lands on state that
    moved on, so whatever it misses shows.
    """
    config = _tiny_config()
    config.rollout.horizon = 2
    config.train_budget_steps = 2
    step = _make(config)
    try:
        step.train_step()
        saved = io.BytesIO()
        torch.save(step.state_dict(), saved)
        expected = [step.train_step()["model"]]
        expected += [p.detach().clone() for p in step.model.parameters()]
        # The rollout keeps its bootstrap row.
        assert step.rollout.slots[0].observations.shape[0] == 3
        step.load_state_dict(
            from_plain(
                cast(
                    "object",
                    torch.load(io.BytesIO(saved.getvalue()), weights_only=True),
                ),
                dict[str, object],
            ),
        )
        actual = [step.train_step()["model"]]
        actual += list(step.model.parameters())
    finally:
        step.close()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    assert bool(torch.isfinite(actual[0]).all())


def test_the_update_carries_no_state_and_refuses_another_learners() -> None:
    """Each epoch's shuffles come from the seed and the epoch, so nothing is saved."""
    update = ShuffledTransitions(ShuffledTransitions.Config())
    assert update.state_dict() == {}
    update.load_state_dict({})
    with pytest.raises(ValueError, match=r"carry no state, not \['cursor'\]"):
        update.load_state_dict({"cursor": torch.tensor(1)})


def test_a_rollout_without_the_bootstrap_row_is_refused_at_construction() -> None:
    config = _tiny_config()
    config.rollout.bootstrap = False
    with pytest.raises(ValueError, match="bootstrap"):
        config.make()


def test_minibatches_that_do_not_split_the_rollout_are_refused() -> None:
    """The tiny rollout's 16 transitions do not split into 3 minibatches."""
    config = _tiny_config()
    assert config.rollout.horizon == 4
    learner = config.learner
    assert isinstance(learner, ShuffledTransitions.Config)
    learner.num_minibatches = 3
    with pytest.raises(ValueError, match="3 minibatches must split the 16"):
        config.make()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("discount", 1.5),
        ("trace_decay", -0.1),
        ("clip_epsilon", -1.0),
        ("value_coefficient", math.inf),
        ("entropy_coefficient", math.nan),
        ("seed", -1),
        ("num_passes", 0),
        ("num_minibatches", -1),
    ],
)
def test_a_coefficient_out_of_range_is_refused(name: str, value: float) -> None:
    config = ShuffledTransitions.Config()
    setattr(config, name, value)
    with pytest.raises(ValueError, match=name):
        config.make()


@pytest.mark.gpu_triton
def test_captured_epochs_resume_exactly_on_the_gpu() -> None:
    """The captured learner replays the shuffles drawn between its replays."""
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


def _reference_loss(
    decoded: Tensor,
    *,
    actions: Tensor,
    old_logprobs: Tensor,
    old_values: Tensor,
    advantages: Tensor,
    targets: Tensor,
) -> Tensor:
    """``_loss_fn`` of Craftax_Baselines' ``ppo.py``, in torch, its epsilon fp32's."""
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
    ratio = (log_prob - old_logprobs).exp()
    gae = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + torch.finfo(torch.float32).eps
    )
    loss_actor = -torch.minimum(ratio * gae, ratio.clip(0.8, 1.2) * gae).mean()
    entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
    return loss_actor + 0.5 * value_loss - 0.01 * entropy


@dataclass(frozen=True, slots=True, kw_only=True)
class _Step:
    """What the epoch reads of a train step, around a policy the test built."""

    config: CraftaxTrainStep.Config
    device: torch.device
    model: Policy
    optimizer: torch.optim.Optimizer


def _masked_reference_loss(
    policy: ActorCritic,
    rollout: LearnerRollout,
    *,
    agents: Tensor,
    times: Tensor,
    advantages: Tensor,
    returns: Tensor,
) -> Tensor:
    """Return :func:`_reference_loss` of the transitions, their logits masked as sampled."""
    decoded = policy(rollout.observations[agents, times])
    logits = torch.where(
        rollout.action_mask[agents, times] != 0,
        decoded[:, :-1],
        TorchPPO.Config.MASKED_LOGIT,
    )
    return _reference_loss(
        torch.cat((logits, decoded[:, -1:]), dim=-1),
        actions=rollout.actions[agents, times].long(),
        old_logprobs=rollout.logprobs[agents, times],
        old_values=rollout.values[agents, times],
        advantages=advantages,
        targets=returns,
    )


def _collected(policy: Policy) -> LearnerRollout:
    """Return a window ``policy`` collected on ``FakeEnv``, agent-major, its bootstrap row kept."""
    config = Rollout.Config()
    config.num_slots = 1
    config.horizon = HORIZON
    config.bootstrap = True
    rollout = Rollout(
        config,
        policy=policy,
        sampler=TorchPhiloxSampler.Config().make(),
        env=FakeEnv.make(num_envs=ENVS, num_buffers=ENVS, seed=0),
        device=torch.device("cpu"),
    )
    try:
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


def _learner() -> ShuffledTransitions:
    """Return exp003's epoch at two passes of two minibatches, sized for the window."""
    config = ShuffledTransitions.Config()
    config.num_passes = 2
    config.num_minibatches = 2
    learner = config.make()
    learner.prepare(_window())
    return learner


def _window() -> CraftaxTrainStep.Config:
    """Return a step recipe whose rollout is the collected window, its bootstrap row kept."""
    config = CraftaxTrainStep.Config()
    config.env.num_envs = ENVS
    config.rollout.horizon = HORIZON
    config.rollout.bootstrap = True
    return config


def _tiny_policy() -> ActorCritic.Config:
    """Return exp003's policy at test size, reading ``FakeEnv``'s packed observations."""
    config = ActorCritic.Config()
    config.observation_size = OBS_SIZE
    config.channels_hidden = 4
    config.num_layers = 1
    return config


def _rollout(
    *,
    agents: int,
    rows: int,
    observation_size: int = 6,
    num_actions: int = 3,
) -> LearnerRollout:
    """Return a random agent-major rollout: ratios and values spread past the clips."""
    generator = torch.Generator().manual_seed(0)
    return LearnerRollout(
        observations=torch.randn(agents, rows, observation_size, generator=generator),
        actions=torch.randint(num_actions, (agents, rows), generator=generator).float(),
        logprobs=torch.randn(agents, rows, generator=generator) - num_actions**0.5,
        rewards=torch.randn(agents, rows, generator=generator),
        terminals=(torch.rand(agents, rows, generator=generator) < 0.3).float(),
        values=torch.randn(agents, rows, generator=generator),
        action_mask=torch.ones(agents, rows, num_actions),
        # A feed-forward policy's empty carry: no layers and no width.
        initial_states=torch.zeros(0, agents, 0),
        branch_starts=torch.zeros(agents, dtype=torch.uint8),
    )


def _tiny_config() -> CraftaxTrainStep.Config:
    """Return exp003's learner on exp000's tiny CPU pipeline: two passes of two minibatches."""
    config = tiny_train_step()
    model = config.model = ActorCritic.Config()
    model.observation_size = OBS_SIZE
    model.channels_hidden = 8
    model.num_layers = 1
    config.optimizer = PartialConfig(ClippedAdam, lr=1e-3, eps=1e-5, max_grad_norm=1.0)
    learner = config.learner = ShuffledTransitions.Config()
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
