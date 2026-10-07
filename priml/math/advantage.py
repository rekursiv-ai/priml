"""Advantage estimators for policy-gradient learning.

A rollout gives rewards and value estimates at each step; an advantage says how
much better a step turned out than the critic expected. These functions turn
one into the other. They are pure: tensors in, tensors out, no configuration
and no state, so a caller can test them against a hand-computed recursion
without building a model or an environment.
"""

from __future__ import annotations

from torch import Tensor

import torch


def generalized_advantage(
    *,
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    last_value: Tensor,
    discount: float,
    trace_decay: float,
) -> tuple[Tensor, Tensor]:
    """Estimate advantages and value targets by the GAE recursion.

    Walks the rollout backwards accumulating the exponentially-weighted sum of
    temporal-difference residuals. ``trace_decay`` interpolates between the
    one-step residual (0, low variance and high bias) and the full Monte-Carlo
    return (1, the reverse). A terminal step zeroes both the bootstrap and the
    carried trace, so credit never crosses an episode boundary.

    Args:
      rewards: Per-transition rewards, shape ``[time, envs]``.
      values: Critic estimates at each pre-step observation, ``[time, envs]``.
      dones: Terminal flags for each transition, ``[time, envs]``; any dtype
        that compares as 0/1.
      last_value: Critic estimate after the final transition, ``[envs]``.
      discount: Reward discount factor, usually written gamma.
      trace_decay: Eligibility-trace decay, usually written lambda.

    Returns:
      advantages: Advantage estimates, shape ``[time, envs]``.
      targets: Value-regression targets, ``advantages + values``.

    References:
      https://arxiv.org/abs/1506.02438
        Schulman et al. 2015. High-dimensional continuous control using
        generalized advantage estimation.

    """
    # Selected, not multiplied by a 0/1 mask: 0 * nan is nan, so a value from
    # past a terminal would otherwise leak into the episode that ended.
    done = dones.bool()
    advantages = torch.empty_like(values)
    trace = torch.zeros_like(last_value)
    next_value = last_value
    for step in range(values.shape[0] - 1, -1, -1):
        bootstrap = torch.where(done[step], 0.0, discount * next_value)
        residual = rewards[step] + bootstrap - values[step]
        trace = residual + torch.where(done[step], 0.0, discount * trace_decay * trace)
        advantages[step] = trace
        next_value = values[step]
    return advantages, advantages + values


def observation_aligned_advantage(
    *,
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    discount: float,
    trace_decay: float,
) -> tuple[Tensor, Tensor]:
    """Estimate advantages by GAE when each reward is stored with its observation.

    :func:`generalized_advantage` stores beside each observation the reward and
    terminal of the transition that LEAVES it. Here step ``t`` holds the ones
    that ARRIVED with observation ``t``, as an actor stores them when it writes
    a step's reward beside the observation that step produced. Step ``t``'s
    residual therefore reads step ``t + 1``, and the last step has no next
    step: its advantage is zero and its target is its own value. On the other
    steps this is :func:`generalized_advantage` of the rewards and terminals
    shifted back by one, bootstrapped from the last value, which is how it is
    computed.

    The recursion runs in the tensors' dtype, so the caller chooses the
    precision. Time is the LAST axis. The decay ``discount * trace_decay``
    rounds once, from float64, where a kernel that multiplies two fp32
    arguments (``TritonPPO``'s walk) rounds each first: the two decays differ
    by one ulp at 30% of random pairs, though not at PufferLib's Craftax pair
    or at the policy-gradient rule's defaults.

    Args:
      rewards: The reward that arrived with each observation, ``[..., time]``.
      values: Critic estimates at each observation, ``[..., time]``.
      dones: Whether the transition into each observation ended an episode,
        ``[..., time]``; any dtype that compares as 0/1.
      discount: Reward discount factor, usually written gamma.
      trace_decay: Eligibility-trace decay, usually written lambda.

    Returns:
      advantages: Advantage estimates, ``[..., time]``; zero at the last step.
      targets: Value-regression targets, ``advantages + values``.

    References:
      https://arxiv.org/abs/1506.02438
        Schulman et al. 2015. High-dimensional continuous control using
        generalized advantage estimation.
      https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``puff_advantage`` in
        ``src/algo.cu``, pin ``6ffa5b10``.

    """
    advantages, _ = generalized_advantage(
        rewards=rewards[..., 1:].movedim(-1, 0),
        values=values[..., :-1].movedim(-1, 0),
        dones=dones[..., 1:].movedim(-1, 0),
        last_value=values[..., -1],
        discount=discount,
        trace_decay=trace_decay,
    )
    advantages = torch.cat(
        [advantages.movedim(0, -1), torch.zeros_like(values[..., -1:])],
        dim=-1,
    )
    return advantages, advantages + values


def q_lambda_targets(
    *,
    rewards: Tensor,
    q_values: Tensor,
    dones: Tensor,
    discount: float,
    trace_decay: float,
) -> Tensor:
    """Build multi-step regression targets from a policy's own Q-values.

    The value-based counterpart to :func:`generalized_advantage`, and the same
    idea: walk the rollout backwards mixing a one-step bootstrap with the
    return that follows it, so ``trace_decay`` trades bias for variance. What
    differs is where the bootstrap comes from -- the greedy action's Q-value
    rather than a separate critic, which is what lets a Q-learner train
    without one.

    A terminal step takes its reward alone. Not merely a zeroed bootstrap:
    there is no next state to be greedy in, so anything carried across the
    boundary would be a value from a world that ended.

    Args:
      rewards: Per-transition rewards, ``[time, envs]``.
      q_values: Q-values at each state INCLUDING the bootstrap state after the
        last transition, ``[time + 1, envs, actions]``.
      dones: Terminal flags per transition, ``[time, envs]``.
      discount: Reward discount factor, usually written gamma.
      trace_decay: Multi-step mixing factor, usually written lambda.

    Returns:
      targets: Regression targets, ``[time, envs]``.

    Raises:
      ValueError: The sequence is empty, or the Q-values do not carry exactly
        one more step than the rewards.

    References:
      https://arxiv.org/abs/2407.04811
        Gallici et al. 2024. Simplifying deep temporal difference learning.

    """
    if rewards.shape[0] == 0:
        raise ValueError("Q(lambda) sequence must be non-empty")
    if q_values.shape[0] != rewards.shape[0] + 1:
        raise ValueError("Q(lambda) requires one more Q-value step than rewards")

    greedy = q_values.max(dim=-1).values
    targets = torch.empty_like(rewards)

    carried = torch.where(
        dones[-1].bool(),
        rewards[-1],
        rewards[-1] + discount * greedy[-1],
    )
    targets[-1] = carried
    for step in range(rewards.shape[0] - 2, -1, -1):
        bootstrap = rewards[step] + discount * greedy[step + 1]
        carried = bootstrap + discount * trace_decay * (carried - greedy[step + 1])
        targets[step] = torch.where(dones[step].bool(), rewards[step], carried)
        carried = targets[step]
    return targets


def explained_variance(predictions: Tensor, targets: Tensor) -> Tensor:
    """Measure the fraction of target variance the predictions account for.

    One is a perfect fit, zero is no better than predicting the mean, and a
    negative value is worse than that. Constant targets have no variance to
    explain, so they score zero rather than dividing by it.

    Args:
      predictions: Value predictions.
      targets: Corresponding regression targets.

    Returns:
      fraction: Explained variance, or zero when the targets are constant.

    """
    variance = targets.var(unbiased=False)
    fraction = 1.0 - (targets - predictions).var(unbiased=False) / variance.clamp_min(
        torch.finfo(targets.dtype).eps,
    )
    return torch.where(variance > 0.0, fraction, torch.zeros_like(fraction))
