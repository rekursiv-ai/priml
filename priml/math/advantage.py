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
    not_done = 1.0 - dones.to(values.dtype)
    advantages = torch.empty_like(values)
    trace = torch.zeros_like(last_value)
    next_value = last_value
    for step in range(values.shape[0] - 1, -1, -1):
        residual = rewards[step] + discount * next_value * not_done[step] - values[step]
        trace = residual + discount * trace_decay * not_done[step] * trace
        advantages[step] = trace
        next_value = values[step]
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

    not_done = 1.0 - dones.to(rewards.dtype)
    greedy = q_values.max(dim=-1).values
    targets = torch.empty_like(rewards)

    carried = rewards[-1] + discount * not_done[-1] * greedy[-1]
    targets[-1] = carried
    for step in range(rewards.shape[0] - 2, -1, -1):
        bootstrap = rewards[step] + discount * not_done[step] * greedy[step + 1]
        carried = bootstrap + discount * trace_decay * not_done[step] * (
            carried - greedy[step + 1]
        )
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
