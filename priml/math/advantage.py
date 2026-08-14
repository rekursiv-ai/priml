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
