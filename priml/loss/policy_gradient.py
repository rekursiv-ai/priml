"""Clipped policy-gradient objective for on-policy learning.

The objective compares the probability a policy NOW assigns to an action with
the probability it assigned when the action was taken, and clips that ratio so
one update cannot move the policy arbitrarily far from the data it was fit on.
The value term is clipped the same way and for the same reason.

The functions here take flat tensors -- the caller has already decided what a
batch is -- and hold no state, so they can be checked against a hand-computed
ratio without an environment or an optimizer.

References:
    https://arxiv.org/abs/1707.06347
        Schulman et al. 2017. Proximal policy optimization algorithms.

"""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor

import torch


class ClippedPolicyLoss(NamedTuple):
    """The three terms of the objective, plus its optimization diagnostics."""

    policy: Tensor
    """Clipped policy-gradient term; the quantity being minimized."""

    value: Tensor
    """Clipped value-regression term."""

    entropy: Tensor
    """Mean policy entropy; subtracted from the total to reward exploration."""

    approx_kl: Tensor
    """Estimated divergence from the behavior policy, in nats."""

    clip_fraction: Tensor
    """Fraction of samples whose ratio left the trust region."""


def clipped_policy_loss(
    *,
    log_probs: Tensor,
    behavior_log_probs: Tensor,
    advantages: Tensor,
    values: Tensor,
    behavior_values: Tensor,
    targets: Tensor,
    entropy: Tensor,
    clip_epsilon: float,
) -> ClippedPolicyLoss:
    """Compute the clipped policy and value terms over one flat batch.

    Advantages are standardized across the batch, which fixes the gradient
    scale so the learning rate does not have to absorb the reward magnitude.
    The standardization is centered and scaled by the ORIGINAL advantages, not
    the centered ones, matching the reference implementation.

    Args:
      log_probs: Log-probability of each taken action under the current policy.
      behavior_log_probs: The same quantity recorded during the rollout.
      advantages: Advantage estimate per sample.
      values: Current critic estimate per sample.
      behavior_values: Critic estimate recorded during the rollout.
      targets: Value-regression target per sample.
      entropy: Policy entropy per sample.
      clip_epsilon: Half-width of the trust region, for both ratio and value.

    Returns:
      terms: The policy, value, and entropy terms with their diagnostics.

    """
    log_ratio = log_probs - behavior_log_probs
    ratio = log_ratio.exp()
    # Standardize with the raw spread: an all-equal advantage vector has zero
    # spread, and the floor is what keeps that case finite rather than NaN.
    normalized = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + torch.finfo(advantages.dtype).eps
    )
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy = -torch.minimum(ratio * normalized, clipped_ratio * normalized).mean()

    clipped_values = behavior_values + (values - behavior_values).clamp(
        -clip_epsilon,
        clip_epsilon,
    )
    value = (
        0.5
        * torch.maximum(
            (values - targets) ** 2,
            (clipped_values - targets) ** 2,
        ).mean()
    )

    return ClippedPolicyLoss(
        policy=policy,
        value=value,
        entropy=entropy.mean(),
        # The k3 estimator: non-negative and lower variance than -log_ratio,
        # which is what makes it readable as a trust-region alarm.
        approx_kl=((ratio - 1.0) - log_ratio).mean(),
        clip_fraction=((ratio - 1.0).abs() > clip_epsilon).to(ratio.dtype).mean(),
    )


def categorical_entropy(log_probs: Tensor) -> Tensor:
    """Return the entropy of a categorical distribution given its log-probs.

    Computed as ``-sum(p * log p)`` over the last axis. A masked-out action
    carries ``log p = -inf`` and ``p = 0``, whose product is NaN rather than
    the zero the limit gives, so those terms are dropped explicitly.

    Args:
      log_probs: Normalized log-probabilities, ``[..., actions]``.

    Returns:
      entropy: Entropy in nats, shape ``[...]``.

    """
    terms = log_probs.exp() * log_probs
    return -torch.where(terms.isfinite(), terms, torch.zeros_like(terms)).sum(-1)
