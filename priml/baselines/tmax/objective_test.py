"""Unit tests for the source-faithful DPPO objective."""

from __future__ import annotations

import torch

from priml.baselines.tmax.objective import (
    INVALID_LOGPROB,
    binary_divergence,
    binary_kl,
    binary_total_variation,
    dppo_mask,
    dppo_token_loss,
    importance_ratio,
    mask_logprobs,
    masked_mean,
)


def test_importance_ratio_is_unclamped() -> None:
    """Keep the full probability ratio instead of applying PPO's clip."""
    ratio = importance_ratio(
        torch.tensor([-0.1, -10.0]),
        torch.tensor([-0.2, -10.5]),
    )
    torch.testing.assert_close(ratio, torch.exp(torch.tensor([0.1, 0.5])))


def test_dppo_freezes_only_a_push_outside_the_region() -> None:
    """Freeze only tokens moving too far in the requested direction."""
    behavior = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
    policy = torch.tensor([[-0.2, -2.0, -0.2, -2.0]])
    advantages = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    response_mask = torch.ones_like(policy, dtype=torch.bool)
    ratio = importance_ratio(policy, behavior)
    mask, divergence = dppo_mask(
        new_logprobs=policy,
        behavior_logprobs=behavior,
        advantages=advantages,
        ratio=ratio,
        response_mask=response_mask,
        divergence_threshold=0.1,
    )
    assert mask.tolist() == [[0.0, 1.0, 1.0, 0.0]]
    assert torch.all(divergence > 0.1)


def test_binary_kl_is_zero_for_equal_probabilities_and_masked_off_tokens() -> None:
    """Return zero for equal probabilities and excluded tokens."""
    values = torch.tensor([[-0.5, -1.0]])
    mask = torch.tensor([[True, False]])
    result = binary_divergence(
        behavior_logprobs=values,
        policy_logprobs=values,
        response_mask=mask,
        divergence_fn=binary_kl,
    )
    assert result[0, 0] == 0.0
    assert result[0, 1] == 0.0


def test_binary_divergences_match_their_closed_forms() -> None:
    """TV is ``|p - q|``; KL is the Bernoulli KL of the sampled token."""
    rollout = torch.tensor([0.5, 0.25])
    policy = torch.tensor([0.25, 0.5])
    torch.testing.assert_close(
        binary_total_variation(rollout, policy),
        torch.tensor([0.25, 0.25]),
    )
    expected = (
        rollout * (rollout / policy).log()
        + (1 - rollout) * ((1 - rollout) / (1 - policy)).log()
    )
    torch.testing.assert_close(binary_kl(rollout, policy), expected)


def test_the_divergence_is_injected_into_the_mask() -> None:
    """The mask thresholds whichever divergence the caller supplies."""
    behavior = torch.tensor([[-1.0]])
    policy = torch.tensor([[-0.2]])
    mask, divergence = dppo_mask(
        new_logprobs=policy,
        behavior_logprobs=behavior,
        advantages=torch.tensor([[1.0]]),
        ratio=importance_ratio(policy, behavior),
        response_mask=torch.tensor([[True]]),
        divergence_threshold=0.1,
        divergence_fn=_no_divergence,
    )
    assert mask.tolist() == [[1.0]]
    assert divergence.tolist() == [[0.0]]


def _no_divergence(rollout: torch.Tensor, policy: torch.Tensor) -> torch.Tensor:
    del policy
    return torch.zeros_like(rollout)


def test_mask_logprobs_invalidates_unsampled_and_nan_positions() -> None:
    """Prompt, tool, and missing values all read as the invalid marker."""
    logprobs = torch.tensor([[-0.5, -1.0, float("nan"), -2.0]])
    response_mask = torch.tensor([[True, False, True, True]])
    masked = mask_logprobs(logprobs, response_mask)
    assert masked.tolist() == [[-0.5, INVALID_LOGPROB, INVALID_LOGPROB, -2.0]]


def test_dppo_token_loss_is_the_masked_advantage_weighted_ratio() -> None:
    """Eq. 11: ``-A * ratio`` per token, zero where the mask blocks it."""
    loss = dppo_token_loss(
        advantages=torch.tensor([[2.0, -3.0, 5.0]]),
        ratio=torch.tensor([[0.5, 2.0, 4.0]]),
        policy_mask=torch.tensor([[1.0, 1.0, 0.0]]),
    )
    assert loss.tolist() == [[-1.0, 6.0, -0.0]]


def test_masked_mean_uses_the_global_denominator() -> None:
    """Divide by the update's token count rather than the local mask count."""
    values = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([True, False, True])
    assert masked_mean(values, mask, denominator=8.0).item() == 0.5


def test_masked_mean_is_zero_without_a_denominator() -> None:
    """An update with no tokens left to score contributes nothing."""
    values = torch.tensor([1.0, 2.0])
    mask = torch.tensor([True, True])
    assert masked_mean(values, mask, denominator=0.0).item() == 0.0


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
