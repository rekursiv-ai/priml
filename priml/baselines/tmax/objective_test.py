"""Unit tests for the source-faithful DPPO objective."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from priml.baselines.tmax.objective import (
    DivergenceType,
    binary_divergence,
    dppo_mask,
    importance_ratio,
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
        divergence_type="kl",
    )
    assert result[0, 0] == 0.0
    assert result[0, 1] == 0.0


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


def test_unknown_divergence_is_rejected() -> None:
    """Reject divergence choices that TMax does not support."""

    def invalid_divergence() -> str:
        """Return a value outside the supported divergence choices."""
        return "not-a-divergence"

    with pytest.raises(ValueError, match="Unknown DPPO"):
        binary_divergence(
            behavior_logprobs=torch.zeros(1),
            policy_logprobs=torch.zeros(1),
            response_mask=torch.ones(1, dtype=torch.bool),
            divergence_type=cast(DivergenceType, invalid_divergence()),
        )
