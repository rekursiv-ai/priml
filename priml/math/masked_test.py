"""Pin masked values, weighted denominators, and full gradients."""

import math

import pytest
import torch

from priml.math.masked import (
    masked_cdf_error,
    masked_expectation,
    masked_huber,
    masked_mean,
    masked_soft_cross_entropy,
)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_masked_mean_separates_weight_normalization_from_scale(
    weighted: bool,
    scaled: bool,
) -> None:
    value = torch.arange(6).reshape(2, 3).float().requires_grad_()
    mask = torch.tensor([[True, False, True], [False, True, True]])
    weight = torch.tensor([[1.0, 2, 3], [4, 5, 7]]) if weighted else None
    scale = torch.tensor([[6.0, 5, 4], [3, 2, 1]]) if scaled else None
    expected_weight = torch.where(
        mask,
        torch.ones_like(value) if weight is None else weight,
        0,
    )
    denominator = expected_weight.sum()
    gradient = expected_weight / denominator
    if scale is not None:
        gradient = gradient * scale
    actual = masked_mean(value, mask=mask, weight=weight, scale=scale)
    assert torch.equal(actual, (value[mask] * gradient[mask]).sum())
    actual.backward()
    assert value.grad is not None
    assert torch.equal(value.grad, gradient)


@pytest.mark.parametrize("empty_mask", [False, True])
def test_empty_or_nonpositive_weight_gives_connected_zero(empty_mask: bool) -> None:
    value = torch.arange(6).reshape(2, 3).float().requires_grad_()
    mask = torch.full((2, 3), not empty_mask)
    actual = masked_mean(value, mask=mask, weight=torch.zeros_like(value))
    assert actual.item() == 0
    actual.backward()
    assert value.grad is not None
    assert torch.equal(value.grad, torch.zeros_like(value))


def test_huber_uses_quadratic_and_linear_regions_with_masked_gradients() -> None:
    value = torch.tensor([[-2.0, -0.5, 0], [0.5, 2, 4]], requires_grad=True)
    target = torch.zeros_like(value)
    mask = torch.tensor([[True, True, False], [True, True, False]])
    actual = masked_huber(value, target, mask=mask, delta=1.0)
    assert actual.item() == (1.5 + 0.125 + 0.125 + 1.5) / 4
    actual.backward()
    assert value.grad is not None
    assert torch.equal(value.grad, torch.tensor([[-1.0, -0.5, 0], [0.5, 1, 0]]) / 4)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_huber_preserves_weighted_and_empty_reductions(
    empty: bool,
    scaled: bool,
) -> None:
    value = torch.tensor([[-2.0, -0.5, 0], [0.5, 2, 4]], requires_grad=True)
    mask = (
        torch.zeros_like(value, dtype=torch.bool)
        if empty
        else torch.tensor(
            [[True, True, False], [True, True, False]],
        )
    )
    weight = torch.tensor([[1.0, 2, 4], [3, 2, 5]])
    scale = torch.full_like(value, 2) if scaled else None
    actual = masked_huber(
        value,
        torch.zeros_like(value),
        mask=mask,
        delta=1.0,
        weight=weight,
        scale=scale,
    )
    multiplier = 2 if scaled else 1
    assert actual.item() == (0 if empty else 5.125 / 8 * multiplier)
    actual.backward()
    assert value.grad is not None
    expected = (
        torch.zeros_like(value)
        if empty
        else torch.tensor(
            [[-1.0, -1, 0], [1.5, 2, 0]],
        )
        / 8
        * multiplier
    )
    assert torch.equal(value.grad, expected)


def test_huber_empty_unweighted_mask_keeps_all_input_gradients_zero() -> None:
    value = torch.arange(6).reshape(2, 3).float().requires_grad_()
    actual = masked_huber(
        value,
        torch.zeros_like(value),
        mask=torch.zeros_like(value, dtype=torch.bool),
        delta=1.0,
    )
    assert actual.item() == 0
    actual.backward()
    assert value.grad is not None
    assert torch.equal(value.grad, torch.zeros_like(value))


def test_soft_cross_entropy_and_gradient_keep_all_dense_mass() -> None:
    logits = torch.zeros(2, 3, 4, requires_grad=True)
    target = torch.tensor([0.125, 0.25, 0.5, 0.125]).expand_as(logits)
    mask = torch.tensor([[True, False, False], [False, True, False]])
    actual = masked_soft_cross_entropy(logits, target, mask=mask)
    assert actual.item() == torch.tensor(math.log(4)).item()
    actual.backward()
    assert logits.grad is not None
    expected = torch.where(mask[..., None], (1 / 4 - target) / 2, 0)
    assert torch.equal(logits.grad, expected)


def test_cdf_score_observes_order_and_expectation_keeps_value_axis() -> None:
    logits = torch.zeros(2, 3, 4, requires_grad=True)
    target = torch.zeros_like(logits)
    target[..., 0] = 1
    mask = torch.tensor([[True, False, False], [False, True, False]])
    cdf = masked_cdf_error(logits, target, mask=mask)
    assert cdf.item() == (0.75**2 + 0.5**2 + 0.25**2) / 4
    values = torch.arange(24).reshape(2, 3, 4).float()
    expectation = masked_expectation(logits, values, mask=mask)
    assert expectation.item() == (1.5 + 17.5) / 2
    expectation.backward()
    assert logits.grad is not None
    expected = torch.where(
        mask[..., None],
        (values - values.mean(-1, keepdim=True)) / 8,
        0,
    )
    assert torch.equal(logits.grad, expected)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_all_posterior_losses_forward_both_weight_roles(
    weighted: bool,
    scaled: bool,
) -> None:
    logits = torch.arange(24).reshape(2, 3, 4).float() / 16
    target = torch.arange(24, 0, -1).reshape(2, 3, 4).float()
    target = target / target.sum(-1, keepdim=True)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    weight = torch.arange(1, 7).reshape(2, 3).float() if weighted else None
    scale = torch.arange(6, 0, -1).reshape(2, 3).float() if scaled else None
    softmax = logits.softmax(-1)
    cross_entropy = -(target * logits.log_softmax(-1)).sum(-1)
    cdf = (softmax.cumsum(-1) - target.cumsum(-1)).square().mean(-1)
    expectation = (softmax * target).sum(-1)
    for actual, values in (
        (
            masked_soft_cross_entropy(
                logits,
                target,
                mask=mask,
                weight=weight,
                scale=scale,
            ),
            cross_entropy,
        ),
        (masked_cdf_error(logits, target, mask=mask, weight=weight, scale=scale), cdf),
        (
            masked_expectation(logits, target, mask=mask, weight=weight, scale=scale),
            expectation,
        ),
    ):
        assert torch.equal(
            actual,
            masked_mean(values, mask=mask, weight=weight, scale=scale),
        )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
