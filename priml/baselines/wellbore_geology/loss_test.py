"""Pin every active public objective term and left-to-right weighted addition."""

from typing import TYPE_CHECKING, cast

import math

import pytest
import torch

from priml.baselines.wellbore_geology.loss import (
    PosteriorLoss,
    WellboreLoss,
    masked_expectation,
    masked_huber,
    masked_soft_cross_entropy,
)
from priml.baselines.wellbore_geology.testing import (
    numerical_tensors,
    tiny_batches,
)
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from priml.baselines.wellbore_geology.custom_types import Batch


def test_public_objective_preserves_complete_terms_and_scalar_order() -> None:
    prediction = torch.arange(6).reshape(2, 3).float().requires_grad_()
    target = torch.zeros_like(prediction)
    logits = torch.arange(24).reshape(2, 3, 4).float() / 17
    probability = torch.arange(24, 0, -1).reshape(2, 3, 4).float()
    probability = probability / probability.sum(-1, keepdim=True)
    gamma = torch.arange(24).reshape(2, 3, 4).float() / 19
    mask = torch.tensor([[True, False, True], [False, True, True]])
    scale = torch.arange(1, 7).reshape(2, 3).float()
    config = PosteriorLoss.Config()
    config.regression_weight = 0.7
    config.alignment_weight = 0.2
    config.gamma_ray_weight = 0.1
    actual, details = config.make()(
        prediction,
        logits=logits,
        target=target,
        probabilities=probability,
        gamma_error=gamma,
        mask=mask,
        scale=scale,
    )
    expected = {
        "regression": masked_huber(
            prediction,
            target,
            mask=mask,
            delta=1.0,
            scale=scale,
        ),
        "alignment": masked_soft_cross_entropy(
            logits,
            probability,
            mask=mask,
            scale=scale,
        ),
        "GR_penalty": masked_expectation(logits, gamma, mask=mask, scale=scale),
    }
    assert list(details) == list(expected)
    for name, value in details.items():
        assert torch.equal(value, expected[name])
    assert torch.equal(
        actual,
        0.7 * expected["regression"]
        + 0.2 * expected["alignment"]
        + 0.1 * expected["GR_penalty"],
    )
    actual.backward()
    assert prediction.grad is not None
    assert torch.equal(
        prediction.grad,
        torch.tensor([[0.0, 0, 0.7 * 3 / 4], [0, 0.7 * 5 / 4, 0.7 * 6 / 4]]),
    )


def test_loss_slot_retains_every_ordered_posterior_objective_term() -> None:
    batch = cast("Batch", tiny_batches()[0])
    prediction = batch["label"] * 0.5
    logits = (
        torch.arange(batch["typewell_target_probs"].numel())
        .reshape(
            batch["typewell_target_probs"].shape,
        )
        .float()
        / 71
    )
    config = WellboreLoss.Config()
    loss, details = config.objective.make()(
        prediction,
        logits=logits,
        target=batch["label"],
        probabilities=batch["typewell_target_probs"],
        gamma_error=batch["aux_targets"]["GR_penalty_error"],
        mask=batch["target_mask"],
    )
    actual = config.make()((prediction, {"alignment_logits": logits}), **batch)
    assert (
        mismatches(
            numerical_tensors({"loss": loss.expand(len(batch["meta"])), **details}),
            numerical_tensors(actual),
            ulps=True,
        )
        == []
    )


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
def test_huber_scale_multiplies_the_selected_mean(empty: bool) -> None:
    value = torch.tensor([[-2.0, -0.5, 0], [0.5, 2, 4]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, False]])
    if empty:
        mask = torch.zeros_like(mask)
    actual = masked_huber(
        value,
        torch.zeros_like(value),
        mask=mask,
        delta=1.0,
        scale=torch.full_like(value, 2),
    )
    assert actual.item() == (0 if empty else 3.25 / 4 * 2)
    actual.backward()
    assert value.grad is not None
    expected = torch.tensor([[-1.0, -0.5, 0], [0.5, 1, 0]]) / 4 * 2
    assert torch.equal(value.grad, torch.zeros_like(value) if empty else expected)


def test_huber_empty_unscaled_mask_keeps_all_input_gradients_zero() -> None:
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


def test_expectation_keeps_the_value_axis_and_scale_weights_selected_rows() -> None:
    logits = torch.zeros(2, 3, 4, requires_grad=True)
    values = torch.arange(24).reshape(2, 3, 4).float()
    mask = torch.tensor([[True, False, False], [False, True, False]])
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
    scale = torch.tensor([[2.0, 3, 5], [7, 11, 13]])
    scaled = masked_expectation(logits, values, mask=mask, scale=scale)
    assert scaled.item() == (1.5 * 2 + 17.5 * 11) / 2


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
