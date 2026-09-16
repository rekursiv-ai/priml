"""Tests for AdversarialLoss."""

from __future__ import annotations

from torch import Tensor, nn

import pytest
import torch

from priml.loss.gan import AdversarialLoss
from priml.loss.weighted_loss import WeightedSum
from priml.model.cost import Compute, Cost, Flops, cost, elementwise_cost
from priml.testing.cost import assert_cost_matches_torch


def test_adversarial_loss_is_pointwise_over_batch() -> None:
    loss = AdversarialLoss.Config(adversarial_weight=1.0, content_weight=2.0).make()
    fake_media = torch.zeros(2, 3, 4)
    result = loss(
        fake_media,
        fake_logits=torch.zeros(2, 1),
        fake_media=fake_media,
        real_media=torch.ones(2, 3, 4),
    )
    # BCE of a zero logit against one is log 2; L1 against ones is 1.
    torch.testing.assert_close(
        result["loss"],
        torch.full((2,), torch.tensor(2.0).log().item() + 2.0),
    )


def test_adversarial_loss_cost_spreads_per_sample_work_over_media() -> None:
    """One logit's BCE and the two weights are per sample; L1 and its mean are per element."""
    config = AdversarialLoss.Config()
    real_media = torch.randn(2, 3, 4)
    measured = assert_cost_matches_torch(
        WeightedSum.Config(fns=[config], weights=[1.0]),
        build_input=lambda: (
            torch.randn(2, 1, requires_grad=True),
            torch.randn(2, 3, 4, requires_grad=True),
        ),
        num_tokens=12,
        run=lambda module, inputs: _loss(
            module,
            inputs[1],
            fake_logits=inputs[0],
            fake_media=inputs[1],
            real_media=real_media,
        ),
    )
    expected = elementwise_cost(
        primal=2 + (8 + 3) / 12,
        adjoint=2 + 1 + (5 + 2) / 12,
    ) + Cost(primal=Compute(flops=Flops(reduction=(12 - 1) / 12)))
    assert cost(config, num_tokens=12) == expected
    assert measured == expected + elementwise_cost(primal=2, adjoint=2)
    assert measured.params == 0
    assert measured.training.flops.matmul == 0


def test_adversarial_loss_cost_needs_num_tokens() -> None:
    with pytest.raises(TypeError, match="num_tokens"):
        cost(AdversarialLoss.Config())


def _loss(module: nn.Module, model_output: Tensor, **batch: Tensor) -> Tensor:
    """Run the wrapped adversarial loss and return its ``loss`` tensor."""
    assert isinstance(module, WeightedSum)
    return module(model_output, **batch)["loss"]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
