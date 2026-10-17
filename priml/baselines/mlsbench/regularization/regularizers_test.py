"""Tests for the ``dl-regularization`` regularizers, against their closed forms."""

from __future__ import annotations

from typing import Final

import math

from torch import nn
from torch.nn import functional

import pytest
import torch

from priml.baselines.mlsbench.pytorch_vision.train_step import LossContext
from priml.baselines.mlsbench.regularization.regularizers import (
    confidence_penalty,
    orthogonal,
    spatial_coactivation,
    zero,
)
from priml.optimizers.parameter_filter import everything, matching, with_ndim


CONTEXT: Final = LossContext(num_classes=4, epoch=0, total_epochs=2)
"""What the context-free regularizers are told."""


def inputs() -> dict[str, torch.Tensor]:
    """Return the batch every regularizer receives: media, logits, labels."""
    return {
        "media": torch.zeros(2, 3, 4, 5),
        "logits": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        "label": torch.tensor([0, 3]),
    }


def test_zero_adds_nothing() -> None:
    assert torch.equal(
        zero(nn.Linear(2, 3), **inputs(), context=CONTEXT),
        torch.tensor(0.0),
    )


def test_confidence_penalty_is_minus_beta_times_the_mean_entropy() -> None:
    batch = inputs()
    value = confidence_penalty(
        nn.Linear(2, 3),
        **batch,
        context=CONTEXT,
        beta=0.5,
        eps=0.0,
    )
    p = functional.softmax(batch["logits"], dim=-1)
    entropy = -(p * p.log()).sum(-1).mean()
    assert value == pytest.approx(-0.5 * entropy.item())
    uniform = confidence_penalty(
        nn.Linear(2, 3),
        **(batch | {"logits": torch.zeros(2, 4)}),
        context=CONTEXT,
        beta=1.0,
        eps=0.0,
    )
    assert uniform == pytest.approx(-math.log(4))


def test_orthogonal_is_zero_on_orthonormal_rows_and_skips_unselected() -> None:
    model = nn.Sequential(nn.Conv2d(2, 2, 1, bias=False), nn.Linear(3, 2))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
        model[1].weight.fill_(1.0)
    value = orthogonal(
        model,
        **inputs(),
        context=CONTEXT,
        strength=1.0,
        select=with_ndim(everything, 4),
    )
    assert value.item() == 0
    # A (2, 3) weight of ones has W W^T - I = [[2, 3], [3, 2]].
    linear = orthogonal(
        model,
        **inputs(),
        context=CONTEXT,
        strength=0.5,
        select=matching("1.weight"),
    )
    assert linear.item() == pytest.approx(0.5 * (2 * 2**2 + 2 * 3**2))


def test_spatial_coactivation_is_off_until_the_delay_then_ramps() -> None:
    model = nn.Sequential(nn.Conv2d(2, 3, 3), nn.Conv2d(3, 2, 1))
    with torch.no_grad():
        model[0].weight.fill_(2.0)

    def at(epoch: int) -> torch.Tensor:
        return spatial_coactivation(
            model,
            **inputs(),
            context=LossContext(num_classes=4, epoch=epoch, total_epochs=6),
            block_size=3,
            strength=1.0,
            delay=0.2,
        )

    assert at(0).item() == 0
    # A 3x3 window of ones zero-padded at the border: corner 4/9, edge 6/9,
    # centre 1, so the mean over the 3x3 grid is 4 * (4 * 4 + 4 * 6 + 9) / 81.
    energy = 4 * (4 * 4 + 4 * 6 + 9) / 81
    assert at(5).item() == pytest.approx(energy)
    assert at(3).item() == pytest.approx((0.6 - 0.2) / 0.8 * energy)


def test_spatial_coactivation_of_a_net_without_large_kernels_is_zero() -> None:
    value = spatial_coactivation(
        nn.Conv2d(2, 3, 1),
        **inputs(),
        context=LossContext(num_classes=4, epoch=1, total_epochs=1),
        block_size=3,
        strength=1.0,
        delay=0.0,
    )
    assert value.item() == 0


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
