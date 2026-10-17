"""Tests for the ``dl-normalization`` layers, against their closed forms."""

from __future__ import annotations

from typing import Final

import pytest
import torch

from priml.baselines.mlsbench.normalization.norms import (
    BatchInstanceNorm,
    GroupNorm,
    SwitchableNorm,
)


SHAPE: Final = (2, 3, 4, 5)
"""Batch, channels, height and width of every test input."""


def features() -> torch.Tensor:
    """Return a fixed input of shape ``SHAPE``."""
    return torch.randn(SHAPE, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize(
    ("channels", "groups"),
    [(3, 3), (32, 32), (33, 11), (64, 32), (100, 25)],
)
def test_group_norm_takes_the_most_groups_up_to_32_that_divide(
    channels: int,
    groups: int,
) -> None:
    assert GroupNorm(channels).num_groups == groups


def test_a_saturated_gate_gives_batch_or_instance_norm() -> None:
    x = features()
    layer = BatchInstanceNorm(SHAPE[1])
    with torch.no_grad():
        _ = layer.rho.fill_(100.0)
        batch = layer(x)
        _ = layer.rho.fill_(-100.0)
        instance = layer(x)
    torch.testing.assert_close(
        batch,
        torch.nn.functional.batch_norm(x, None, None, training=True),
    )
    torch.testing.assert_close(instance, torch.nn.functional.instance_norm(x))


@pytest.mark.parametrize("statistic", [0, 1, 2])
def test_one_dominant_importance_picks_its_statistics(statistic: int) -> None:
    x = features()
    layer = SwitchableNorm(SHAPE[1])
    with torch.no_grad():
        for weights in (layer.mean_weight, layer.var_weight):
            _ = weights.fill_(-100.0)
            weights[statistic] = 100.0
        got = layer(x)
    expected = [
        torch.nn.functional.instance_norm(x),
        torch.nn.functional.layer_norm(x, SHAPE[1:]),
        torch.nn.functional.batch_norm(x, None, None, training=True),
    ][statistic]
    torch.testing.assert_close(got, expected)


def test_parameters_register_in_the_reference_order() -> None:
    assert [n for n, _ in BatchInstanceNorm(SHAPE[1]).named_parameters()] == [
        "weight",
        "bias",
        "rho",
    ]
    assert [n for n, _ in SwitchableNorm(SHAPE[1]).named_parameters()] == [
        "weight",
        "bias",
        "mean_weight",
        "var_weight",
    ]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
