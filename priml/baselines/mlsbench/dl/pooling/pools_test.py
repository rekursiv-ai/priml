"""Tests for the ``cv-pooling-aggregation`` pools, against their closed forms."""

from __future__ import annotations

from typing import Final

from configgle import Makeable

import pytest
import torch

from priml.baselines.mlsbench.dl.pooling.pools import (
    AvgMaxPool,
    GeneralizedMeanPool,
    GlobalMaxPool,
)
from priml.baselines.mlsbench.dl.vision.model import GlobalAvgPool, ResNet
from priml.testing.cost import assert_cost_matches_torch


SHAPE: Final = (2, 3, 4, 5)


def features() -> torch.Tensor:
    """Return distinct positive and negative features of shape ``SHAPE``."""
    count = SHAPE[0] * SHAPE[1] * SHAPE[2] * SHAPE[3]
    return ((torch.arange(count) * 7 % 23 - 11) / 8).reshape(SHAPE)


def test_global_max_takes_each_channels_peak() -> None:
    x = features()
    assert torch.equal(GlobalMaxPool.Config().make()(x), x.amax(dim=(-2, -1)))


def test_avg_max_is_the_mean_of_both_pools() -> None:
    x = features()
    expected = (x.mean(dim=(-2, -1)) + x.amax(dim=(-2, -1))) / 2
    torch.testing.assert_close(AvgMaxPool.Config().make()(x), expected)


def test_gem_starts_at_p_3_and_floors_activations_at_eps() -> None:
    pool = GeneralizedMeanPool.Config().make()
    assert pool.p.tolist() == [3.0]
    x = features().double()
    pool = pool.double()
    expected = x.clamp(min=1e-6).pow(3).mean(dim=(-2, -1)).pow(1 / 3)
    torch.testing.assert_close(pool(x), expected)


def test_gem_with_p_below_the_floor_is_the_mean_of_clamped_features() -> None:
    config = GeneralizedMeanPool.Config()
    config.p = 0.5
    x = features().double()
    pool = config.make().double()
    torch.testing.assert_close(pool(x), x.clamp(min=1e-6).mean(dim=(-2, -1)))


def test_gem_learns_its_exponent_and_only_above_the_floor() -> None:
    pool = GeneralizedMeanPool.Config().make()
    pool(features()).sum().backward()
    assert pool.p.grad is not None
    assert pool.p.grad.item() != 0
    config = GeneralizedMeanPool.Config()
    config.p = 0.5
    floored = config.make()
    floored(features()).sum().backward()
    assert floored.p.grad is not None
    assert floored.p.grad.tolist() == [0.0]


@pytest.mark.parametrize("config", [GlobalMaxPool.Config(), AvgMaxPool.Config()])
def test_parameter_free_pools_hold_no_parameters(
    config: GlobalMaxPool.Config | AvgMaxPool.Config,
) -> None:
    assert list(config.make().parameters()) == []


@pytest.mark.parametrize(
    ("config", "params"),
    [
        (GlobalMaxPool.Config(), 0),
        (GeneralizedMeanPool.Config(), 1),
        (AvgMaxPool.Config(), 0),
        (GlobalAvgPool.Config(), 0),
    ],
    ids=["max", "gem", "avg_max", "avg"],
)
def test_every_pool_costs_what_it_owns_and_issues_no_matmul(
    config: Makeable[torch.nn.Module],
    params: int,
) -> None:
    channels, height, width = SHAPE[1:]
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.rand(SHAPE, requires_grad=True),
        channels=channels,
        seq_len=height * width,
        batch_size=SHAPE[0],
        dtype=None,
    )
    assert analytical.params == params
    assert analytical["flops", "primal", "matmul"].sum() == 0


def test_resnet_cost_follows_its_pool_slot() -> None:
    model = ResNet.Config(channels_in=3, channels_out=5, blocks_per_stage=(1, 1, 1))
    baseline = model.cost(batch_size=2, dtype=None)
    model.pool = GeneralizedMeanPool.Config()
    assert model.cost(batch_size=2, dtype=None).params == baseline.params + 1


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
