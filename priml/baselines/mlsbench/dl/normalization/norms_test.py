"""Tests for the ``dl-normalization`` layers, against their closed forms."""

from __future__ import annotations

from typing import Final, cast, override

from configgle import Fig
from torch import nn

import pytest
import torch

from priml.baselines.mlsbench.dl.normalization.norms import (
    BatchInstanceNorm,
    GroupNorm,
    SwitchableNorm,
)
from priml.baselines.mlsbench.dl.vision.model import (
    BatchNorm,
    NormFn,
    ResNet,
)
from priml.cost import Cost, cost
from priml.testing.cost import assert_cost_matches_torch


SHAPE: Final = (2, 3, 4, 5)
"""Batch, channels, height and width of every test input."""


class Norm(nn.Module):
    """One norm from a builder, so the cost harness can build and price it."""

    class Config(Fig["Norm"]):
        """The builder and the channels it normalizes."""

        norm: NormFn = BatchNorm
        channels: int = SHAPE[1]

        def cost(self, **bus: object) -> Cost:
            """Return the builder's cost for ``bus``."""
            return cost(self.norm, **bus)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.layer = config.norm(config.channels)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.layer(x))


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


@pytest.mark.parametrize(
    "norm",
    [BatchNorm, GroupNorm, BatchInstanceNorm, SwitchableNorm],
    ids=["batch", "group", "batch_instance", "switchable"],
)
def test_every_norm_costs_what_it_owns_and_issues_no_matmul(
    norm: NormFn,
) -> None:
    analytical = assert_cost_matches_torch(
        Norm.Config(norm=norm),
        build_input=lambda: torch.randn(SHAPE, requires_grad=True),
        channels=SHAPE[1],
        seq_len=SHAPE[2] * SHAPE[3],
        batch_size=SHAPE[0],
        dtype=None,
    )
    assert analytical["flops", "primal", "matmul"].sum() == 0
    assert analytical["flops", "primal", "elementwise"].sum() > 0


@pytest.mark.parametrize(
    "norm",
    [BatchNorm, GroupNorm, BatchInstanceNorm, SwitchableNorm],
    ids=["batch", "group", "batch_instance", "switchable"],
)
def test_resnet_cost_owns_exactly_the_parameters_its_norm_slot_builds(
    norm: NormFn,
) -> None:
    config = ResNet.Config(channels_in=3, channels_out=5, blocks_per_stage=(1, 1, 1))
    config.norm = norm
    finalized = config.copy_tree().finalize()
    owned = sum(p.numel() for p in finalized.make().parameters())
    assert finalized.cost(batch_size=2, dtype=None).params == owned


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
