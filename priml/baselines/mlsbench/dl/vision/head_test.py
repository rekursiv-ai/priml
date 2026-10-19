"""Tests for the classifier heads each vision network config builds on request."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Final, cast, override

from configgle import Fig
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import Activation, ResNet
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from priml.cost import Cost

BATCH: Final = 2
WIDTH: Final = 5
CLASSES: Final = 7
"""Logits of the network's own head."""

OTHER: Final = 3
"""Logits of a second head built from the same config."""

type NetworkConfig = ResNet.Config | VGG16BN.Config | MobileNetV2.Config


def configs() -> list[NetworkConfig]:
    """Return one config of each network, its features ``WIDTH`` wide."""
    resnet = ResNet.Config(channels_in=3, channels_out=CLASSES)
    resnet.channels_hidden = (4, WIDTH)
    resnet.blocks_per_stage = (1, 1)
    vgg = VGG16BN.Config(channels_in=3, channels_out=CLASSES)
    vgg.channels_hidden = ((4,), (WIDTH,))
    vgg.channels_hidden_head = 6
    mobilenet = MobileNetV2.Config(channels_in=3, channels_out=CLASSES)
    mobilenet.channels_stem = 4
    mobilenet.channels_hidden = (4,)
    mobilenet.blocks_per_stage = (1,)
    mobilenet.expand_ratios = (1,)
    mobilenet.strides = (1,)
    mobilenet.channels_last = WIDTH
    return [resnet, vgg, mobilenet]


IDS: Final = ["resnet", "vgg16bn", "mobilenetv2"]


class HeadOnly(nn.Module):
    """A head of width ``OTHER`` built from a network config, alone."""

    class Config(Fig["HeadOnly"]):
        """The network whose head is built."""

        network: NetworkConfig = field(default_factory=ResNet.Config)
        """Builds and costs the head."""

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the head alone."""
            del kwargs
            return self.network.head_cost(
                channels_out=OTHER,
                batch_size=batch_size,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.head = config.network.head(OTHER)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


@pytest.mark.parametrize("config", configs(), ids=IDS)
def test_a_head_of_another_width_has_the_networks_own_layout(
    config: NetworkConfig,
) -> None:
    network = config.make()
    assert isinstance(network, (ResNet, VGG16BN, MobileNetV2))
    own: nn.Module = network.head
    other: nn.Module = config.head(OTHER)
    assert [type(m) for m in other.modules()] == [type(m) for m in own.modules()]
    linears = [m for m in other.modules() if isinstance(m, nn.Linear)]
    assert linears[0].in_features == WIDTH
    assert linears[-1].out_features == OTHER


def test_the_vgg_head_is_the_whole_classifier_with_the_configs_activation() -> None:
    config = configs()[1]
    assert isinstance(config, VGG16BN.Config)
    config.activation = torch.sigmoid
    config.dropout = 0.25
    linear, activation, dropout, last = config.head(OTHER)
    assert isinstance(activation, Activation)
    assert activation.fn is torch.sigmoid
    assert isinstance(dropout, nn.Dropout)
    assert dropout.p == 0.25
    assert cast(nn.Linear, linear).out_features == 6
    assert cast(nn.Linear, last).in_features == 6


@pytest.mark.parametrize("config", configs(), ids=IDS)
def test_a_heads_cost_counts_every_matmul_it_issues(config: NetworkConfig) -> None:
    carrier = HeadOnly.Config()
    carrier.network = config
    analytical = assert_cost_matches_torch(
        carrier,
        build_input=lambda: torch.randn(BATCH, WIDTH, requires_grad=True),
        batch_size=BATCH,
        dtype=None,
    )
    assert analytical["flops", "primal", "matmul"].sum() > 0


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
