"""Tests for the functions that switch a recipe to another setting."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from configgle import Makes
from torch import nn

import pytest

from priml.baselines.mlsbench.dl.vision import settings
from priml.baselines.mlsbench.dl.vision.data import Cifar100Data
from priml.baselines.mlsbench.dl.vision.experiments import exp000
from priml.baselines.mlsbench.dl.vision.experiments_test import changed_fields
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


@pytest.mark.parametrize(
    ("switch", "dataset"),
    [
        (settings.vgg16bn_cifar100, "cifar100"),
        (settings.mobilenetv2_fmnist, "fmnist"),
    ],
)
def test_a_setting_changes_only_the_network_and_the_dataset(
    switch: Callable[[VisionTrainLoop], VisionTrainLoop],
    dataset: str,
) -> None:
    changed = changed_fields(exp000(), switch(exp000()))
    assert changed
    assert all(path.startswith(("step.model", "dataset.")) for path in changed)
    assert switch(exp000()).dataset.working_dir == f"/datasets/{dataset}"


def test_vgg16bn_cifar100_classifies_cifar100() -> None:
    cfg = settings.vgg16bn_cifar100(exp000())
    assert isinstance(cfg.step.model, VGG16BN.Config)
    assert (cfg.step.model.channels_in, cfg.step.model.channels_out) == (3, 100)
    assert cfg.dataset.mean == exp000().dataset.mean


def test_mobilenetv2_fmnist_classifies_fashion_mnist_with_its_statistics() -> None:
    cfg = settings.mobilenetv2_fmnist(exp000())
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    assert (cfg.step.model.channels_in, cfg.step.model.channels_out) == (3, 10)
    assert cfg.dataset.mean == (0.2860,) * 3
    assert cfg.dataset.std == (0.3530,) * 3


def test_resnet110_cifar100_deepens_the_resnet_and_keeps_its_slots() -> None:
    cfg = exp000()
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.norm = nn.InstanceNorm2d
    assert changed_fields(exp000(), settings.resnet110_cifar100(exp000())) == {
        f"step.model.blocks_per_stage[{stage}]" for stage in range(3)
    }
    switched = settings.resnet110_cifar100(cfg)
    assert isinstance(switched.step.model, ResNet.Config)
    assert switched.step.model.blocks_per_stage == (18, 18, 18)
    assert switched.step.model.norm is nn.InstanceNorm2d


def test_resnet20_cifar10_shortens_the_resnet_and_reads_cifar10() -> None:
    cfg = exp000()
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.norm = nn.InstanceNorm2d
    switched = settings.resnet20_cifar10(cfg)
    assert isinstance(switched.step.model, ResNet.Config)
    assert switched.step.model.blocks_per_stage == (3, 3, 3)
    assert switched.step.model.channels_out == 10
    assert switched.step.model.norm is nn.InstanceNorm2d
    assert switched.dataset.working_dir == "/datasets/cifar10"
    assert switched.dataset.mean == (0.4914, 0.4822, 0.4465)
    assert switched.dataset.std == (0.2023, 0.1994, 0.2010)


@pytest.mark.parametrize(
    "switch",
    [settings.resnet110_cifar100, settings.resnet20_cifar10],
    ids=["resnet110_cifar100", "resnet20_cifar10"],
)
def test_a_resnet_setting_refuses_a_network_that_is_no_resnet(
    switch: Callable[[VisionTrainLoop], VisionTrainLoop],
) -> None:
    with pytest.raises(TypeError, match="ResNet"):
        switch(settings.vgg16bn_cifar100(exp000()))


def test_a_setting_returns_the_config_it_was_given() -> None:
    cfg = exp000()
    assert settings.vgg16bn_cifar100(cfg) is cfg


class _OtherData(Makes["Cifar100Data"], Cifar100Data.Config):
    """A dataset config deriving from CIFAR-100's, as a task's own may."""


class _OtherLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[VisionTrainStep.Config, _OtherData],
):
    """A loop that is no ``VisionTrainLoop`` but holds the vision step."""

    step: VisionTrainStep.Config = field(default_factory=VisionTrainStep.Config)
    dataset: _OtherData = field(default_factory=_OtherData)


def test_a_setting_switches_any_loop_holding_the_vision_step() -> None:
    cfg = settings.mobilenetv2_fmnist(_OtherLoop())
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    assert cfg.dataset.working_dir == "/datasets/fmnist"


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
