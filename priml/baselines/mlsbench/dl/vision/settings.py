"""Switch a vision recipe to another of the reference's ``--arch``/``--dataset`` settings.

Each function changes the network and the dataset of a config and nothing
else, so a task's fork for a setting is one call followed by its own slot::

    cfg = settings.vgg16bn_cifar100(exp000())
    assert isinstance(cfg.step.model, VGG16BN.Config)
    cfg.step.model.norm = GroupNorm

A setting with another architecture builds a fresh network config at its
defaults, so a slot the task set on the recipe's ResNet must be set again on
it; ``resnet110_cifar100`` only deepens the ResNet and keeps its slots.
"""

from __future__ import annotations

from priml.baselines.mlsbench.dl.vision.data import Cifar100Data
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.train.train_loop import TrainLoop


def vgg16bn_cifar100[C: TrainLoop.Config[VisionTrainStep.Config, Cifar100Data.Config]](
    cfg: C,
) -> C:
    """Train VGG-16-BN on CIFAR-100: the reference's ``vgg16bn-cifar100`` setting.

    Args:
      cfg: The recipe, changed in place.

    Returns:
      cfg: The same config.

    """
    model = cfg.step.model = VGG16BN.Config()
    model.channels_in = 3
    model.channels_out = 100
    cfg.dataset.working_dir = "/datasets/cifar100"
    cfg.dataset.mean = (0.5071, 0.4867, 0.4408)
    cfg.dataset.std = (0.2675, 0.2565, 0.2761)
    return cfg


def resnet110_cifar100[
    C: TrainLoop.Config[VisionTrainStep.Config, Cifar100Data.Config],
](cfg: C) -> C:
    """Deepen the recipe's ResNet to 110 layers: the reference's ``resnet110-cifar100``.

    Unlike the other settings this keeps the network's config, so a slot the
    task set on it survives.

    Args:
      cfg: A CIFAR-100 recipe whose network is a ResNet, changed in place.

    Returns:
      cfg: The same config.

    Raises:
      TypeError: The network is no ResNet.

    """
    if not isinstance(cfg.step.model, ResNet.Config):
        raise TypeError(
            f"resnet110_cifar100 deepens a ResNet; got {type(cfg.step.model).__name__}.",
        )
    cfg.step.model.blocks_per_stage = (18, 18, 18)
    return cfg


def resnet20_cifar10[
    C: TrainLoop.Config[VisionTrainStep.Config, Cifar100Data.Config],
](cfg: C) -> C:
    """Train ResNet-20 on CIFAR-10: the reference's ``resnet20-cifar10`` setting.

    Like ``resnet110_cifar100`` this keeps the network's config, so a slot the
    task set on it survives.

    Args:
      cfg: A recipe whose network is a ResNet, changed in place.

    Returns:
      cfg: The same config.

    Raises:
      TypeError: The network is no ResNet.

    """
    if not isinstance(cfg.step.model, ResNet.Config):
        raise TypeError(
            f"resnet20_cifar10 shortens a ResNet; got {type(cfg.step.model).__name__}.",
        )
    cfg.step.model.blocks_per_stage = (3, 3, 3)
    cfg.step.model.channels_out = 10
    cfg.dataset.working_dir = "/datasets/cifar10"
    cfg.dataset.mean = (0.4914, 0.4822, 0.4465)
    cfg.dataset.std = (0.2023, 0.1994, 0.2010)
    return cfg


def mobilenetv2_fmnist[
    C: TrainLoop.Config[VisionTrainStep.Config, Cifar100Data.Config],
](cfg: C) -> C:
    """Train MobileNetV2 on Fashion-MNIST: the reference's ``mobilenetv2-fmnist`` setting.

    The prepared images have one channel; normalization repeats it across the
    network's three, as the reference does.

    Args:
      cfg: The recipe, changed in place.

    Returns:
      cfg: The same config.

    """
    model = cfg.step.model = MobileNetV2.Config()
    model.channels_in = 3
    model.channels_out = 10
    cfg.dataset.working_dir = "/datasets/fmnist"
    cfg.dataset.mean = (0.2860, 0.2860, 0.2860)
    cfg.dataset.std = (0.3530, 0.3530, 0.3530)
    return cfg
