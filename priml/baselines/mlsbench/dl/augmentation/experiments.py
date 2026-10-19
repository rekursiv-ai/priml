r"""MLS-Bench ``cv-data-augmentation``: vision networks, augmentation as a slot.

``exp000`` is the task's pristine recipe at its ``resnet56-cifar100`` setting:
pad, crop and mirror. ``exp004`` and ``exp008`` move it to the task's other
settings, changing only the network and the dataset. Every other experiment
forks one of the three and changes ``dataset.augment``, the one function the
task lets an edit touch. On Fashion-MNIST the slot receives one-channel images,
as the reference's pipeline sees grayscale PIL images; normalization repeats
the channel.

    exp000  ResNet-56, CIFAR-100, pad, crop and mirror   (resnet56-cifar100)
      +-- exp001  + Cutout, one 16-pixel hole
      +-- exp002  RandAugment, two operations at magnitude 9, first
      +-- exp003  TrivialAugment Wide, first
      +-- exp004  ResNet-20, CIFAR-10                     (resnet20-cifar10)
      |     +-- exp005  + Cutout
      |     +-- exp006  RandAugment
      |     +-- exp007  TrivialAugment Wide
      +-- exp008  MobileNetV2, Fashion-MNIST              (mobilenetv2-fmnist, hidden)
            +-- exp009  + Cutout
            +-- exp010  RandAugment
            +-- exp011  TrivialAugment Wide

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.augmentation.experiments.exp000

The task is pinned to MLS-Bench commit ``80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import PartialConfig

from priml.baselines.mlsbench.dl.augmentation.augment import (
    cutout_normalize,
    randaugment_normalize,
    trivialaugment_normalize,
)
from priml.baselines.mlsbench.dl.vision import experiments, settings


if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


def exp000() -> VisionTrainLoop:
    """ResNet-56 on CIFAR-100, padded crop and mirror, SGD with cosine annealing.

    The pristine recipe of the task's ``resnet56-cifar100`` setting. Its script
    is the ``cv-classification-loss`` script with the loss fixed to
    cross-entropy and the CIFAR test pipeline without its no-op resize, so this
    is that task's ``exp000`` under this task's name. Frozen: improvements
    belong in a fork.

    Hypothesis:
      Padded random crops and mirrors are the bar an augmentation must clear
      to earn its complexity.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``cv-data-augmentation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = experiments.exp000()
    cfg.study_name = "cv_data_augmentation"
    return cfg


def exp001() -> VisionTrainLoop:
    """exp000 + Cutout: zero one 16-pixel square of every crop.

    Hypothesis:
      Hiding a random patch forces the network to use the whole image rather
      than its most discriminative part, which regularizes a network that
      fits CIFAR-100's training split.

    References:
      https://arxiv.org/abs/1708.04552
        DeVries and Taylor 2017, "Improved Regularization of Convolutional
        Neural Networks with Cutout."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``cutout`` of ``cv-data-augmentation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    augment = cfg.dataset.augment = PartialConfig(cutout_normalize)
    augment.holes = 1
    augment.length = 16
    return cfg


def exp002() -> VisionTrainLoop:
    """exp000 + RandAugment, two operations at magnitude 9, before the crop.

    Hypothesis:
      Two operations drawn from a fixed list at one shared magnitude span most
      of AutoAugment's gain without its policy search.

    References:
      https://arxiv.org/abs/1909.13719
        Cubuk et al. 2020, "RandAugment: Practical automated data augmentation
        with a reduced search space."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``randaugment`` of ``cv-data-augmentation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    augment = cfg.dataset.augment = PartialConfig(randaugment_normalize)
    augment.num_ops = 2
    augment.magnitude = 9
    augment.num_magnitude_bins = 31
    return cfg


def exp003() -> VisionTrainLoop:
    """exp000 + TrivialAugment Wide, one operation at a random magnitude, first.

    Hypothesis:
      One operation per image at a uniformly drawn magnitude needs no tuning
      and matches tuned policies, which TrivialAugment reports on CIFAR.

    References:
      https://arxiv.org/abs/2103.10158
        Mueller and Hutter 2021, "TrivialAugment: Tuning-free Yet
        State-of-the-Art Data Augmentation."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``trivialaugment`` of ``cv-data-augmentation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"
    augment = cfg.dataset.augment = PartialConfig(trivialaugment_normalize)
    augment.num_magnitude_bins = 31
    return cfg


def exp004() -> VisionTrainLoop:
    """exp000 at the resnet20-cifar10 setting: ResNet-20 on CIFAR-10, pad, crop and mirror.

    Only the network's depth and classes and the dataset change.

    Hypothesis:
      Padded random crops and mirrors are the bar an augmentation must clear
      on CIFAR-10, where a shallower network overfits less.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``resnet20-cifar10`` of ``cv-data-augmentation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = settings.resnet20_cifar10(exp000())
    cfg.experiment_name = "exp004"
    return cfg


def exp005() -> VisionTrainLoop:
    """exp004 + Cutout: zero one 16-pixel square of every crop.

    Hypothesis:
      As ``exp001``, on CIFAR-10 at ResNet-20 depth.

    References:
      https://arxiv.org/abs/1708.04552
        DeVries and Taylor 2017, "Improved Regularization of Convolutional
        Neural Networks with Cutout."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``cutout`` of ``cv-data-augmentation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"
    augment = cfg.dataset.augment = PartialConfig(cutout_normalize)
    augment.holes = 1
    augment.length = 16
    return cfg


def exp006() -> VisionTrainLoop:
    """exp004 + RandAugment, two operations at magnitude 9, before the crop.

    Hypothesis:
      As ``exp002``, on CIFAR-10 at ResNet-20 depth.

    References:
      https://arxiv.org/abs/1909.13719
        Cubuk et al. 2020, "RandAugment: Practical automated data augmentation
        with a reduced search space."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``randaugment`` of ``cv-data-augmentation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp006"
    augment = cfg.dataset.augment = PartialConfig(randaugment_normalize)
    augment.num_ops = 2
    augment.magnitude = 9
    augment.num_magnitude_bins = 31
    return cfg


def exp007() -> VisionTrainLoop:
    """exp004 + TrivialAugment Wide, one operation at a random magnitude, first.

    Hypothesis:
      As ``exp003``, on CIFAR-10 at ResNet-20 depth.

    References:
      https://arxiv.org/abs/2103.10158
        Mueller and Hutter 2021, "TrivialAugment: Tuning-free Yet
        State-of-the-Art Data Augmentation."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``trivialaugment`` of ``cv-data-augmentation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp007"
    augment = cfg.dataset.augment = PartialConfig(trivialaugment_normalize)
    augment.num_magnitude_bins = 31
    return cfg


def exp008() -> VisionTrainLoop:
    """exp000 at the hidden mobilenetv2-fmnist setting: MobileNetV2 on Fashion-MNIST.

    Only the network and the dataset change. The augmentation receives the
    prepared one-channel 32x32 images, as the reference's pipeline receives
    grayscale images after its ``Resize(32)``.

    Hypothesis:
      Padded random crops and mirrors are the bar an augmentation must clear
      on grayscale clothing images, where colour operations do little.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Hidden setting ``mobilenetv2-fmnist`` of ``cv-data-augmentation`` at
        commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.
      https://arxiv.org/abs/1801.04381
        Sandler et al. 2018, "MobileNetV2: Inverted Residuals and Linear
        Bottlenecks."

    Results:
      TBD.

    """
    cfg = settings.mobilenetv2_fmnist(exp000())
    cfg.experiment_name = "exp008"
    return cfg


def exp009() -> VisionTrainLoop:
    """exp008 + Cutout: zero one 16-pixel square of every crop.

    Hypothesis:
      As ``exp001``; hiding a patch of a garment forces the network to use
      its whole silhouette.

    References:
      https://arxiv.org/abs/1708.04552
        DeVries and Taylor 2017, "Improved Regularization of Convolutional
        Neural Networks with Cutout."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``cutout`` of ``cv-data-augmentation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp009"
    augment = cfg.dataset.augment = PartialConfig(cutout_normalize)
    augment.holes = 1
    augment.length = 16
    return cfg


def exp010() -> VisionTrainLoop:
    """exp008 + RandAugment, two operations at magnitude 9, before the crop.

    Hypothesis:
      As ``exp002``; on grayscale images ``Color`` is the identity, so the
      geometric and tonal operations carry the gain.

    References:
      https://arxiv.org/abs/1909.13719
        Cubuk et al. 2020, "RandAugment: Practical automated data augmentation
        with a reduced search space."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``randaugment`` of ``cv-data-augmentation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp010"
    augment = cfg.dataset.augment = PartialConfig(randaugment_normalize)
    augment.num_ops = 2
    augment.magnitude = 9
    augment.num_magnitude_bins = 31
    return cfg


def exp011() -> VisionTrainLoop:
    """exp008 + TrivialAugment Wide, one operation at a random magnitude, first.

    Hypothesis:
      As ``exp003``, on grayscale Fashion-MNIST.

    References:
      https://arxiv.org/abs/2103.10158
        Mueller and Hutter 2021, "TrivialAugment: Tuning-free Yet
        State-of-the-Art Data Augmentation."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``trivialaugment`` of ``cv-data-augmentation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp011"
    augment = cfg.dataset.augment = PartialConfig(trivialaugment_normalize)
    augment.num_magnitude_bins = 31
    return cfg
