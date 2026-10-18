r"""MLS-Bench ``cv-data-augmentation``: ResNet-56 on CIFAR-100, augmentation as a slot.

``exp000`` is the task's pristine recipe: pad, crop and mirror. Every later
experiment forks it and changes ``dataset.augment``, the one function the task
lets an edit touch.

    exp000  pad, crop and mirror
      +-- exp001  + Cutout, one 16-pixel hole
      +-- exp002  RandAugment, two operations at magnitude 9, first
      +-- exp003  TrivialAugment Wide, first

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
from priml.baselines.mlsbench.dl.vision import experiments


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
