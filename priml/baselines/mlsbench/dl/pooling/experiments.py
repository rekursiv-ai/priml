r"""MLS-Bench ``cv-pooling-aggregation``: ResNet-56 on CIFAR-100, the pool as a slot.

``exp000`` is the task's pristine recipe, global average pooling. Every later
experiment forks it and changes ``step.model.pool``, the one module the task
lets an edit touch.

    exp000  global average pool
      +-- exp001  global max pool
      +-- exp002  generalized mean, learned p from 3
      +-- exp003  mean of the average and max pools
      +-- exp004  VGG-16-BN on CIFAR-100 (the ``vgg16bn-cifar100`` setting)
      |     +-- exp005  global max pool
      |     +-- exp006  generalized mean, learned p from 3
      |     +-- exp007  mean of the average and max pools
      +-- exp008  MobileNetV2 on Fashion-MNIST (the ``mobilenetv2-fmnist`` setting)
            +-- exp009  global max pool
            +-- exp010  generalized mean, learned p from 3
            +-- exp011  mean of the average and max pools

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.pooling.experiments.exp000

The task is pinned to MLS-Bench commit ``80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.baselines.mlsbench.dl.pooling.pools import (
    AvgMaxPool,
    GeneralizedMeanPool,
    GlobalMaxPool,
)
from priml.baselines.mlsbench.dl.vision import experiments, settings
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN


if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


def exp000() -> VisionTrainLoop:
    """ResNet-56 on CIFAR-100, global average pooling, SGD with cosine annealing.

    The pristine recipe of the task's ``resnet56-cifar100`` setting. Its script
    is the ``cv-classification-loss`` script with the pool made a module and
    the loss an ``nn.CrossEntropyLoss``, which computes what that task's
    ``cross_entropy`` does; so this is that task's ``exp000``. Frozen:
    improvements belong in a fork.

    Hypothesis:
      Averaging over the grid is the bar a pool must clear to earn its
      complexity.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``cv-pooling-aggregation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.
      https://arxiv.org/abs/1312.4400
        Lin, Chen and Yan 2014, "Network In Network."

    Results:
      TBD.

    """
    cfg = experiments.exp000()
    cfg.study_name = "cv_pooling_aggregation"
    cfg.experiment_name = "exp000"
    return cfg


def exp001() -> VisionTrainLoop:
    """exp000 + global max pooling.

    Hypothesis:
      Keeping each channel's strongest response rather than its mean passes
      the most salient evidence to the head, which should help when an object
      fills only part of the grid.

    References:
      https://doi.org/10.1109/CVPR.2015.7298668
        Oquab et al. 2015, "Is object localization for free? Weakly-supervised
        learning with convolutional neural networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``global_max`` of ``cv-pooling-aggregation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.pool = GlobalMaxPool.Config()
    return cfg


def exp002() -> VisionTrainLoop:
    """exp000 + generalized-mean pooling with a learned exponent starting at 3.

    Hypothesis:
      A learned exponent lets training choose a point between the average
      (``p = 1``) and the max (``p -> inf``), so it should match or beat both.

    References:
      https://arxiv.org/abs/1711.02512
        Radenovic, Tolias and Chum 2018, "Fine-tuning CNN Image Retrieval with
        No Human Annotation."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``gem`` of ``cv-pooling-aggregation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.pool = GeneralizedMeanPool.Config()
    return cfg


def exp003() -> VisionTrainLoop:
    """exp000 + the mean of the global average and global max pools.

    Hypothesis:
      The average summarizes the whole grid and the max its peak; their mean
      gives the head both, which should be more robust than either alone.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``avg_max`` of ``cv-pooling-aggregation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.pool = AvgMaxPool.Config()
    return cfg


def exp004() -> VisionTrainLoop:
    """exp000 on the ``vgg16bn-cifar100`` setting: VGG-16-BN, global average pooling.

    After five 2x2 max pools VGG-16's last stage is one position, so every
    global pool there reduces a single value per channel.

    Hypothesis:
      The average pool is the bar on a plain convolutional stack too.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``vgg16bn-cifar100`` of ``cv-pooling-aggregation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.
      https://arxiv.org/abs/1409.1556
        Simonyan and Zisserman 2015, "Very Deep Convolutional Networks for
        Large-Scale Image Recognition."

    Results:
      TBD.

    """
    cfg = settings.vgg16bn_cifar100(exp000())
    cfg.experiment_name = "exp004"
    return cfg


def exp005() -> VisionTrainLoop:
    """exp004 + global max pooling.

    Hypothesis:
      On VGG-16's one-position grid the max equals the average, so this
      should match ``exp004``; a difference would expose the pool's cost.

    References:
      https://doi.org/10.1109/CVPR.2015.7298668
        Oquab et al. 2015, "Is object localization for free? Weakly-supervised
        learning with convolutional neural networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``global_max`` of ``cv-pooling-aggregation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"
    assert isinstance(cfg.step.model, VGG16BN.Config)
    cfg.step.model.pool = GlobalMaxPool.Config()
    return cfg


def exp006() -> VisionTrainLoop:
    """exp004 + generalized-mean pooling with a learned exponent starting at 3.

    Hypothesis:
      On one position the generalized mean is the clamped activation itself,
      so its only effect is the floor at ``eps``, which drops negatives that
      the average passes.

    References:
      https://arxiv.org/abs/1711.02512
        Radenovic, Tolias and Chum 2018, "Fine-tuning CNN Image Retrieval with
        No Human Annotation."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``gem`` of ``cv-pooling-aggregation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp006"
    assert isinstance(cfg.step.model, VGG16BN.Config)
    cfg.step.model.pool = GeneralizedMeanPool.Config()
    return cfg


def exp007() -> VisionTrainLoop:
    """exp004 + the mean of the global average and global max pools.

    Hypothesis:
      As ``exp005``: on one position both pools agree, so this should match
      ``exp004``.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``avg_max`` of ``cv-pooling-aggregation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp007"
    assert isinstance(cfg.step.model, VGG16BN.Config)
    cfg.step.model.pool = AvgMaxPool.Config()
    return cfg


def exp008() -> VisionTrainLoop:
    """exp000 on the ``mobilenetv2-fmnist`` setting: MobileNetV2, global average pooling.

    The task's hidden setting; MobileNetV2's last convolution leaves a 4x4
    grid of 1280 channels for the pool.

    Hypothesis:
      The average pool is the bar on a mobile network and grayscale data too.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``mobilenetv2-fmnist`` of ``cv-pooling-aggregation`` at
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
    """exp008 + global max pooling.

    Hypothesis:
      Fashion-MNIST's garments sit centred on a plain background, so the
      strongest response per channel should carry the class as well as the
      mean does.

    References:
      https://doi.org/10.1109/CVPR.2015.7298668
        Oquab et al. 2015, "Is object localization for free? Weakly-supervised
        learning with convolutional neural networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``global_max`` of ``cv-pooling-aggregation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp009"
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    cfg.step.model.pool = GlobalMaxPool.Config()
    return cfg


def exp010() -> VisionTrainLoop:
    """exp008 + generalized-mean pooling with a learned exponent starting at 3.

    Hypothesis:
      As ``exp002``: a learned exponent can settle between average and max,
      so it should match or beat both on this setting too.

    References:
      https://arxiv.org/abs/1711.02512
        Radenovic, Tolias and Chum 2018, "Fine-tuning CNN Image Retrieval with
        No Human Annotation."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``gem`` of ``cv-pooling-aggregation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp010"
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    cfg.step.model.pool = GeneralizedMeanPool.Config()
    return cfg


def exp011() -> VisionTrainLoop:
    """exp008 + the mean of the global average and global max pools.

    Hypothesis:
      As ``exp003``: giving the head both the grid's summary and its peak
      should be more robust than either alone.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``avg_max`` of ``cv-pooling-aggregation`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp011"
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    cfg.step.model.pool = AvgMaxPool.Config()
    return cfg
