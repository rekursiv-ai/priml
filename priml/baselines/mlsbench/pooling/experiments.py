r"""MLS-Bench ``cv-pooling-aggregation``: ResNet-56 on CIFAR-100, the pool as a slot.

``exp000`` is the task's pristine recipe, global average pooling. Every later
experiment forks it and changes ``step.model.pool``, the one module the task
lets an edit touch.

    exp000  global average pool
      +-- exp001  global max pool
      +-- exp002  generalized mean, learned p from 3
      +-- exp003  mean of the average and max pools

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.pytorch_vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.pooling.experiments.exp000

The task is pinned to MLS-Bench commit ``80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.baselines.mlsbench.pooling.pools import (
    AvgMaxPool,
    GeneralizedMeanPool,
    GlobalMaxPool,
)
from priml.baselines.mlsbench.pytorch_vision import experiments as vision
from priml.baselines.mlsbench.pytorch_vision.model import ResNet


if TYPE_CHECKING:
    from priml.baselines.mlsbench.pytorch_vision.experiments import VisionTrainLoop


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
    cfg = vision.exp000()
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
