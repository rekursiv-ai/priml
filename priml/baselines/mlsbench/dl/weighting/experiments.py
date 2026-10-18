r"""MLS-Bench ``cv-sample-weighting``: ResNet-32 on long-tailed CIFAR-100.

``exp000`` is the task's pristine recipe: uniform class weights, so plain
cross-entropy on a training split whose class ``c`` keeps ``500 * 100^(-c /
99)`` images. Every later experiment forks it and changes
``step.loss.class_weight``, the one function the task lets an edit touch.

    exp000  uniform weights
      +-- exp001  square root of inverse frequency
      +-- exp002  square root of inverse effective number, beta 0.9999
      +-- exp003  log-frequency gap, rarest class weighted 5x

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.weighting.experiments.exp000

The task is pinned to MLS-Bench commit ``80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import PartialConfig

from priml.baselines.mlsbench.dl.vision import experiments
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.weighting.data import LongTailCifar100Data
from priml.baselines.mlsbench.dl.weighting.loss import ClassWeightedCrossEntropy
from priml.baselines.mlsbench.dl.weighting.weights import (
    effective_number,
    inverse_frequency,
    log_frequency_gap,
)


if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


def exp000() -> VisionTrainLoop:
    """ResNet-32 on long-tailed CIFAR-100, uniform class weights, SGD with cosine.

    The pristine recipe of the task's ``resnet32-cifar100lt`` setting: the
    ``cv-classification-loss`` recipe at depth ``(5, 5, 5)``, trained on the
    training split cut to an imbalance ratio of 100 by a NumPy generator
    seeded 42, with a cross-entropy whose class weights are all one. Frozen:
    improvements belong in a fork.

    Hypothesis:
      Unweighted cross-entropy is the bar a reweighting must clear: it fits the
      head classes and neglects the tail.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``cv-sample-weighting`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.
      https://arxiv.org/abs/1901.05555
        Cui et al. 2019, "Class-Balanced Loss Based on Effective Number of
        Samples," whose long-tailed CIFAR this is.

    Results:
      TBD.

    """
    cfg = experiments.exp000()
    cfg.study_name = "cv_sample_weighting"

    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.blocks_per_stage = (5, 5, 5)
    cfg.step.loss = ClassWeightedCrossEntropy.Config()

    cfg.dataset = LongTailCifar100Data.Config()
    return cfg


def exp001() -> VisionTrainLoop:
    """exp000 + class weights the square root of the inverse class frequency.

    Hypothesis:
      Weighting each class by ``sqrt(N / (C n_c))`` gives a tail class ten
      times a head class's say at ratio 100, which should lift tail accuracy
      without the instability of the undamped hundredfold weights.

    References:
      https://arxiv.org/abs/1901.05555
        Cui et al. 2019, "Class-Balanced Loss Based on Effective Number of
        Samples," on inverse-frequency reweighting.
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``inverse_freq`` of ``cv-sample-weighting`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"

    loss = cfg.step.loss
    assert isinstance(loss, ClassWeightedCrossEntropy.Config)
    loss.class_weight = PartialConfig(inverse_frequency)
    return cfg


def exp002() -> VisionTrainLoop:
    """exp000 + class weights the square root of the inverse effective number.

    Hypothesis:
      Near-duplicate images add less than one sample each, so a class's
      effective count ``(1 - beta^n) / (1 - beta)`` grows slower than ``n``;
      weighting by its inverse should rebalance less harshly than raw
      frequency.

    References:
      https://arxiv.org/abs/1901.05555
        Cui et al. 2019, "Class-Balanced Loss Based on Effective Number of
        Samples."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``effective_number`` of ``cv-sample-weighting`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"

    loss = cfg.step.loss
    assert isinstance(loss, ClassWeightedCrossEntropy.Config)
    weight = loss.class_weight = PartialConfig(effective_number)
    weight.beta = 0.9999
    return cfg


def exp003() -> VisionTrainLoop:
    """exp000 + class weights linear in the log-frequency gap, rarest at 5x.

    Hypothesis:
      Weighting each class by its distance below the head in log count, a
      heuristic upstream draws from Balanced Softmax's log-prior logit shift
      but not equivalent to it, is a gentler ramp than inverse frequency.

    References:
      https://arxiv.org/abs/2007.10740
        Ren et al. 2020, "Balanced Meta-Softmax for Long-Tailed Visual
        Recognition."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``balanced_softmax`` of ``cv-sample-weighting`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"

    loss = cfg.step.loss
    assert isinstance(loss, ClassWeightedCrossEntropy.Config)
    weight = loss.class_weight = PartialConfig(log_frequency_gap)
    weight.spread = 4.0
    return cfg
