r"""MLS-Bench ``cv-classification-loss``: ResNet-56 on CIFAR-100, the loss as a slot.

``exp000`` is the task's pristine recipe with plain cross-entropy. Every later
experiment forks it and changes ``step.loss``, the one function the task lets
an edit touch. The loss is a value, not a mode, so adding a loss means writing
a function and a fork; nothing in the step or the loop changes.

    exp000  cross-entropy
      +-- exp001  label smoothing
      +-- exp002  focal loss
      +-- exp003  PolyLoss

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.classification_loss.experiments.exp000

The task is pinned to MLS-Bench commit ``80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import PartialConfig

from priml.baselines.mlsbench.dl.classification_loss.losses import (
    focal,
    label_smoothing,
    poly,
)
from priml.baselines.mlsbench.dl.vision.experiments import (
    exp000 as vision_exp000,
)


if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


def exp000() -> VisionTrainLoop:
    """ResNet-56 on CIFAR-100, cross-entropy, SGD with cosine annealing.

    The reference recipe of the task's ``resnet56-cifar100`` setting and the
    one every fork is measured against. The task's script is the one the
    shared recipe ports, so this is that recipe under the task's name. Frozen:
    improvements belong in a fork, never in an edit here.

    Hypothesis:
      Plain cross-entropy is the bar a loss must clear to earn its complexity.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``cv-classification-loss`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.
      https://arxiv.org/abs/1512.03385
        He et al. 2016, "Deep Residual Learning for Image Recognition."

    Results:
      TBD.

    """
    cfg = vision_exp000()
    cfg.study_name = "cv_classification_loss"
    cfg.experiment_name = "exp000"
    return cfg


def exp001() -> VisionTrainLoop:
    """exp000 + label smoothing 0.1.

    Hypothesis:
      Moving a tenth of the target mass off the true class stops the logits
      growing without bound, which should regularize a 56-layer network that
      fits CIFAR-100's training split.

    References:
      https://arxiv.org/abs/1512.00567
        Szegedy et al. 2016, "Rethinking the Inception Architecture for
        Computer Vision."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``label_smoothing`` of ``cv-classification-loss`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"

    loss = cfg.step.loss = PartialConfig(label_smoothing)
    loss.smoothing = 0.1
    return cfg


def exp002() -> VisionTrainLoop:
    """exp000 + focal loss with gamma 2.

    Hypothesis:
      Scaling each example's loss by ``(1 - p) ** 2`` shifts the gradient to
      the examples the network still gets wrong, which should help on a
      100-way problem where most of the training split is soon fit.

    References:
      https://arxiv.org/abs/1708.02002
        Lin et al. 2017, "Focal Loss for Dense Object Detection."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``focal_loss`` of ``cv-classification-loss`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"

    loss = cfg.step.loss = PartialConfig(focal)
    loss.gamma = 2.0
    return cfg


def exp003() -> VisionTrainLoop:
    """exp000 + PolyLoss with epsilon 2.

    Hypothesis:
      Adding ``2 * (1 - p)`` to cross-entropy raises the weight of the first
      polynomial term of the loss's expansion, which PolyLoss reports as a
      gain for image classification when the leading coefficient is tuned.

    References:
      https://arxiv.org/abs/2204.12511
        Leng et al. 2022, "PolyLoss: A Polynomial Expansion Perspective of
        Classification Loss Functions."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``poly_loss`` of ``cv-classification-loss`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"

    loss = cfg.step.loss = PartialConfig(poly)
    loss.epsilon = 2.0
    return cfg
