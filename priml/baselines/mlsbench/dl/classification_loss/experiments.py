r"""MLS-Bench ``cv-classification-loss``: ResNet-56 on CIFAR-100, the loss as a slot.

``exp000`` is the task's pristine recipe with plain cross-entropy. Every later
experiment forks it and changes ``step.loss``, the one function the task lets
an edit touch. The loss is a value, not a mode, so adding a loss means writing
a function and a fork; nothing in the step or the loop changes.

    exp000  cross-entropy
      +-- exp001  label smoothing
      +-- exp002  focal loss
      +-- exp003  PolyLoss
      +-- exp004  VGG-16-BN on CIFAR-100 (the ``vgg16bn-cifar100`` setting)
      |     +-- exp005  label smoothing
      |     +-- exp006  focal loss
      |     +-- exp007  PolyLoss
      +-- exp008  MobileNetV2 on Fashion-MNIST (the ``mobilenetv2-fmnist`` setting)
            +-- exp009  label smoothing
            +-- exp010  focal loss
            +-- exp011  PolyLoss

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
from priml.baselines.mlsbench.dl.vision import settings
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


def exp004() -> VisionTrainLoop:
    """exp000 on the ``vgg16bn-cifar100`` setting: VGG-16-BN, cross-entropy.

    Hypothesis:
      A plain convolutional stack without residual paths is the second
      network a loss must help; cross-entropy is its bar.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``vgg16bn-cifar100`` of ``cv-classification-loss`` at commit
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
    """exp004 + label smoothing 0.1.

    Hypothesis:
      As in ``exp001``: softened targets bound the logits, which should
      regularize VGG-16-BN's 15 million parameters on CIFAR-100.

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
    cfg = exp004()
    cfg.experiment_name = "exp005"

    loss = cfg.step.loss = PartialConfig(label_smoothing)
    loss.smoothing = 0.1
    return cfg


def exp006() -> VisionTrainLoop:
    """exp004 + focal loss with gamma 2.

    Hypothesis:
      As in ``exp002``: down-weighting fit examples moves the gradient to the
      hard ones of the 100 classes.

    References:
      https://arxiv.org/abs/1708.02002
        Lin et al. 2017, "Focal Loss for Dense Object Detection."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``focal_loss`` of ``cv-classification-loss`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp006"

    loss = cfg.step.loss = PartialConfig(focal)
    loss.gamma = 2.0
    return cfg


def exp007() -> VisionTrainLoop:
    """exp004 + PolyLoss with epsilon 2.

    Hypothesis:
      As in ``exp003``: a heavier first polynomial term should help image
      classification whatever the network.

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
    cfg = exp004()
    cfg.experiment_name = "exp007"

    loss = cfg.step.loss = PartialConfig(poly)
    loss.epsilon = 2.0
    return cfg


def exp008() -> VisionTrainLoop:
    """exp000 on the ``mobilenetv2-fmnist`` setting: MobileNetV2, cross-entropy.

    The task's hidden setting: a mobile network on a 10-class grayscale
    dataset, which a loss tuned on CIFAR-100 has never seen.

    Hypothesis:
      Cross-entropy is the bar on a small, easy dataset too, where little is
      left to gain and a loss that over-regularizes shows.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``mobilenetv2-fmnist`` of ``cv-classification-loss`` at
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
    """exp008 + label smoothing 0.1.

    Hypothesis:
      With ten classes a tenth of the mass spread uniformly is a larger
      per-class target than on CIFAR-100, so the gain of ``exp001`` may not
      carry over.

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
    cfg = exp008()
    cfg.experiment_name = "exp009"

    loss = cfg.step.loss = PartialConfig(label_smoothing)
    loss.smoothing = 0.1
    return cfg


def exp010() -> VisionTrainLoop:
    """exp008 + focal loss with gamma 2.

    Hypothesis:
      Fashion-MNIST's errors concentrate in a few confusable classes, which a
      loss focused on hard examples should target.

    References:
      https://arxiv.org/abs/1708.02002
        Lin et al. 2017, "Focal Loss for Dense Object Detection."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``focal_loss`` of ``cv-classification-loss`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp010"

    loss = cfg.step.loss = PartialConfig(focal)
    loss.gamma = 2.0
    return cfg


def exp011() -> VisionTrainLoop:
    """exp008 + PolyLoss with epsilon 2.

    Hypothesis:
      As in ``exp003``: a heavier first polynomial term should help image
      classification whatever the network and dataset.

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
    cfg = exp008()
    cfg.experiment_name = "exp011"

    loss = cfg.step.loss = PartialConfig(poly)
    loss.epsilon = 2.0
    return cfg
