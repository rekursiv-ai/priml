r"""MLS-Bench ``security-poison-robust-learning``: the robust loss as a slot.

``exp000`` is the task's pristine recipe at its ``resnet20-cifar10-labelflip``
setting: ResNet-20 trained for 100 epochs on CIFAR-10 with a tenth of its
training labels moved to the next class, then scored on the clean test split
and on how often it predicts the flipped labels. Every later experiment forks a
setting's pristine recipe and changes ``step.loss``, the one class the task
lets an edit touch.

    exp000  resnet20-cifar10-labelflip, cross-entropy
      +-- exp001  cross_entropy (the same cross-entropy)
      +-- exp002  generalized cross-entropy, q 0.7
      +-- exp003  symmetric cross-entropy, alpha 1, beta 0.5
      +-- exp004  soft bootstrapping, beta 0.8
      +-- exp005  vgg16bn-cifar100-labelflip, cross-entropy
      |     +-- exp006  cross_entropy
      |     +-- exp007  generalized cross-entropy
      |     +-- exp008  symmetric cross-entropy
      |     +-- exp009  soft bootstrapping
      +-- exp010  mobilenetv2-fmnist-labelflip, cross-entropy
            +-- exp011  cross_entropy
            +-- exp012  generalized cross-entropy
            +-- exp013  symmetric cross-entropy
            +-- exp014  soft bootstrapping

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data --dataset cifar10
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.tl.poison_robust_learning.experiments.exp000

The task is pinned to MLS-Bench commit ``80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c``.
"""

from __future__ import annotations

from dataclasses import field
from typing import Self, override

from configgle import Makes, PartialConfig

from priml.baselines.mlsbench.dl.vision import experiments, settings
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep
from priml.baselines.mlsbench.tl.poison_robust_learning.data import PoisonedData
from priml.baselines.mlsbench.tl.poison_robust_learning.losses import (
    bootstrap,
    generalized_ce,
    symmetric_ce,
)
from priml.baselines.mlsbench.tl.poison_robust_learning.metrics import (
    PoisonRobustness,
)
from priml.model.custom_types import ChannelsOut
from priml.model.swiglu import relu
from priml.train.train_loop import TrainLoop


class PoisonTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[VisionTrainStep.Config, PoisonedData.Config],
):
    """A training loop with the vision step and the label-flipped data already in place."""

    step: VisionTrainStep.Config = field(default_factory=VisionTrainStep.Config)
    """Model, optimizer, robust loss and schedule."""

    dataset: PoisonedData.Config = field(default_factory=PoisonedData.Config)
    """Prepared tensors, trained on flipped labels and evaluated on two splits."""

    @override
    def finalize(self) -> Self:
        experiments.check_reference_loop(
            self,
            budget_epochs=self.step.train_budget_epochs,
        )
        # A flipped label wraps at the dataset's class count, which the
        # reference reads off its dataset and the network's head shares.
        if self.dataset.num_classes == -1 and isinstance(self.step.model, ChannelsOut):
            self.dataset.num_classes = self.step.model.channels_out
        return super().finalize()


def exp000() -> PoisonTrainLoop:
    """ResNet-20 on CIFAR-10 with a tenth of its labels flipped, cross-entropy, SGD with cosine.

    The pristine recipe of the task's ``resnet20-cifar10-labelflip`` setting:
    the ``cv-classification-loss`` recipe at ResNet-20 on CIFAR-10 for 100
    epochs, its training split's labels flipped for a seeded tenth of the
    images. One evaluation after training scores the clean test split and the
    flipped images through the training pipeline. Frozen: improvements belong
    in a fork.

    Hypothesis:
      Cross-entropy fits the flipped labels along with the clean ones; that
      fit, at its test accuracy, is the bar a robust loss must clear.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``security-poison-robust-learning`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.
      https://arxiv.org/abs/1512.03385
        He et al. 2016, "Deep Residual Learning for Image Recognition."

    Results:
      TBD.

    """
    vision = experiments.exp000()
    cfg = PoisonTrainLoop()
    cfg.study_name = "security_poison_robust_learning"
    cfg.experiment_name = "exp000"
    # The reference's ``--seed``; the loop salts it, so the stream still differs.
    cfg.seed = vision.seed
    cfg.step = vision.step
    cfg = settings.resnet20_cifar10(cfg)
    cfg.max_epochs = cfg.step.train_budget_epochs = 100

    # The reference evaluates once, after training; an evaluation per epoch
    # would take a loader seed each time and move every later shuffle.
    cfg.num_steps_eval = -1
    cfg.eval_every_epoch = False
    cfg.metrics_eval["poison"] = PoisonRobustness.Config()
    return cfg


def exp001() -> PoisonTrainLoop:
    """exp000 + the ``cross_entropy`` baseline, which is the pristine loss.

    Hypothesis:
      Upstream publishes plain cross-entropy as a baseline of its own; it is
      the pristine loss, so its result measures run-to-run variation.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``cross_entropy`` of ``security-poison-robust-learning`` at
        commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    return cfg


def exp002() -> PoisonTrainLoop:
    """exp000 + generalized cross-entropy with q 0.7.

    Hypothesis:
      Scaling each example's cross-entropy gradient by ``p^q`` discounts the
      examples the network gives low probability, which a flipped label is,
      so the network should fit fewer of them.

    References:
      https://arxiv.org/abs/1805.07836
        Zhang and Sabuncu 2018, "Generalized Cross Entropy Loss for Training
        Deep Neural Networks with Noisy Labels."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``generalized_ce`` of ``security-poison-robust-learning`` at
        commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    loss = cfg.step.loss = PartialConfig(generalized_ce)
    loss.q = 0.7
    loss.min_probability = 1e-8
    return cfg


def exp003() -> PoisonTrainLoop:
    """exp000 + symmetric cross-entropy with alpha 1 and beta 0.5.

    Hypothesis:
      The reverse cross-entropy is bounded and noise tolerant; adding half of
      it to cross-entropy should keep the clean fit and resist the flips.

    References:
      https://arxiv.org/abs/1908.06112
        Wang et al. 2019, "Symmetric Cross Entropy for Robust Learning with
        Noisy Labels."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``symmetric_ce`` of ``security-poison-robust-learning`` at
        commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"
    loss = cfg.step.loss = PartialConfig(symmetric_ce)
    loss.alpha = 1.0
    loss.beta = 0.5
    loss.min_probability = 1e-8
    loss.min_target = 1e-4
    return cfg


def exp004() -> PoisonTrainLoop:
    """exp000 + soft bootstrapping with beta 0.8.

    Hypothesis:
      Moving a fifth of each target onto the network's own prediction lets a
      confident prediction outvote a flipped label.

    References:
      https://arxiv.org/abs/1412.6596
        Reed et al. 2015, "Training Deep Neural Networks on Noisy Labels with
        Bootstrapping."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``bootstrap`` of ``security-poison-robust-learning`` at
        commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp004"
    loss = cfg.step.loss = PartialConfig(bootstrap)
    loss.beta = 0.8
    return cfg


def exp005() -> PoisonTrainLoop:
    """exp000 at the ``vgg16bn-cifar100-labelflip`` setting: VGG-16-BN on CIFAR-100.

    The pristine recipe of that setting: the same loss, schedule, flip fraction
    and evaluation, with VGG-16-BN on CIFAR-100.

    Hypothesis:
      A hundred classes leave fewer clean images per class to outvote the
      flips, so the bar a robust loss must clear moves.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``vgg16bn-cifar100-labelflip`` of
        ``security-poison-robust-learning`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.
      https://arxiv.org/abs/1409.1556
        Simonyan and Zisserman 2015, "Very Deep Convolutional Networks for
        Large-Scale Image Recognition."

    Results:
      TBD.

    """
    cfg = settings.vgg16bn_cifar100(exp000())
    cfg.experiment_name = "exp005"
    return cfg


def exp006() -> PoisonTrainLoop:
    """exp005 + the ``cross_entropy`` baseline, which is the pristine loss.

    Hypothesis:
      As exp001, at this setting: a measure of run-to-run variation.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``cross_entropy`` of ``security-poison-robust-learning`` at
        commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp005()
    cfg.experiment_name = "exp006"
    return cfg


def exp007() -> PoisonTrainLoop:
    """exp005 + generalized cross-entropy with q 0.7.

    Hypothesis:
      As exp002, at this setting.

    References:
      https://arxiv.org/abs/1805.07836
        Zhang and Sabuncu 2018, "Generalized Cross Entropy Loss for Training
        Deep Neural Networks with Noisy Labels."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``generalized_ce`` of ``security-poison-robust-learning`` at
        commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp005()
    cfg.experiment_name = "exp007"
    loss = cfg.step.loss = PartialConfig(generalized_ce)
    loss.q = 0.7
    loss.min_probability = 1e-8
    return cfg


def exp008() -> PoisonTrainLoop:
    """exp005 + symmetric cross-entropy with alpha 1 and beta 0.5.

    Hypothesis:
      As exp003, at this setting.

    References:
      https://arxiv.org/abs/1908.06112
        Wang et al. 2019, "Symmetric Cross Entropy for Robust Learning with
        Noisy Labels."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``symmetric_ce`` of ``security-poison-robust-learning`` at
        commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp005()
    cfg.experiment_name = "exp008"
    loss = cfg.step.loss = PartialConfig(symmetric_ce)
    loss.alpha = 1.0
    loss.beta = 0.5
    loss.min_probability = 1e-8
    loss.min_target = 1e-4
    return cfg


def exp009() -> PoisonTrainLoop:
    """exp005 + soft bootstrapping with beta 0.8.

    Hypothesis:
      As exp004, at this setting.

    References:
      https://arxiv.org/abs/1412.6596
        Reed et al. 2015, "Training Deep Neural Networks on Noisy Labels with
        Bootstrapping."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``bootstrap`` of ``security-poison-robust-learning`` at
        commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp005()
    cfg.experiment_name = "exp009"
    loss = cfg.step.loss = PartialConfig(bootstrap)
    loss.beta = 0.8
    return cfg


def exp010() -> PoisonTrainLoop:
    """exp000 at the ``mobilenetv2-fmnist-labelflip`` setting: MobileNetV2 on Fashion-MNIST.

    The pristine recipe of that setting: the same loss, schedule and
    evaluation, with MobileNetV2 on Fashion-MNIST and three twentieths of the
    labels flipped. This task's driver activates MobileNetV2 with ReLU, not
    the ReLU6 of the shared scaffold's.

    Hypothesis:
      Fashion-MNIST's neighbouring classes are its most confusable, so a flip
      to the next class is harder to resist than on CIFAR-10.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``mobilenetv2-fmnist-labelflip`` of
        ``security-poison-robust-learning`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.
      https://arxiv.org/abs/1801.04381
        Sandler et al. 2018, "MobileNetV2: Inverted Residuals and Linear
        Bottlenecks."

    Results:
      TBD.

    """
    cfg = settings.mobilenetv2_fmnist(exp000())
    cfg.experiment_name = "exp010"
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    cfg.step.model.activation = relu
    cfg.dataset.poison_fraction = 0.15
    return cfg


def exp011() -> PoisonTrainLoop:
    """exp010 + the ``cross_entropy`` baseline, which is the pristine loss.

    Hypothesis:
      As exp001, at this setting: a measure of run-to-run variation.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``cross_entropy`` of ``security-poison-robust-learning`` at
        commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp010()
    cfg.experiment_name = "exp011"
    return cfg


def exp012() -> PoisonTrainLoop:
    """exp010 + generalized cross-entropy with q 0.7.

    Hypothesis:
      As exp002, at this setting.

    References:
      https://arxiv.org/abs/1805.07836
        Zhang and Sabuncu 2018, "Generalized Cross Entropy Loss for Training
        Deep Neural Networks with Noisy Labels."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``generalized_ce`` of ``security-poison-robust-learning`` at
        commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp010()
    cfg.experiment_name = "exp012"
    loss = cfg.step.loss = PartialConfig(generalized_ce)
    loss.q = 0.7
    loss.min_probability = 1e-8
    return cfg


def exp013() -> PoisonTrainLoop:
    """exp010 + symmetric cross-entropy with alpha 1 and beta 0.5.

    Hypothesis:
      As exp003, at this setting.

    References:
      https://arxiv.org/abs/1908.06112
        Wang et al. 2019, "Symmetric Cross Entropy for Robust Learning with
        Noisy Labels."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``symmetric_ce`` of ``security-poison-robust-learning`` at
        commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp010()
    cfg.experiment_name = "exp013"
    loss = cfg.step.loss = PartialConfig(symmetric_ce)
    loss.alpha = 1.0
    loss.beta = 0.5
    loss.min_probability = 1e-8
    loss.min_target = 1e-4
    return cfg


def exp014() -> PoisonTrainLoop:
    """exp010 + soft bootstrapping with beta 0.8.

    Hypothesis:
      As exp004, at this setting.

    References:
      https://arxiv.org/abs/1412.6596
        Reed et al. 2015, "Training Deep Neural Networks on Noisy Labels with
        Bootstrapping."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``bootstrap`` of ``security-poison-robust-learning`` at
        commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp010()
    cfg.experiment_name = "exp014"
    loss = cfg.step.loss = PartialConfig(bootstrap)
    loss.beta = 0.8
    return cfg
