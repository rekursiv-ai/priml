r"""MLS-Bench ``security-membership-inference-defense``: the defense loss as a slot.

``exp000`` is the task's pristine recipe at its ``resnet20-cifar10`` setting:
ResNet-20 trained on half of CIFAR-10's training split with plain
cross-entropy, then attacked by thresholding its confidence on the trained
half against the other. Every later experiment forks a setting's pristine
recipe and changes ``step.loss``, the one method the task lets an edit touch.

    exp000  resnet20-cifar10, cross-entropy
      +-- exp001  erm (the same cross-entropy)
      +-- exp002  label smoothing 0.1
      +-- exp003  confidence penalty 0.1
      +-- exp004  RelaxLoss, alpha 1
      +-- exp005  vgg16bn-cifar100, cross-entropy
      |     +-- exp006  erm
      |     +-- exp007  label smoothing 0.1
      |     +-- exp008  confidence penalty 0.1
      |     +-- exp009  RelaxLoss, alpha 0.5
      +-- exp010  mobilenetv2-fmnist, cross-entropy
            +-- exp011  erm
            +-- exp012  label smoothing 0.1
            +-- exp013  confidence penalty 0.1
            +-- exp014  RelaxLoss, alpha 1

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.activation.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.tl.membership_inference_defense.experiments.exp000

The task is pinned to MLS-Bench commit ``b873944969c53b01e4ee6e843c77767b8c702485``.
"""

from __future__ import annotations

from dataclasses import field
from typing import Self, override

from configgle import Makes, PartialConfig

from priml.baselines.mlsbench.dl.classification_loss.losses import (
    label_smoothing,
)
from priml.baselines.mlsbench.dl.vision import experiments, settings
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep
from priml.baselines.mlsbench.tl.membership_inference_defense.data import (
    MembershipData,
    normalize_only,
)
from priml.baselines.mlsbench.tl.membership_inference_defense.defenses import (
    confidence_penalty,
    relaxloss,
)
from priml.baselines.mlsbench.tl.membership_inference_defense.metrics import (
    MembershipInference,
)
from priml.baselines.mlsbench.tl.membership_inference_defense.schedules import (
    multistep,
)
from priml.model.swiglu import relu
from priml.train.train_loop import TrainLoop


class MembershipTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[VisionTrainStep.Config, MembershipData.Config],
):
    """A training loop with the vision step and the member split already in place."""

    step: VisionTrainStep.Config = field(default_factory=VisionTrainStep.Config)
    """Model, optimizer, defense and schedule."""

    dataset: MembershipData.Config = field(default_factory=MembershipData.Config)
    """Prepared tensors, trained on the members and evaluated on all three splits."""

    @override
    def finalize(self) -> Self:
        experiments.check_reference_loop(
            self,
            budget_epochs=self.step.train_budget_epochs,
        )
        return super().finalize()


def exp000() -> MembershipTrainLoop:
    """ResNet-20 on half of CIFAR-10, cross-entropy, SGD with step decay, then attacked.

    The pristine recipe of the task's ``resnet20-cifar10`` setting, the
    RelaxLoss paper's: 300 epochs of SGD at 0.1 with momentum 0.9 and weight
    decay 1e-4, divided by 10 at epochs 150 and 225, without augmentation, on
    the first half of a seeded shuffle of the training split. One evaluation
    after training scores the test split, the trained half and the untrained
    half. Frozen: improvements belong in a fork.

    Hypothesis:
      Unregularized cross-entropy overfits the trained half, so its confidence
      separates members from non-members; that leakage, at its accuracy, is
      the bar a defense must clear.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``security-membership-inference-defense`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.
      https://arxiv.org/abs/2207.05801
        Chen, Yu and Fritz 2022, "RelaxLoss: Defending Membership Inference
        Attacks without Losing Utility," whose recipe this is.

    Results:
      TBD.

    """
    vision = experiments.exp000()
    cfg = MembershipTrainLoop()
    cfg.study_name = "security_membership_inference_defense"
    cfg.experiment_name = "exp000"
    # The reference's ``--seed``; the loop salts it, so the stream still differs.
    cfg.seed = vision.seed
    cfg.step = vision.step
    cfg.dataset.augment = PartialConfig(normalize_only)
    cfg = settings.resnet20_cifar10(cfg)

    sgd = cfg.step.optimizer
    assert isinstance(sgd, PartialConfig)
    sgd.weight_decay = 1e-4
    schedule = cfg.step.schedule = PartialConfig(multistep)
    schedule.milestones = [150, 225]
    schedule.gamma = 0.1
    cfg.max_epochs = cfg.step.train_budget_epochs = 300

    # The reference evaluates once, after training; an evaluation per epoch
    # would take a loader seed each time and move every later shuffle.
    cfg.num_steps_eval = -1
    cfg.eval_every_epoch = False
    cfg.metrics_eval["privacy"] = MembershipInference.Config()
    return cfg


def exp001() -> MembershipTrainLoop:
    """exp000 + the ``erm`` baseline, which is the pristine cross-entropy.

    Hypothesis:
      Upstream publishes standard training as a baseline of its own; it is
      the pristine defense, so its result measures run-to-run variation.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``erm`` of ``security-membership-inference-defense`` at
        commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    return cfg


def exp002() -> MembershipTrainLoop:
    """exp000 + label smoothing 0.1.

    Hypothesis:
      Moving a tenth of the target mass off the true class caps how confident
      the network grows on the trained half, which should narrow the
      confidence gap the attack reads.

    References:
      https://arxiv.org/abs/1512.00567
        Szegedy et al. 2016, "Rethinking the Inception Architecture for
        Computer Vision."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``label_smoothing`` of ``security-membership-inference-defense``
        at commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    loss = cfg.step.loss = PartialConfig(label_smoothing)
    loss.smoothing = 0.1
    return cfg


def exp003() -> MembershipTrainLoop:
    """exp000 + a confidence penalty: cross-entropy minus 0.1 times the entropy.

    Hypothesis:
      Rewarding predictive entropy keeps the posteriors on the trained half
      from collapsing to one-hot, which should shrink the attack's signal.

    References:
      https://arxiv.org/abs/1701.06548
        Pereyra et al. 2017, "Regularizing Neural Networks by Penalizing
        Confident Output Distributions."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``confidence_penalty`` of
        ``security-membership-inference-defense`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"
    loss = cfg.step.loss = PartialConfig(confidence_penalty)
    loss.entropy_weight = 0.1
    loss.min_probability = 1e-8
    return cfg


def exp004() -> MembershipTrainLoop:
    """exp000 + RelaxLoss with target level 1, CIFAR-10's.

    Hypothesis:
      Holding the training loss near a level instead of driving it to zero,
      and flattening the posteriors of examples already below it, keeps the
      trained half's loss distribution near the untrained half's.

    References:
      https://arxiv.org/abs/2207.05801
        Chen, Yu and Fritz 2022, "RelaxLoss: Defending Membership Inference
        Attacks without Losing Utility."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``relaxloss`` of ``security-membership-inference-defense`` at
        commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp004"
    loss = cfg.step.loss = PartialConfig(relaxloss)
    loss.alpha = 1.0
    loss.upper = 1.0
    return cfg


def exp005() -> MembershipTrainLoop:
    """exp000 at the ``vgg16bn-cifar100`` setting: VGG-16-BN on half of CIFAR-100.

    Hypothesis:
      A hundred classes and a wider network overfit the trained half more
      than ResNet-20 on CIFAR-10, so the undefended leakage is larger.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``vgg16bn-cifar100`` of ``security-membership-inference-defense``
        at commit b873944969c53b01e4ee6e843c77767b8c702485.
      https://arxiv.org/abs/1409.1556
        Simonyan and Zisserman 2015, "Very Deep Convolutional Networks for
        Large-Scale Image Recognition."

    Results:
      TBD.

    """
    cfg = settings.vgg16bn_cifar100(exp000())
    cfg.experiment_name = "exp005"
    return cfg


def exp006() -> MembershipTrainLoop:
    """exp005 + the ``erm`` baseline, which is the pristine cross-entropy.

    Hypothesis:
      As exp001, at the ``vgg16bn-cifar100`` setting.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``erm`` of ``security-membership-inference-defense`` at
        commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp005()
    cfg.experiment_name = "exp006"
    return cfg


def exp007() -> MembershipTrainLoop:
    """exp005 + label smoothing 0.1.

    Hypothesis:
      As exp002, at the ``vgg16bn-cifar100`` setting.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``label_smoothing`` of ``security-membership-inference-defense``
        at commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp005()
    cfg.experiment_name = "exp007"
    loss = cfg.step.loss = PartialConfig(label_smoothing)
    loss.smoothing = 0.1
    return cfg


def exp008() -> MembershipTrainLoop:
    """exp005 + a confidence penalty: cross-entropy minus 0.1 times the entropy.

    Hypothesis:
      As exp003, at the ``vgg16bn-cifar100`` setting.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``confidence_penalty`` of
        ``security-membership-inference-defense`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp005()
    cfg.experiment_name = "exp008"
    loss = cfg.step.loss = PartialConfig(confidence_penalty)
    loss.entropy_weight = 0.1
    loss.min_probability = 1e-8
    return cfg


def exp009() -> MembershipTrainLoop:
    """exp005 + RelaxLoss with target level 0.5, CIFAR-100's.

    Hypothesis:
      As exp004, at the ``vgg16bn-cifar100`` setting, with the level upstream
      takes from the released VGG-11 CIFAR-100 configuration.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``relaxloss`` of ``security-membership-inference-defense`` at
        commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp005()
    cfg.experiment_name = "exp009"
    loss = cfg.step.loss = PartialConfig(relaxloss)
    loss.alpha = 0.5
    loss.upper = 1.0
    return cfg


def exp010() -> MembershipTrainLoop:
    """exp000 at the ``mobilenetv2-fmnist`` setting: MobileNetV2 on half of Fashion-MNIST.

    The task's MobileNetV2 activates with ReLU, not the scaffold's ReLU6.

    Hypothesis:
      Fashion-MNIST is easy enough that the undefended network generalizes,
      so the leakage a defense can remove is small.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``mobilenetv2-fmnist`` of
        ``security-membership-inference-defense`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.
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
    return cfg


def exp011() -> MembershipTrainLoop:
    """exp010 + the ``erm`` baseline, which is the pristine cross-entropy.

    Hypothesis:
      As exp001, at the ``mobilenetv2-fmnist`` setting.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``erm`` of ``security-membership-inference-defense`` at
        commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp010()
    cfg.experiment_name = "exp011"
    return cfg


def exp012() -> MembershipTrainLoop:
    """exp010 + label smoothing 0.1.

    Hypothesis:
      As exp002, at the ``mobilenetv2-fmnist`` setting.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``label_smoothing`` of ``security-membership-inference-defense``
        at commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp010()
    cfg.experiment_name = "exp012"
    loss = cfg.step.loss = PartialConfig(label_smoothing)
    loss.smoothing = 0.1
    return cfg


def exp013() -> MembershipTrainLoop:
    """exp010 + a confidence penalty: cross-entropy minus 0.1 times the entropy.

    Hypothesis:
      As exp003, at the ``mobilenetv2-fmnist`` setting.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``confidence_penalty`` of
        ``security-membership-inference-defense`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp010()
    cfg.experiment_name = "exp013"
    loss = cfg.step.loss = PartialConfig(confidence_penalty)
    loss.entropy_weight = 0.1
    loss.min_probability = 1e-8
    return cfg


def exp014() -> MembershipTrainLoop:
    """exp010 + RelaxLoss with target level 1, the ten-class level.

    Hypothesis:
      As exp004, at the ``mobilenetv2-fmnist`` setting.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``relaxloss`` of ``security-membership-inference-defense`` at
        commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp010()
    cfg.experiment_name = "exp014"
    loss = cfg.step.loss = PartialConfig(relaxloss)
    loss.alpha = 1.0
    loss.upper = 1.0
    return cfg
