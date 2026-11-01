r"""MLS-Bench ``cv-multitask-loss``: ResNet-56 on CIFAR-100 classes and superclasses.

``exp000`` is the task's pristine recipe: a ResNet with a fine head (100
classes) and a coarse head (20 superclasses), trained on the sum of their
cross-entropies. Every later experiment forks it and changes
``step.mtl_loss``, the one module the task lets an edit touch.

    exp000  sum of the two losses
      +-- exp001  uncertainty weighting
      +-- exp002  dynamic weight average
      +-- exp003  PCGrad
      +-- exp004  ResNet-20 trunk (the ``resnet20-cifar100mt`` setting), sum
      |     +-- exp005  uncertainty weighting
      |     +-- exp006  dynamic weight average
      |     +-- exp007  PCGrad
      +-- exp008  VGG-16-BN trunk (the hidden ``vgg16bn-cifar100mt`` setting), sum
            +-- exp009  uncertainty weighting
            +-- exp010  dynamic weight average
            +-- exp011  PCGrad

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.multitask.experiments.exp000

The task is pinned to MLS-Bench commit ``b873944969c53b01e4ee6e843c77767b8c702485``.
"""

from __future__ import annotations

from dataclasses import field
from typing import Self, override

from configgle import Makes

from priml.baselines.mlsbench.dl.multitask.data import (
    NUM_COARSE,
    MultiTaskCifar100Data,
)
from priml.baselines.mlsbench.dl.multitask.losses import (
    DynamicWeightAverage,
    PCGrad,
    UncertaintyLoss,
)
from priml.baselines.mlsbench.dl.multitask.metrics import HeadAccuracy
from priml.baselines.mlsbench.dl.multitask.model import TwoHeadResNet
from priml.baselines.mlsbench.dl.multitask.train_step import MultiTaskStep
from priml.baselines.mlsbench.dl.vision import experiments, settings
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.train.train_loop import TrainLoop


class MultiTaskTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[MultiTaskStep.Config, MultiTaskCifar100Data.Config],
):
    """A training loop with the multi-task step and data already in place."""

    step: MultiTaskStep.Config = field(default_factory=MultiTaskStep.Config)
    """Two-head model, optimizer, per-head loss and their combination."""

    dataset: MultiTaskCifar100Data.Config = field(
        default_factory=MultiTaskCifar100Data.Config,
    )
    """Prepared CIFAR-100 tensors with each image's superclass."""

    @override
    def finalize(self) -> Self:
        experiments.check_reference_loop(
            self,
            budget_epochs=self.step.train_budget_epochs,
        )
        return super().finalize()


def exp000() -> MultiTaskTrainLoop:
    """ResNet-56 on CIFAR-100 with fine and coarse heads, the sum of their losses.

    The pristine recipe of the task's ``resnet56-cifar100mt`` setting. Its
    script is the ``cv-classification-loss`` script with a second head of 20
    superclasses, registered after the first and initialized with it, each
    head's cross-entropy summed, and the summing module's parameters (none
    here) appended to the optimizer's. So this is that task's ``exp000`` with
    those four changes. Frozen: improvements belong in a fork.

    Hypothesis:
      An unweighted sum is the bar a combination of the two losses must clear
      to earn its complexity.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``cv-multitask-loss`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.
      https://arxiv.org/abs/1512.03385
        He et al. 2016, "Deep Residual Learning for Image Recognition."

    Results:
      TBD.

    """
    parent = experiments.exp000()
    cfg = MultiTaskTrainLoop().update(parent, skip_missing=True)
    cfg.study_name = "cv_multitask_loss"
    cfg.experiment_name = "exp000"

    cfg.dataset = MultiTaskCifar100Data.Config()
    cfg.dataset.data = parent.dataset
    cfg.step = MultiTaskStep.Config().update(parent.step)
    assert isinstance(parent.step.model, ResNet.Config)
    model = cfg.step.model = TwoHeadResNet.Config()
    model.trunk = parent.step.model
    model.channels_out = NUM_COARSE

    # Both heads' logits reach the metrics joined, fine then coarse.
    fine = cfg.metrics_eval["test_acc"] = HeadAccuracy.Config()
    fine.logits_stop = model.trunk.channels_out
    coarse = cfg.metrics_eval["test_acc_coarse"] = HeadAccuracy.Config()
    coarse.logits_start = model.trunk.channels_out
    coarse.logits_stop = model.trunk.channels_out + NUM_COARSE
    coarse.label_key = "label_coarse"
    return cfg


def exp001() -> MultiTaskTrainLoop:
    """exp000 + uncertainty weighting with learned log-variances.

    Hypothesis:
      Learning each task's noise scale weights the losses by their
      homoscedastic uncertainty, so the easier coarse task stops dominating
      the shared features once it is fit.

    References:
      https://arxiv.org/abs/1705.07115
        Kendall, Gal and Cipolla 2018, "Multi-Task Learning Using Uncertainty
        to Weigh Losses for Scene Geometry and Semantics."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``uncertainty`` of ``cv-multitask-loss`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    cfg.step.mtl_loss = UncertaintyLoss.Config()
    return cfg


def exp002() -> MultiTaskTrainLoop:
    """exp000 + dynamic weight average at temperature 2.

    Hypothesis:
      Up-weighting the task whose loss fell least since the previous step
      balances the two tasks' rates of progress without any learned weight.

    References:
      https://arxiv.org/abs/1803.10704
        Liu, Johns and Davison 2019, "End-to-End Multi-Task Learning with
        Attention."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``dwa`` of ``cv-multitask-loss`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    cfg.step.mtl_loss = DynamicWeightAverage.Config()
    return cfg


def exp003() -> MultiTaskTrainLoop:
    """exp000 + PCGrad gradient surgery over every shared parameter.

    Hypothesis:
      Removing the component of each task's gradient that opposes the other's
      stops the two heads from undoing each other's updates to the trunk. As
      upstream, the surgery covers the fine loss's leaves only, so the coarse
      head is never updated, ``test_acc_coarse`` scores its initialization,
      and the logged training loss is the surrogate, not a cross-entropy.

    References:
      https://arxiv.org/abs/2001.06782
        Yu et al. 2020, "Gradient Surgery for Multi-Task Learning."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``pcgrad`` of ``cv-multitask-loss`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485. Its walk of the graph skips
        a heap-dependent subset of the parameters; this takes all of them.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"
    cfg.step.mtl_loss = PCGrad.Config()
    return cfg


def exp004() -> MultiTaskTrainLoop:
    """exp000 at the ``resnet20-cifar100mt`` setting: a ResNet-20 trunk, the sum.

    The setting's script builds ``ResNet(BasicBlock, [3, 3, 3])`` with both
    heads; the data, heads and recipe are ``resnet56-cifar100mt``'s.

    Hypothesis:
      A shallower trunk has less capacity to share between the two tasks,
      so the unweighted sum is the bar where interference costs most.

    References:
      https://arxiv.org/abs/1512.03385
        He et al. 2016, "Deep Residual Learning for Image Recognition."
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``resnet20-cifar100mt`` of ``cv-multitask-loss`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp004"
    assert isinstance(cfg.step.model, TwoHeadResNet.Config)
    trunk = cfg.step.model.trunk
    assert isinstance(trunk, ResNet.Config)
    trunk.blocks_per_stage = (3, 3, 3)
    return cfg


def exp005() -> MultiTaskTrainLoop:
    """exp004 + uncertainty weighting with learned log-variances.

    Hypothesis:
      As exp001, on the shallower trunk.

    References:
      https://arxiv.org/abs/1705.07115
        Kendall, Gal and Cipolla 2018, "Multi-Task Learning Using Uncertainty
        to Weigh Losses for Scene Geometry and Semantics."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``uncertainty`` of ``cv-multitask-loss`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"
    cfg.step.mtl_loss = UncertaintyLoss.Config()
    return cfg


def exp006() -> MultiTaskTrainLoop:
    """exp004 + dynamic weight average at temperature 2.

    Hypothesis:
      As exp002, on the shallower trunk.

    References:
      https://arxiv.org/abs/1803.10704
        Liu, Johns and Davison 2019, "End-to-End Multi-Task Learning with
        Attention."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``dwa`` of ``cv-multitask-loss`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp006"
    cfg.step.mtl_loss = DynamicWeightAverage.Config()
    return cfg


def exp007() -> MultiTaskTrainLoop:
    """exp004 + PCGrad gradient surgery over every shared parameter.

    Hypothesis:
      As exp003, on the shallower trunk; the coarse head is again never
      updated.

    References:
      https://arxiv.org/abs/2001.06782
        Yu et al. 2020, "Gradient Surgery for Multi-Task Learning."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``pcgrad`` of ``cv-multitask-loss`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485. Its walk of the graph skips
        a heap-dependent subset of the parameters; this takes all of them.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp007"
    cfg.step.mtl_loss = PCGrad.Config()
    return cfg


def exp008() -> MultiTaskTrainLoop:
    """exp000 at the hidden ``vgg16bn-cifar100mt`` setting: a VGG-16-BN trunk, the sum.

    The script's VGG gives each task a whole classifier, ``Linear(512, 512)``,
    ReLU, dropout and the last linear layer, fine then coarse; the added head
    is the trunk's own classifier at 20 classes. The script keeps
    ``CosineAnnealingLR`` for VGG here, unlike ``dl-regularization``.

    Hypothesis:
      Two deep classifiers over a plain stack leave the tasks less to share,
      so the sum is the bar on the network least like the scored ones.

    References:
      https://arxiv.org/abs/1409.1556
        Simonyan and Zisserman 2015, "Very Deep Convolutional Networks for
        Large-Scale Image Recognition."
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``vgg16bn-cifar100mt`` of ``cv-multitask-loss`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp008"
    assert isinstance(cfg.step.model, TwoHeadResNet.Config)
    trunk = settings.vgg16bn_cifar100(experiments.exp000()).step.model
    assert isinstance(trunk, VGG16BN.Config)
    cfg.step.model.trunk = trunk
    return cfg


def exp009() -> MultiTaskTrainLoop:
    """exp008 + uncertainty weighting with learned log-variances.

    Hypothesis:
      As exp001, with a classifier per task.

    References:
      https://arxiv.org/abs/1705.07115
        Kendall, Gal and Cipolla 2018, "Multi-Task Learning Using Uncertainty
        to Weigh Losses for Scene Geometry and Semantics."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``uncertainty`` of ``cv-multitask-loss`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp009"
    cfg.step.mtl_loss = UncertaintyLoss.Config()
    return cfg


def exp010() -> MultiTaskTrainLoop:
    """exp008 + dynamic weight average at temperature 2.

    Hypothesis:
      As exp002, with a classifier per task.

    References:
      https://arxiv.org/abs/1803.10704
        Liu, Johns and Davison 2019, "End-to-End Multi-Task Learning with
        Attention."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``dwa`` of ``cv-multitask-loss`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp010"
    cfg.step.mtl_loss = DynamicWeightAverage.Config()
    return cfg


def exp011() -> MultiTaskTrainLoop:
    """exp008 + PCGrad gradient surgery over every shared parameter.

    Hypothesis:
      As exp003; the fine loss's leaves are the features and the fine
      classifier, so the whole coarse classifier stays at its initialization.

    References:
      https://arxiv.org/abs/2001.06782
        Yu et al. 2020, "Gradient Surgery for Multi-Task Learning."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``pcgrad`` of ``cv-multitask-loss`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485. Its walk of the graph skips
        a heap-dependent subset of the parameters; this takes all of them.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp011"
    cfg.step.mtl_loss = PCGrad.Config()
    return cfg
