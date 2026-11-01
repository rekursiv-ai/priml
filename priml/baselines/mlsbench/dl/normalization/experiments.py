r"""MLS-Bench ``dl-normalization``: ResNet-56 on CIFAR-100, the norm layer as a slot.

``exp000`` is the task's pristine recipe, batch normalization. Every later
experiment forks it and changes ``step.model.norm``, the one class the task
lets an edit touch; the model builds the stem's, every block's and every
projection's normalization from it.

    exp000  batch norm
      +-- exp001  group norm
      +-- exp002  batch-instance norm
      +-- exp003  switchable norm
      +-- exp004  ResNet-110 on CIFAR-100 (the hidden setting), batch norm
      |     +-- exp005  group norm
      |     +-- exp006  batch-instance norm
      |     +-- exp007  switchable norm
      +-- exp008  MobileNetV2 on Fashion-MNIST, batch norm
            +-- exp009  group norm
            +-- exp010  batch-instance norm
            +-- exp011  switchable norm

The task's script also has a ``vgg16bn-cifar100`` setting, but the task does
not score it, so it is not ported.

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.normalization.experiments.exp000

The task is pinned to MLS-Bench commit ``b873944969c53b01e4ee6e843c77767b8c702485``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.baselines.mlsbench.dl.normalization.norms import (
    BatchInstanceNorm,
    GroupNorm,
    SwitchableNorm,
)
from priml.baselines.mlsbench.dl.vision import experiments, settings
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet


if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


def exp000() -> VisionTrainLoop:
    """ResNet-56 on CIFAR-100, batch norm, cross-entropy, SGD with cosine annealing.

    The pristine recipe of the task's ``resnet56-cifar100`` setting. Its script
    is the ``cv-classification-loss`` script with every ``nn.BatchNorm2d``
    behind a ``CustomNorm`` that wraps one, so this is that task's ``exp000``.
    The wrapper only renames the state's keys, and the script's
    ``initialize_weights`` sets each norm to the scale and shift it is built
    with. Frozen: improvements belong in a fork.

    Hypothesis:
      Batch norm is the bar a normalization layer must clear.

    References:
      https://arxiv.org/abs/1502.03167
        Ioffe and Szegedy 2015, "Batch Normalization: Accelerating Deep Network
        Training by Reducing Internal Covariate Shift."
      https://github.com/Imbernoulli/MLS-Bench
        Task ``dl-normalization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = experiments.exp000()
    cfg.study_name = "dl_normalization"
    cfg.experiment_name = "exp000"
    return cfg


def exp001() -> VisionTrainLoop:
    """exp000 + group norm, up to 32 groups.

    Hypothesis:
      Statistics over channel groups of one image do not depend on the batch,
      so training and evaluation normalize alike, which group norm reports as
      matching batch norm on classification.

    References:
      https://arxiv.org/abs/1803.08494
        Wu and He 2018, "Group Normalization."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``group_norm`` of ``dl-normalization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.norm = GroupNorm
    return cfg


def exp002() -> VisionTrainLoop:
    """exp000 + batch-instance norm.

    Hypothesis:
      A learned per-channel gate between batch and instance statistics lets
      each channel drop the per-image contrast it does not need.

    References:
      https://arxiv.org/abs/1805.07925
        Name and Kim 2018, "Batch-Instance Normalization for Adaptively
        Style-Invariant Neural Networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``batch_instance_norm`` of ``dl-normalization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.norm = BatchInstanceNorm
    return cfg


def exp003() -> VisionTrainLoop:
    """exp000 + switchable norm.

    Hypothesis:
      Learning a mixture of instance, layer and batch statistics per layer
      picks the normalization each depth suits, which switchable norm reports
      as beating each of the three alone.

    References:
      https://arxiv.org/abs/1806.10779
        Luo et al. 2019, "Differentiable Learning-to-Normalize via Switchable
        Normalization."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``switchable_norm`` of ``dl-normalization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.norm = SwitchableNorm
    return cfg


def exp004() -> VisionTrainLoop:
    """exp000 at the hidden ``resnet110-cifar100`` setting: ResNet-110, batch norm.

    The task scores this setting without showing it: 18 blocks per stage
    instead of 9, everything else the ``resnet56-cifar100`` recipe.

    Hypothesis:
      A norm that holds up at 54 blocks and fails at 27 would show here;
      batch norm is the bar at this depth.

    References:
      https://arxiv.org/abs/1512.03385
        He et al. 2016, "Deep Residual Learning for Image Recognition."
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``resnet110-cifar100`` of ``dl-normalization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = settings.resnet110_cifar100(exp000())
    cfg.experiment_name = "exp004"
    return cfg


def exp005() -> VisionTrainLoop:
    """exp004 + group norm, up to 32 groups.

    Hypothesis:
      As exp001, at twice the depth, where batch statistics compound over
      more layers.

    References:
      https://arxiv.org/abs/1803.08494
        Wu and He 2018, "Group Normalization."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``group_norm`` of ``dl-normalization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.norm = GroupNorm
    return cfg


def exp006() -> VisionTrainLoop:
    """exp004 + batch-instance norm.

    Hypothesis:
      As exp002, at twice the depth.

    References:
      https://arxiv.org/abs/1805.07925
        Name and Kim 2018, "Batch-Instance Normalization for Adaptively
        Style-Invariant Neural Networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``batch_instance_norm`` of ``dl-normalization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp006"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.norm = BatchInstanceNorm
    return cfg


def exp007() -> VisionTrainLoop:
    """exp004 + switchable norm.

    Hypothesis:
      As exp003, at twice the depth: more layers to pick a mixture for.

    References:
      https://arxiv.org/abs/1806.10779
        Luo et al. 2019, "Differentiable Learning-to-Normalize via Switchable
        Normalization."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``switchable_norm`` of ``dl-normalization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp007"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.norm = SwitchableNorm
    return cfg


def exp008() -> VisionTrainLoop:
    """exp000 at the ``mobilenetv2-fmnist`` setting: MobileNetV2, batch norm.

    The script's MobileNetV2 builds every norm, the stem's, each inverted
    residual block's three and the last convolution's, as a ``CustomNorm``.

    Hypothesis:
      Depthwise convolutions normalize one channel at a time, so batch norm
      on MobileNetV2 is the bar a norm must clear on a network without
      cross-channel mixing in half its layers.

    References:
      https://arxiv.org/abs/1801.04381
        Sandler et al. 2018, "MobileNetV2: Inverted Residuals and Linear
        Bottlenecks."
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``mobilenetv2-fmnist`` of ``dl-normalization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = settings.mobilenetv2_fmnist(exp000())
    cfg.experiment_name = "exp008"
    return cfg


def exp009() -> VisionTrainLoop:
    """exp008 + group norm, up to 32 groups.

    Hypothesis:
      As exp001; MobileNetV2's widths (16 to 1280, with 24, 96, 144 and 960)
      exercise group counts below 32.

    References:
      https://arxiv.org/abs/1803.08494
        Wu and He 2018, "Group Normalization."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``group_norm`` of ``dl-normalization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp009"
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    cfg.step.model.norm = GroupNorm
    return cfg


def exp010() -> VisionTrainLoop:
    """exp008 + batch-instance norm.

    Hypothesis:
      As exp002, on grayscale images with less style to drop.

    References:
      https://arxiv.org/abs/1805.07925
        Name and Kim 2018, "Batch-Instance Normalization for Adaptively
        Style-Invariant Neural Networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``batch_instance_norm`` of ``dl-normalization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp010"
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    cfg.step.model.norm = BatchInstanceNorm
    return cfg


def exp011() -> VisionTrainLoop:
    """exp008 + switchable norm.

    Hypothesis:
      As exp003, on a network whose depthwise layers may prefer other
      statistics than its pointwise ones.

    References:
      https://arxiv.org/abs/1806.10779
        Luo et al. 2019, "Differentiable Learning-to-Normalize via Switchable
        Normalization."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``switchable_norm`` of ``dl-normalization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp011"
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    cfg.step.model.norm = SwitchableNorm
    return cfg
