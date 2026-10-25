r"""MLS-Bench ``dl-regularization``: ResNet-56 on CIFAR-100, the regularizer as a slot.

``exp000`` is the task's pristine recipe, cross-entropy with nothing added.
Every later experiment forks it and changes ``step.regularization``, the one
function the task lets an edit touch.

    exp000  no regularization
      +-- exp001  spatial co-activation penalty on conv kernels (DropBlock-inspired)
      +-- exp002  confidence penalty
      +-- exp003  orthogonal regularization
      +-- exp004  VGG-16-BN on CIFAR-100, warmup then cosine, no regularization
      |     +-- exp005  spatial co-activation penalty
      |     +-- exp006  confidence penalty
      |     +-- exp007  orthogonal regularization (selects nothing on VGG)
      +-- exp008  MobileNetV2 on Fashion-MNIST (the hidden setting), no regularization
            +-- exp009  spatial co-activation penalty
            +-- exp010  confidence penalty
            +-- exp011  orthogonal regularization

The script also has a ``resnet20_cifar10.sh``, but the task does not run it,
so it is not ported.

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.regularization.experiments.exp000

The task is pinned to MLS-Bench commit ``b873944969c53b01e4ee6e843c77767b8c702485``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import PartialConfig

from priml.baselines.mlsbench.dl.regularization.regularizers import (
    confidence_penalty,
    orthogonal,
    spatial_coactivation,
)
from priml.baselines.mlsbench.dl.regularization.schedules import (
    warmup_cosine_factor,
)
from priml.baselines.mlsbench.dl.regularization.train_step import (
    RegularizedTrainStep,
)
from priml.baselines.mlsbench.dl.vision import experiments, settings
from priml.optimizers.parameter_filter import (
    complement,
    everything,
    excluding,
    with_ndim,
)


if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


def exp000() -> VisionTrainLoop:
    """ResNet-56 on CIFAR-100, cross-entropy plus zero, SGD with cosine annealing.

    The pristine recipe of the task's ``resnet56-cifar100`` setting. Its script
    is the ``cv-classification-loss`` script with the loss fixed to
    cross-entropy and a regularizer added; the setting passes the same
    ``--epochs``, ``--batch-size``, ``--lr``, ``--momentum`` and
    ``--weight-decay 5e-4`` that task uses, overriding the script's own
    ``weight-decay`` default of 0. So this is that task's ``exp000`` on the step
    with a regularization slot, left at zero: the pristine body raises, and
    its docstring names zero as "no regularization". Frozen: improvements
    belong in a fork.

    Hypothesis:
      Weight decay alone is the bar an added penalty must clear.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``dl-regularization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = experiments.exp000()
    cfg.step = RegularizedTrainStep.Config().update(cfg.step)
    cfg.study_name = "dl_regularization"
    cfg.experiment_name = "exp000"
    return cfg


def exp001() -> VisionTrainLoop:
    """exp000 + a penalty on the energy of 3x3 blocks of every conv kernel.

    Hypothesis:
      DropBlock stops a network relying on contiguous regions; penalizing the
      local energy of each kernel, off for the first fifth of training and
      then ramping to 1e-4, asks the same of the weights without changing
      the graph.

    References:
      https://arxiv.org/abs/1810.12890
        Ghiasi, Lin and Le 2018, "DropBlock: A regularization method for
        convolutional networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``dropblock`` of ``dl-regularization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    assert isinstance(cfg.step, RegularizedTrainStep.Config)
    cfg.experiment_name = "exp001"
    reg = cfg.step.regularization = PartialConfig(spatial_coactivation)
    reg.block_size = 3
    reg.strength = 1e-4
    reg.delay = 0.2
    return cfg


def exp002() -> VisionTrainLoop:
    """exp000 + confidence penalty, beta 0.1.

    Hypothesis:
      Rewarding the entropy of the predicted distribution keeps a 56-layer
      network from becoming over-confident on CIFAR-100's training split.

    References:
      https://arxiv.org/abs/1701.06548
        Pereyra et al. 2017, "Regularizing Neural Networks by Penalizing
        Confident Output Distributions."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``confidence_penalty`` of ``dl-regularization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    assert isinstance(cfg.step, RegularizedTrainStep.Config)
    cfg.experiment_name = "exp002"
    reg = cfg.step.regularization = PartialConfig(confidence_penalty)
    reg.beta = 0.1
    reg.eps = 1e-8
    return cfg


def exp003() -> VisionTrainLoop:
    """exp000 + orthogonal regularization, 1e-4, on the stem and block convolutions.

    Hypothesis:
      Pulling each convolution's filters towards an orthonormal set preserves
      signal norms through 56 layers. The reference selects weights whose
      name holds ``conv``, which is the stem and both convolutions of every
      block but not a projection shortcut.

    References:
      https://arxiv.org/abs/1609.07093
        Brock et al. 2017, "Neural Photo Editing with Introspective
        Adversarial Networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``orthogonal_reg`` of ``dl-regularization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    assert isinstance(cfg.step, RegularizedTrainStep.Config)
    cfg.experiment_name = "exp003"
    reg = cfg.step.regularization = PartialConfig(orthogonal)
    reg.strength = 1e-4
    reg.select = with_ndim(excluding(everything, "shortcut"), 4)
    return cfg


def exp004() -> VisionTrainLoop:
    """exp000 at the ``vgg16bn-cifar100`` setting: VGG-16-BN, warmup, no regularization.

    The script gives an ``--arch`` starting ``vgg`` its own schedule: five
    epochs of linear warmup, then half a cosine over the remaining 195, as a
    ``LambdaLR`` factor. Every other network keeps ``CosineAnnealingLR``. The
    schedule travels with the setting, so this fork changes the network, the
    dataset and the schedule together.

    Hypothesis:
      VGG has no residual path, so weight decay and warmup alone are the bar a
      penalty must clear on a plain deep stack.

    References:
      https://arxiv.org/abs/1409.1556
        Simonyan and Zisserman 2015, "Very Deep Convolutional Networks for
        Large-Scale Image Recognition."
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``vgg16bn-cifar100`` of ``dl-regularization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = settings.vgg16bn_cifar100(exp000())
    cfg.experiment_name = "exp004"
    schedule = cfg.step.schedule = PartialConfig(warmup_cosine_factor)
    schedule.warmup_epochs = 5
    return cfg


def exp005() -> VisionTrainLoop:
    """exp004 + the spatial co-activation penalty of exp001.

    Hypothesis:
      As exp001; every one of VGG's thirteen convolutions is 3x3, so the
      penalty covers the whole network rather than ResNet's stem and blocks.

    References:
      https://arxiv.org/abs/1810.12890
        Ghiasi, Lin and Le 2018, "DropBlock: A regularization method for
        convolutional networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``dropblock`` of ``dl-regularization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    assert isinstance(cfg.step, RegularizedTrainStep.Config)
    cfg.experiment_name = "exp005"
    reg = cfg.step.regularization = PartialConfig(spatial_coactivation)
    reg.block_size = 3
    reg.strength = 1e-4
    reg.delay = 0.2
    return cfg


def exp006() -> VisionTrainLoop:
    """exp004 + confidence penalty, beta 0.1.

    Hypothesis:
      As exp002, on a network whose classifier carries dropout of its own.

    References:
      https://arxiv.org/abs/1701.06548
        Pereyra et al. 2017, "Regularizing Neural Networks by Penalizing
        Confident Output Distributions."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``confidence_penalty`` of ``dl-regularization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    assert isinstance(cfg.step, RegularizedTrainStep.Config)
    cfg.experiment_name = "exp006"
    reg = cfg.step.regularization = PartialConfig(confidence_penalty)
    reg.beta = 0.1
    reg.eps = 1e-8
    return cfg


def exp007() -> VisionTrainLoop:
    """exp004 + orthogonal regularization, 1e-4, on no parameter at all.

    The baseline penalizes 4-D weights whose name holds ``conv``. The script's
    VGG names its convolutions ``features.<index>``, so the published
    baseline selects nothing on this network and its term is zero: the run
    is exp004 with a zero added. The selector states that as a value.

    Hypothesis:
      None beyond exp004's: the published ``orthogonal_reg`` row at this
      setting measures an unregularized VGG.

    References:
      https://arxiv.org/abs/1609.07093
        Brock et al. 2017, "Neural Photo Editing with Introspective
        Adversarial Networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``orthogonal_reg`` of ``dl-regularization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    assert isinstance(cfg.step, RegularizedTrainStep.Config)
    cfg.experiment_name = "exp007"
    reg = cfg.step.regularization = PartialConfig(orthogonal)
    reg.strength = 1e-4
    reg.select = complement(everything)
    return cfg


def exp008() -> VisionTrainLoop:
    """exp000 at the hidden ``mobilenetv2-fmnist`` setting: MobileNetV2, no regularization.

    Hypothesis:
      Weight decay alone is the bar on a network of depthwise and pointwise
      convolutions over grayscale images.

    References:
      https://arxiv.org/abs/1801.04381
        Sandler et al. 2018, "MobileNetV2: Inverted Residuals and Linear
        Bottlenecks."
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``mobilenetv2-fmnist`` of ``dl-regularization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = settings.mobilenetv2_fmnist(exp000())
    cfg.experiment_name = "exp008"
    return cfg


def exp009() -> VisionTrainLoop:
    """exp008 + the spatial co-activation penalty of exp001.

    Hypothesis:
      As exp001; only the stem and the depthwise convolutions are 3x3, so the
      penalty reaches the spatial filters and none of the pointwise ones.

    References:
      https://arxiv.org/abs/1810.12890
        Ghiasi, Lin and Le 2018, "DropBlock: A regularization method for
        convolutional networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``dropblock`` of ``dl-regularization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    assert isinstance(cfg.step, RegularizedTrainStep.Config)
    cfg.experiment_name = "exp009"
    reg = cfg.step.regularization = PartialConfig(spatial_coactivation)
    reg.block_size = 3
    reg.strength = 1e-4
    reg.delay = 0.2
    return cfg


def exp010() -> VisionTrainLoop:
    """exp008 + confidence penalty, beta 0.1.

    Hypothesis:
      As exp002, on ten classes, where the entropy's ceiling is lower.

    References:
      https://arxiv.org/abs/1701.06548
        Pereyra et al. 2017, "Regularizing Neural Networks by Penalizing
        Confident Output Distributions."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``confidence_penalty`` of ``dl-regularization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    assert isinstance(cfg.step, RegularizedTrainStep.Config)
    cfg.experiment_name = "exp010"
    reg = cfg.step.regularization = PartialConfig(confidence_penalty)
    reg.beta = 0.1
    reg.eps = 1e-8
    return cfg


def exp011() -> VisionTrainLoop:
    """exp008 + orthogonal regularization, 1e-4, on every convolution.

    Hypothesis:
      As exp003. The script names every MobileNetV2 convolution with ``conv``
      (``conv1``, each block's ``conv``, ``conv_last``), so the baseline takes
      all 4-D weights, depthwise ones included, whose rows are 9-wide.

    References:
      https://arxiv.org/abs/1609.07093
        Brock et al. 2017, "Neural Photo Editing with Introspective
        Adversarial Networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``orthogonal_reg`` of ``dl-regularization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    assert isinstance(cfg.step, RegularizedTrainStep.Config)
    cfg.experiment_name = "exp011"
    reg = cfg.step.regularization = PartialConfig(orthogonal)
    reg.strength = 1e-4
    reg.select = with_ndim(everything, 4)
    return cfg
