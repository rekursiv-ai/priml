r"""MLS-Bench ``dl-weight-initialization``: ResNet-56 on CIFAR-100, init as slots.

``exp000`` is the task's pristine recipe, every weight drawn from a normal with
standard deviation 0.01. Every later experiment forks it and changes the
network's initialization, the one function the task lets an edit touch: the
per-tensor ``step.model.init_conv`` and ``init_head``, and ``step.model.init``
for a rule that needs the network's structure.

    exp000  normal, std 0.01
      +-- exp001  Kaiming normal
      +-- exp002  Fixup: Kaiming, then shrink every residual branch
      +-- exp003  orthogonal
      +-- exp004  VGG-16-BN on CIFAR-100, normal, std 0.01
      |     +-- exp005  Kaiming normal
      |     +-- exp006  Fixup, which on VGG is Kaiming normal alone
      |     +-- exp007  orthogonal
      +-- exp008  MobileNetV2 on Fashion-MNIST, normal, std 0.01
            +-- exp009  Kaiming normal
            +-- exp010  Fixup, which on MobileNetV2 is Kaiming normal alone
            +-- exp011  orthogonal

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.initialization.experiments.exp000

The task is pinned to MLS-Bench commit ``b873944969c53b01e4ee6e843c77767b8c702485``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.baselines.mlsbench.dl.initialization.initializers import (
    fixup,
    orthogonal_relu,
    small_normal,
)
from priml.baselines.mlsbench.dl.vision import experiments, settings
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import (
    ResNet,
    kaiming_normal_fan_in,
    kaiming_normal_fan_out,
)
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN


if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


def exp000() -> VisionTrainLoop:
    """ResNet-56 on CIFAR-100, every weight drawn from a normal with std 0.01.

    The pristine recipe of the task's ``resnet56-cifar100`` setting. Its script
    is the ``cv-classification-loss`` script with the initialization made
    editable, so this is that task's ``exp000`` with the convolutions and the
    classifier drawn at a fixed small scale instead of He's. Frozen:
    improvements belong in a fork.

    Hypothesis:
      A fixed, depth-blind scale is the bar an initialization must clear to
      earn its complexity; BatchNorm after every convolution should rescue
      most of it.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``dl-weight-initialization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = experiments.exp000()
    cfg.study_name = "dl_weight_initialization"
    cfg.experiment_name = "exp000"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.init_conv = small_normal
    cfg.step.model.init_head = small_normal
    return cfg


def exp001() -> VisionTrainLoop:
    """exp000 + Kaiming normal: fan-out for convolutions, fan-in for the head.

    Hypothesis:
      Drawing each weight with variance ``2 / fan`` keeps the gradients'
      scale (fan-out) and the logits' (fan-in) constant through ReLU layers, which should train faster than a
      fixed 0.01 that ignores each layer's width.

    References:
      https://arxiv.org/abs/1502.01852
        He et al. 2015, "Delving Deep into Rectifiers: Surpassing Human-Level
        Performance on ImageNet Classification."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``kaiming_normal`` of ``dl-weight-initialization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.init_conv = kaiming_normal_fan_out
    cfg.step.model.init_head = kaiming_normal_fan_in
    return cfg


def exp002() -> VisionTrainLoop:
    """exp000 + Fixup: Kaiming normal, then shrink every residual branch.

    Hypothesis:
      Zeroing the scale of each block's last normalization starts every
      residual branch at its bias, so a block passes on its activated
      shortcut and the variance of a 27-block network does not grow with
      depth. Behind batch norm the ``L ** -0.5`` scaling of the last
      convolution changes only that layer's effective learning rate. The
      Kaiming draw is Fixup's base, so the two are one change.

    References:
      https://arxiv.org/abs/1901.09321
        Zhang et al. 2019, "Fixup Initialization: Residual Learning Without
        Normalization."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``fixup`` of ``dl-weight-initialization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.init_conv = kaiming_normal_fan_out
    cfg.step.model.init_head = kaiming_normal_fan_in
    cfg.step.model.init = fixup
    return cfg


def exp003() -> VisionTrainLoop:
    """exp000 + orthogonal weights scaled by ReLU's gain.

    Hypothesis:
      An orthogonal matrix preserves every direction's norm, so gradients
      neither vanish nor explode through depth, which should help a deep
      network more than matching the variance alone.

    References:
      https://arxiv.org/abs/1312.6120
        Saxe et al. 2014, "Exact solutions to the nonlinear dynamics of
        learning in deep linear neural networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``orthogonal`` of ``dl-weight-initialization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.init_conv = orthogonal_relu
    cfg.step.model.init_head = orthogonal_relu
    return cfg


def exp004() -> VisionTrainLoop:
    """exp000 at the ``vgg16bn-cifar100`` setting: VGG-16-BN, normal with std 0.01.

    The switch builds a fresh network, so the pristine rule is set on it again:
    every convolution weight and both classifier weights from a normal with std
    0.01. The convolutions' biases keep torch's draw, as the script leaves them.

    Hypothesis:
      Without residual paths a fixed small scale shrinks the signal through
      13 convolutions; batch norm rescues the forward, so this is the bar on a
      plain deep network.

    References:
      https://arxiv.org/abs/1409.1556
        Simonyan and Zisserman 2015, "Very Deep Convolutional Networks for
        Large-Scale Image Recognition."
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``vgg16bn-cifar100`` of ``dl-weight-initialization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = settings.vgg16bn_cifar100(exp000())
    cfg.experiment_name = "exp004"
    assert isinstance(cfg.step.model, VGG16BN.Config)
    cfg.step.model.init_conv = small_normal
    cfg.step.model.init_head = small_normal
    return cfg


def exp005() -> VisionTrainLoop:
    """exp004 + Kaiming normal: fan-out for convolutions, fan-in for the classifier.

    Hypothesis:
      As exp001, on a network with no shortcut to carry the signal past a
      badly scaled layer.

    References:
      https://arxiv.org/abs/1502.01852
        He et al. 2015, "Delving Deep into Rectifiers: Surpassing Human-Level
        Performance on ImageNet Classification."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``kaiming_normal`` of ``dl-weight-initialization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"
    assert isinstance(cfg.step.model, VGG16BN.Config)
    cfg.step.model.init_conv = kaiming_normal_fan_out
    cfg.step.model.init_head = kaiming_normal_fan_in
    return cfg


def exp006() -> VisionTrainLoop:
    """exp004 + Fixup, which on VGG-16-BN is Kaiming normal alone.

    Upstream's ``fixup`` edit scales residual branches only when the
    architecture's name starts with ``resnet``; on VGG it runs its Kaiming
    phase and skips the rest. This fork is therefore exp005's configuration,
    kept as its own experiment because the task scores it as its own baseline.

    Hypothesis:
      With no residual branch to shrink, Fixup reduces to its Kaiming base
      and should match exp005.

    References:
      https://arxiv.org/abs/1901.09321
        Zhang et al. 2019, "Fixup Initialization: Residual Learning Without
        Normalization."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``fixup`` of ``dl-weight-initialization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp006"
    assert isinstance(cfg.step.model, VGG16BN.Config)
    cfg.step.model.init_conv = kaiming_normal_fan_out
    cfg.step.model.init_head = kaiming_normal_fan_in
    return cfg


def exp007() -> VisionTrainLoop:
    """exp004 + orthogonal weights scaled by ReLU's gain.

    Hypothesis:
      As exp003; a plain chain of 15 layers is where norm preservation
      through depth matters most.

    References:
      https://arxiv.org/abs/1312.6120
        Saxe et al. 2014, "Exact solutions to the nonlinear dynamics of
        learning in deep linear neural networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``orthogonal`` of ``dl-weight-initialization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp007"
    assert isinstance(cfg.step.model, VGG16BN.Config)
    cfg.step.model.init_conv = orthogonal_relu
    cfg.step.model.init_head = orthogonal_relu
    return cfg


def exp008() -> VisionTrainLoop:
    """exp000 at the ``mobilenetv2-fmnist`` setting: MobileNetV2, normal with std 0.01.

    The switch builds a fresh network, so the pristine rule is set on it again:
    every convolution weight, depthwise ones included, and the classifier
    weight from a normal with std 0.01. The task scores this setting hidden.

    Hypothesis:
      A depthwise filter has a fan of nine, so a fixed 0.01 is far further
      from He's scale there than on a dense convolution; this is the bar on
      MobileNetV2.

    References:
      https://arxiv.org/abs/1801.04381
        Sandler et al. 2018, "MobileNetV2: Inverted Residuals and Linear
        Bottlenecks."
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``mobilenetv2-fmnist`` of ``dl-weight-initialization`` at
        commit b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = settings.mobilenetv2_fmnist(exp000())
    cfg.experiment_name = "exp008"
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    cfg.step.model.init_conv = small_normal
    cfg.step.model.init_head = small_normal
    return cfg


def exp009() -> VisionTrainLoop:
    """exp008 + Kaiming normal: fan-out for convolutions, fan-in for the classifier.

    Hypothesis:
      As exp001; fan-out scaling sizes each depthwise filter by its own
      small fan.

    References:
      https://arxiv.org/abs/1502.01852
        He et al. 2015, "Delving Deep into Rectifiers: Surpassing Human-Level
        Performance on ImageNet Classification."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``kaiming_normal`` of ``dl-weight-initialization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp009"
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    cfg.step.model.init_conv = kaiming_normal_fan_out
    cfg.step.model.init_head = kaiming_normal_fan_in
    return cfg


def exp010() -> VisionTrainLoop:
    """exp008 + Fixup, which on MobileNetV2 is Kaiming normal alone.

    Upstream's ``fixup`` edit scales residual branches only when the
    architecture's name starts with ``resnet``; MobileNetV2's inverted
    residual blocks are left at its Kaiming phase. This fork is therefore
    exp009's configuration, kept as its own experiment because the task
    scores it as its own baseline.

    Hypothesis:
      Fixup's branch scaling never reaches MobileNetV2 here, so it should
      match exp009.

    References:
      https://arxiv.org/abs/1901.09321
        Zhang et al. 2019, "Fixup Initialization: Residual Learning Without
        Normalization."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``fixup`` of ``dl-weight-initialization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp010"
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    cfg.step.model.init_conv = kaiming_normal_fan_out
    cfg.step.model.init_head = kaiming_normal_fan_in
    return cfg


def exp011() -> VisionTrainLoop:
    """exp008 + orthogonal weights scaled by ReLU's gain.

    Hypothesis:
      As exp003; a depthwise filter flattens to a ``(channels, 9)`` matrix,
      so orthogonality there constrains each filter to a row of a
      semi-orthogonal set.

    References:
      https://arxiv.org/abs/1312.6120
        Saxe et al. 2014, "Exact solutions to the nonlinear dynamics of
        learning in deep linear neural networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``orthogonal`` of ``dl-weight-initialization`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp011"
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    cfg.step.model.init_conv = orthogonal_relu
    cfg.step.model.init_head = orthogonal_relu
    return cfg
