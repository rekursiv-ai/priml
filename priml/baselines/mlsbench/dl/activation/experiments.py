r"""MLS-Bench ``dl-activation-function``: ResNet-20 on CIFAR-10, the activation as a slot.

``exp000`` is the task's pristine recipe, ReLU, at its ``resnet20-cifar10``
setting; the task has no ``resnet56-cifar100`` setting. Every later experiment
forks it and changes ``step.model.activation``, the one function the task lets
an edit touch. The network uses it after the stem and twice in every block.

    exp000  ReLU
      +-- exp001  GELU
      +-- exp002  SiLU
      +-- exp003  Mish
      +-- exp004  VGG-16-BN on CIFAR-100, ReLU
      |     +-- exp005  GELU
      |     +-- exp006  SiLU
      |     +-- exp007  Mish
      +-- exp008  MobileNetV2 on Fashion-MNIST, ReLU
            +-- exp009  GELU
            +-- exp010  SiLU
            +-- exp011  Mish

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.activation.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.activation.experiments.exp000

The task is pinned to MLS-Bench commit ``80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.baselines.mlsbench.dl.activation.activations import gelu, mish
from priml.baselines.mlsbench.dl.vision import experiments, settings
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.model.swiglu import relu, silu


if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


def exp000() -> VisionTrainLoop:
    """ResNet-20 on CIFAR-10 with ReLU, cross-entropy, SGD with cosine annealing.

    The pristine recipe of the task's ``resnet20-cifar10`` setting. Its script
    is the ``cv-classification-loss`` script with every activation made a
    ``CustomActivation`` that applies ReLU, so this is that task's ``exp000``
    at ResNet-20 depth on CIFAR-10. Frozen: improvements belong in a fork.

    Hypothesis:
      ReLU is the bar an activation must clear to earn its complexity.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``dl-activation-function`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.
      https://arxiv.org/abs/1512.03385
        He et al. 2016, "Deep Residual Learning for Image Recognition."

    Results:
      TBD.

    """
    cfg = settings.resnet20_cifar10(experiments.exp000())
    cfg.study_name = "dl_activation_function"
    cfg.experiment_name = "exp000"
    return cfg


def exp001() -> VisionTrainLoop:
    """exp000 + GELU in place of every ReLU.

    Hypothesis:
      Weighting each input by the normal CDF rather than gating it at zero
      passes a small gradient for negative inputs and is smooth at the
      origin, which should help the gradient through 19 activations.

    References:
      https://arxiv.org/abs/1606.08415
        Hendrycks and Gimpel 2016, "Gaussian Error Linear Units (GELUs)."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``gelu`` of ``dl-activation-function`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.activation = gelu
    return cfg


def exp002() -> VisionTrainLoop:
    """exp000 + SiLU in place of every ReLU.

    Hypothesis:
      Gating each input by its own sigmoid is smooth and non-monotonic,
      which the automated search that found it reports as a gain over ReLU
      on image classification.

    References:
      https://arxiv.org/abs/1710.05941
        Ramachandran et al. 2017, "Searching for Activation Functions."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``silu`` of ``dl-activation-function`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.activation = silu
    return cfg


def exp003() -> VisionTrainLoop:
    """exp000 + Mish in place of every ReLU.

    Hypothesis:
      ``x * tanh(softplus(x))`` is smooth, unbounded above and bounded below
      like SiLU, with a smoother loss landscape that its author reports as a
      gain on CIFAR-10.

    References:
      https://arxiv.org/abs/1908.08681
        Misra 2020, "Mish: A Self Regularized Non-Monotonic Activation
        Function."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``mish`` of ``dl-activation-function`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.activation = mish
    return cfg


def exp004() -> VisionTrainLoop:
    """exp000 at the ``vgg16bn-cifar100`` setting: VGG-16-BN on CIFAR-100, ReLU.

    The script's VGG applies its ``CustomActivation`` after every one of the
    13 normalizations and in the classifier, which is the network's default
    ReLU, so the switch alone is the pristine recipe.

    Hypothesis:
      Without shortcuts every activation sits on the only gradient path, so
      ReLU on VGG is the bar where the activation matters most.

    References:
      https://arxiv.org/abs/1409.1556
        Simonyan and Zisserman 2015, "Very Deep Convolutional Networks for
        Large-Scale Image Recognition."
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``vgg16bn-cifar100`` of ``dl-activation-function`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = settings.vgg16bn_cifar100(exp000())
    cfg.experiment_name = "exp004"
    return cfg


def exp005() -> VisionTrainLoop:
    """exp004 + GELU in place of every ReLU.

    Hypothesis:
      As exp001, on 14 activations with no shortcut around them.

    References:
      https://arxiv.org/abs/1606.08415
        Hendrycks and Gimpel 2016, "Gaussian Error Linear Units (GELUs)."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``gelu`` of ``dl-activation-function`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"
    assert isinstance(cfg.step.model, VGG16BN.Config)
    cfg.step.model.activation = gelu
    return cfg


def exp006() -> VisionTrainLoop:
    """exp004 + SiLU in place of every ReLU.

    Hypothesis:
      As exp002, on a plain deep network.

    References:
      https://arxiv.org/abs/1710.05941
        Ramachandran et al. 2017, "Searching for Activation Functions."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``silu`` of ``dl-activation-function`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp006"
    assert isinstance(cfg.step.model, VGG16BN.Config)
    cfg.step.model.activation = silu
    return cfg


def exp007() -> VisionTrainLoop:
    """exp004 + Mish in place of every ReLU.

    Hypothesis:
      As exp003, on a plain deep network.

    References:
      https://arxiv.org/abs/1908.08681
        Misra 2020, "Mish: A Self Regularized Non-Monotonic Activation
        Function."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``mish`` of ``dl-activation-function`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp007"
    assert isinstance(cfg.step.model, VGG16BN.Config)
    cfg.step.model.activation = mish
    return cfg


def exp008() -> VisionTrainLoop:
    """exp000 at the ``mobilenetv2-fmnist`` setting: MobileNetV2, ReLU.

    The task's script builds MobileNetV2 with its ``CustomActivation``, ReLU,
    where the network's own default is ReLU6, so the pristine rule is set on
    the switched network: ``activation = relu``.

    Hypothesis:
      Unclipped ReLU in the expansion layers is the bar on MobileNetV2 here;
      the script never uses ReLU6.

    References:
      https://arxiv.org/abs/1801.04381
        Sandler et al. 2018, "MobileNetV2: Inverted Residuals and Linear
        Bottlenecks."
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``mobilenetv2-fmnist`` of ``dl-activation-function`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = settings.mobilenetv2_fmnist(exp000())
    cfg.experiment_name = "exp008"
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    cfg.step.model.activation = relu
    return cfg


def exp009() -> VisionTrainLoop:
    """exp008 + GELU in place of every ReLU.

    Hypothesis:
      As exp001; GELU passes a gradient through the negative inputs a
      depthwise filter's few weights produce.

    References:
      https://arxiv.org/abs/1606.08415
        Hendrycks and Gimpel 2016, "Gaussian Error Linear Units (GELUs)."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``gelu`` of ``dl-activation-function`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp009"
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    cfg.step.model.activation = gelu
    return cfg


def exp010() -> VisionTrainLoop:
    """exp008 + SiLU in place of every ReLU.

    Hypothesis:
      As exp002; SiLU is the activation of later efficient networks built
      from inverted residuals.

    References:
      https://arxiv.org/abs/1710.05941
        Ramachandran et al. 2017, "Searching for Activation Functions."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``silu`` of ``dl-activation-function`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp010"
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    cfg.step.model.activation = silu
    return cfg


def exp011() -> VisionTrainLoop:
    """exp008 + Mish in place of every ReLU.

    Hypothesis:
      As exp003, on MobileNetV2.

    References:
      https://arxiv.org/abs/1908.08681
        Misra 2020, "Mish: A Self Regularized Non-Monotonic Activation
        Function."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``mish`` of ``dl-activation-function`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp011"
    assert isinstance(cfg.step.model, MobileNetV2.Config)
    cfg.step.model.activation = mish
    return cfg
