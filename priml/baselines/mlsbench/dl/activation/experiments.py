r"""MLS-Bench ``dl-activation-function``: ResNet-20 on CIFAR-10, the activation as a slot.

``exp000`` is the task's pristine recipe, ReLU, at its ``resnet20-cifar10``
setting; the task has no ``resnet56-cifar100`` setting. Every later experiment
forks it and changes ``step.model.activation``, the one function the task lets
an edit touch. The network uses it after the stem and twice in every block.

    exp000  ReLU
      +-- exp001  GELU
      +-- exp002  SiLU
      +-- exp003  Mish

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.activation.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.activation.experiments.exp000

The task is pinned to MLS-Bench commit ``80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.baselines.mlsbench.dl.activation.activations import gelu, mish
from priml.baselines.mlsbench.dl.vision import experiments
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.model.swiglu import silu


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
    cfg = experiments.exp000()
    cfg.study_name = "dl_activation_function"
    cfg.experiment_name = "exp000"

    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.channels_out = 10
    cfg.step.model.blocks_per_stage = (3, 3, 3)

    cfg.dataset.working_dir = "/datasets/cifar10"
    cfg.dataset.mean = (0.4914, 0.4822, 0.4465)
    cfg.dataset.std = (0.2023, 0.1994, 0.2010)
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
