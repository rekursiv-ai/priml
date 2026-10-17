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

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.pytorch_vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.initialization.experiments.exp000

The task is pinned to MLS-Bench commit ``80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.baselines.mlsbench.initialization.initializers import (
    fixup,
    orthogonal_relu,
    small_normal,
)
from priml.baselines.mlsbench.pytorch_vision import experiments as vision
from priml.baselines.mlsbench.pytorch_vision.model import (
    ResNet,
    kaiming_normal_fan_in,
    kaiming_normal_fan_out,
)


if TYPE_CHECKING:
    from priml.baselines.mlsbench.pytorch_vision.experiments import VisionTrainLoop


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
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = vision.exp000()
    cfg.study_name = "dl_weight_initialization"
    cfg.experiment_name = "exp000"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.init_conv = small_normal
    cfg.step.model.init_head = small_normal
    return cfg


def exp001() -> VisionTrainLoop:
    """exp000 + Kaiming normal: fan-out for convolutions, fan-in for the head.

    Hypothesis:
      Drawing each weight with variance ``2 / fan`` keeps the activations'
      scale constant through ReLU layers, which should train faster than a
      fixed 0.01 that ignores each layer's width.

    References:
      https://arxiv.org/abs/1502.01852
        He et al. 2015, "Delving Deep into Rectifiers: Surpassing Human-Level
        Performance on ImageNet Classification."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``kaiming_normal`` of ``dl-weight-initialization`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

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
      Scaling each block's last convolution by ``L ** -0.5`` and zeroing the
      normalization after it starts every block as the identity, so the
      variance of a 27-block network does not grow with depth. The Kaiming
      draw is Fixup's base, so the two are one change.

    References:
      https://arxiv.org/abs/1901.09321
        Zhang et al. 2019, "Fixup Initialization: Residual Learning Without
        Normalization."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``fixup`` of ``dl-weight-initialization`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

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
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"
    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.init_conv = orthogonal_relu
    cfg.step.model.init_head = orthogonal_relu
    return cfg
