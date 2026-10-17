r"""MLS-Bench ``dl-regularization``: ResNet-56 on CIFAR-100, the regularizer as a slot.

``exp000`` is the task's pristine recipe, cross-entropy with nothing added.
Every later experiment forks it and changes ``step.regularization``, the one
function the task lets an edit touch.

    exp000  no regularization
      +-- exp001  spatial co-activation penalty on conv kernels (DropBlock-inspired)
      +-- exp002  confidence penalty
      +-- exp003  orthogonal regularization

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.pytorch_vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.regularization.experiments.exp000

The task is pinned to MLS-Bench commit ``80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import PartialConfig

from priml.baselines.mlsbench.pytorch_vision import experiments as vision
from priml.baselines.mlsbench.regularization.regularizers import (
    confidence_penalty,
    orthogonal,
    spatial_coactivation,
)
from priml.baselines.mlsbench.regularization.train_step import RegularizedTrainStep
from priml.optimizers.parameter_filter import everything, excluding, with_ndim


if TYPE_CHECKING:
    from priml.baselines.mlsbench.pytorch_vision.experiments import VisionTrainLoop


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
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = vision.exp000()
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
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

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
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

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
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

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
