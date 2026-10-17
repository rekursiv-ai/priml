r"""MLS-Bench ``dl-residual-connection``: ResNet-56 on CIFAR-100, the block as a slot.

``exp000`` is the task's pristine recipe, the post-activation basic block. Every
later experiment forks it and changes ``step.model.block``, the one class the
task lets an edit touch.

    exp000  basic block
      +-- exp001  pre-activation, branch scaled by a learned scalar from 0.1
      +-- exp002  basic block, branch gated by a learned scalar from 1
      +-- exp003  stochastic depth, deepest block survives with probability 0.5

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.pytorch_vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.residual.experiments.exp000

The task is pinned to MLS-Bench commit ``80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.baselines.mlsbench.pytorch_vision import experiments as vision
from priml.baselines.mlsbench.pytorch_vision.model import ResNet
from priml.baselines.mlsbench.residual.blocks import (
    GatedResidualBlock,
    PreActivationBlock,
    stochastic_depth,
)


if TYPE_CHECKING:
    from priml.baselines.mlsbench.pytorch_vision.experiments import VisionTrainLoop


def exp000() -> VisionTrainLoop:
    """ResNet-56 on CIFAR-100, post-activation basic blocks, SGD with cosine annealing.

    The pristine recipe of the task's ``resnet56-cifar100`` setting. Its script
    is the ``cv-classification-loss`` script with the block made the editable
    class and the loss an ``nn.CrossEntropyLoss``, which computes what that
    task's ``cross_entropy`` does; so this is that task's ``exp000``. Frozen:
    improvements belong in a fork.

    Hypothesis:
      The basic block is the bar a residual design must clear to earn its
      complexity.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``dl-residual-connection`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.
      https://arxiv.org/abs/1512.03385
        He et al. 2016, "Deep Residual Learning for Image Recognition."

    Results:
      TBD.

    """
    cfg = vision.exp000()
    cfg.study_name = "dl_residual_connection"
    cfg.experiment_name = "exp000"
    return cfg


def exp001() -> VisionTrainLoop:
    """exp000 + pre-activation blocks, the branch scaled by a learned scalar from 0.1.

    Hypothesis:
      Normalizing and activating before each convolution leaves the shortcut a
      clean identity, and starting the branch at a tenth of its weight keeps a
      56-layer network near the identity early, so gradients reach the stem
      undamped.

    References:
      https://arxiv.org/abs/1603.05027
        He et al. 2016, "Identity Mappings in Deep Residual Networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``pre_activation`` of ``dl-residual-connection`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"

    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.block = PreActivationBlock.Config()
    return cfg


def exp002() -> VisionTrainLoop:
    """exp000 + a learned scalar gate on each block's branch, starting at 1.

    Hypothesis:
      A per-block gate lets training shrink a branch that hurts and grow one
      that helps, while starting exactly at the basic block.

    References:
      https://arxiv.org/abs/2003.04887
        Bachlechner et al. 2021, "ReZero is All You Need: Fast Convergence at
        Large Depth."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``gated_residual`` of ``dl-residual-connection`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"

    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.block = GatedResidualBlock.Config()
    return cfg


def exp003() -> VisionTrainLoop:
    """exp000 + stochastic depth, survival decaying linearly to 0.5 at the deepest block.

    One block config per position, since each position's survival differs. The
    positions follow the reference's counter, which restarts on every
    first-stage block: all nine there sit at position 1, the rest at 2 to 19.

    Hypothesis:
      Dropping whole branches at random trains an implicit ensemble of
      shallower networks, which regularizes a 56-layer network that fits
      CIFAR-100's training split.

    References:
      https://arxiv.org/abs/1603.09382
        Huang et al. 2016, "Deep Networks with Stochastic Depth."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``stochastic_depth`` of ``dl-residual-connection`` at commit
        80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"

    model = cfg.step.model
    assert isinstance(model, ResNet.Config)
    model.block = stochastic_depth(
        model.channels_hidden,
        model.blocks_per_stage,
        survival_last=0.5,
    )
    return cfg
