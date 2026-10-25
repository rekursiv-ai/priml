r"""MLS-Bench ``dl-residual-connection``: CIFAR ResNets, the block as a slot.

``exp000`` is the task's pristine recipe at its ``resnet56-cifar100`` setting,
the post-activation basic block. ``exp004`` and ``exp008`` move it to the
task's other settings, changing only depth and dataset. Every other experiment
forks one of the three and changes ``step.model.block``, the one class the task
lets an edit touch.

    exp000  ResNet-56, CIFAR-100, basic block       (resnet56-cifar100)
      +-- exp001  pre-activation, branch scaled by a learned scalar from 0.1
      +-- exp002  basic block, branch gated by a learned scalar from 1
      +-- exp003  stochastic depth, deepest block survives with probability 0.5
      +-- exp004  ResNet-20, CIFAR-10                (resnet20-cifar10)
      |     +-- exp005  pre-activation
      |     +-- exp006  gated residual
      |     +-- exp007  stochastic depth
      +-- exp008  ResNet-110, CIFAR-100              (resnet110-cifar100, hidden)
            +-- exp009  pre-activation
            +-- exp010  gated residual
            +-- exp011  stochastic depth

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.residual.experiments.exp000

The task is pinned to MLS-Bench commit ``b873944969c53b01e4ee6e843c77767b8c702485``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from priml.baselines.mlsbench.dl.residual.blocks import (
    GatedResidualBlock,
    PreActivationBlock,
    StochasticDepthBlock,
)
from priml.baselines.mlsbench.dl.vision import experiments, settings
from priml.baselines.mlsbench.dl.vision.model import ResNet


if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


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
        b873944969c53b01e4ee6e843c77767b8c702485.
      https://arxiv.org/abs/1512.03385
        He et al. 2016, "Deep Residual Learning for Image Recognition."

    Results:
      TBD.

    """
    cfg = experiments.exp000()
    cfg.study_name = "dl_residual_connection"
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
        b873944969c53b01e4ee6e843c77767b8c702485.

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
        b873944969c53b01e4ee6e843c77767b8c702485.

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

    The template numbers its copies as the reference's counter does, which
    restarts on every first-stage block: all nine there sit at position 1, the
    rest at 2 to 19.

    Hypothesis:
      Dropping whole branches at random trains an implicit ensemble of
      shallower networks, which regularizes a 56-layer network that fits
      CIFAR-100's training split.

    References:
      https://arxiv.org/abs/1603.09382
        Huang et al. 2016, "Deep Networks with Stochastic Depth."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``stochastic_depth`` of ``dl-residual-connection`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"

    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.block = StochasticDepthBlock.Config()
    return cfg


def exp004() -> VisionTrainLoop:
    """exp000 at the resnet20-cifar10 setting: ResNet-20 on CIFAR-10, basic blocks.

    Only depth, classes and the dataset change; the recipe is ``exp000``'s.

    Hypothesis:
      The basic block at ResNet-20 depth is the bar a residual design must
      clear where depth is too shallow for gradient flow to bind.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``resnet20-cifar10`` of ``dl-residual-connection`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = settings.resnet20_cifar10(exp000())
    cfg.experiment_name = "exp004"
    return cfg


def exp005() -> VisionTrainLoop:
    """exp004 + pre-activation blocks, the branch scaled by a learned scalar from 0.1.

    Hypothesis:
      As ``exp001``, at ResNet-20 depth on CIFAR-10.

    References:
      https://arxiv.org/abs/1603.05027
        He et al. 2016, "Identity Mappings in Deep Residual Networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``pre_activation`` of ``dl-residual-connection`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"

    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.block = PreActivationBlock.Config()
    return cfg


def exp006() -> VisionTrainLoop:
    """exp004 + a learned scalar gate on each block's branch, starting at 1.

    Hypothesis:
      As ``exp002``, at ResNet-20 depth on CIFAR-10.

    References:
      https://arxiv.org/abs/2003.04887
        Bachlechner et al. 2021, "ReZero is All You Need: Fast Convergence at
        Large Depth."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``gated_residual`` of ``dl-residual-connection`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp006"

    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.block = GatedResidualBlock.Config()
    return cfg


def exp007() -> VisionTrainLoop:
    """exp004 + stochastic depth, survival decaying linearly to 0.5 at the deepest block.

    As upstream counts them, the three first-stage blocks sit at position 1
    and the rest at 2 to 7.

    Hypothesis:
      As ``exp003``, at ResNet-20 depth on CIFAR-10, where fewer blocks leave
      the implicit ensemble smaller.

    References:
      https://arxiv.org/abs/1603.09382
        Huang et al. 2016, "Deep Networks with Stochastic Depth."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``stochastic_depth`` of ``dl-residual-connection`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp007"

    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.block = StochasticDepthBlock.Config()
    return cfg


def exp008() -> VisionTrainLoop:
    """exp000 at the hidden resnet110-cifar100 setting: ResNet-110, basic blocks.

    Only the depth changes; the dataset and recipe are ``exp000``'s.

    Hypothesis:
      At 110 layers the basic block's post-activation shortcut is where
      gradient flow binds, so this is the bar the other blocks must clear.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Hidden setting ``resnet110-cifar100`` of ``dl-residual-connection`` at
        commit b873944969c53b01e4ee6e843c77767b8c702485.
      https://arxiv.org/abs/1512.03385
        He et al. 2016, "Deep Residual Learning for Image Recognition."

    Results:
      TBD.

    """
    cfg = settings.resnet110_cifar100(exp000())
    cfg.experiment_name = "exp008"
    return cfg


def exp009() -> VisionTrainLoop:
    """exp008 + pre-activation blocks, the branch scaled by a learned scalar from 0.1.

    Hypothesis:
      As ``exp001``; a clean identity shortcut matters most at 110 layers.

    References:
      https://arxiv.org/abs/1603.05027
        He et al. 2016, "Identity Mappings in Deep Residual Networks."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``pre_activation`` of ``dl-residual-connection`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp009"

    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.block = PreActivationBlock.Config()
    return cfg


def exp010() -> VisionTrainLoop:
    """exp008 + a learned scalar gate on each block's branch, starting at 1.

    Hypothesis:
      As ``exp002``, at ResNet-110 depth.

    References:
      https://arxiv.org/abs/2003.04887
        Bachlechner et al. 2021, "ReZero is All You Need: Fast Convergence at
        Large Depth."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``gated_residual`` of ``dl-residual-connection`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp010"

    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.block = GatedResidualBlock.Config()
    return cfg


def exp011() -> VisionTrainLoop:
    """exp008 + stochastic depth, survival decaying linearly to 0.5 at the deepest block.

    As upstream counts them, the eighteen first-stage blocks sit at position 1
    and the rest at 2 to 37.

    Hypothesis:
      As ``exp003``; stochastic depth was proposed for networks this deep.

    References:
      https://arxiv.org/abs/1603.09382
        Huang et al. 2016, "Deep Networks with Stochastic Depth."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``stochastic_depth`` of ``dl-residual-connection`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp011"

    assert isinstance(cfg.step.model, ResNet.Config)
    cfg.step.model.block = StochasticDepthBlock.Config()
    return cfg
