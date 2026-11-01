r"""MLS-Bench ``dl-lr-schedule``: ResNet-56 on CIFAR-100, the schedule as a slot.

``exp000`` is the task's pristine recipe, a constant learning rate. Every later
experiment forks it and changes ``step.schedule``, the one function the task
lets an edit touch.

    exp000  constant
      +-- exp001  cosine annealing
      +-- exp002  5-epoch linear warmup, then cosine
      +-- exp003  one cycle
      +-- exp004  ResNet-20 on CIFAR-10 (the ``resnet20-cifar10`` setting)
      |     +-- exp005  cosine annealing
      |     +-- exp006  5-epoch linear warmup, then cosine
      |     +-- exp007  one cycle
      +-- exp008  MobileNetV2 on Fashion-MNIST (the ``mobilenetv2-fmnist`` setting)
            +-- exp009  cosine annealing
            +-- exp010  5-epoch linear warmup, then cosine
            +-- exp011  one cycle

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.dl.vision.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.mlsbench.dl.lr_schedule.experiments.exp000

The task is pinned to MLS-Bench commit ``b873944969c53b01e4ee6e843c77767b8c702485``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import PartialConfig

from priml.baselines.mlsbench.dl.lr_schedule.schedules import (
    constant,
    cosine,
    one_cycle,
    warmup_cosine,
)
from priml.baselines.mlsbench.dl.vision import experiments, settings


if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


def exp000() -> VisionTrainLoop:
    """ResNet-56 on CIFAR-100, cross-entropy, SGD at a constant learning rate.

    The pristine recipe of the task's ``resnet56-cifar100`` setting. Its script
    is the ``cv-classification-loss`` script with the scheduler removed, so this
    is that task's ``exp000`` with the schedule held at ``lr``. Frozen:
    improvements belong in a fork.

    Hypothesis:
      A constant rate is the bar a schedule must clear to earn its complexity.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Task ``dl-lr-schedule`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = experiments.exp000()
    cfg.study_name = "dl_lr_schedule"
    cfg.experiment_name = "exp000"
    cfg.step.schedule = PartialConfig(constant)
    return cfg


def exp001() -> VisionTrainLoop:
    """exp000 + cosine annealing to zero over the run.

    Hypothesis:
      Large early steps explore and small late steps settle, so annealing the
      rate along a cosine should beat holding it constant.

    References:
      https://arxiv.org/abs/1608.03983
        Loshchilov and Hutter 2017, "SGDR: Stochastic Gradient Descent with
        Warm Restarts."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``cosine`` of ``dl-lr-schedule`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    cfg.step.schedule = PartialConfig(cosine)
    return cfg


def exp002() -> VisionTrainLoop:
    """exp000 + a 5-epoch linear warmup, then cosine annealing.

    Hypothesis:
      Ramping up over the first epochs avoids the unstable start of a large
      rate on a freshly initialized network, then anneals as ``exp001`` does.

    References:
      https://arxiv.org/abs/1706.02677
        Goyal et al. 2017, "Accurate, Large Minibatch SGD: Training ImageNet in
        1 Hour."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``warmup_cosine`` of ``dl-lr-schedule`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    schedule = cfg.step.schedule = PartialConfig(warmup_cosine)
    schedule.warmup_epochs = 5
    return cfg


def exp003() -> VisionTrainLoop:
    """exp000 + one cycle: rise to the peak over 30% of the run, then fall.

    Hypothesis:
      Starting low and rising to the peak lets the network reach a flat
      region before the long anneal, which one-cycle reports as faster
      convergence at the same budget.

    References:
      https://arxiv.org/abs/1708.07120
        Smith and Topin 2019, "Super-Convergence: Very Fast Training of Neural
        Networks Using Large Learning Rates."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``one_cycle`` of ``dl-lr-schedule`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp000()
    cfg.experiment_name = "exp003"
    schedule = cfg.step.schedule = PartialConfig(one_cycle)
    schedule.pct_start = 0.3
    schedule.div_factor = 25.0
    schedule.final_div = 25.0
    return cfg


def exp004() -> VisionTrainLoop:
    """exp000 on the ``resnet20-cifar10`` setting: ResNet-20, constant rate.

    Hypothesis:
      A shallower network on an easier dataset converges sooner, so a
      constant rate is a weaker bar here than on CIFAR-100.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``resnet20-cifar10`` of ``dl-lr-schedule`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.
      https://arxiv.org/abs/1512.03385
        He et al. 2016, "Deep Residual Learning for Image Recognition."

    Results:
      TBD.

    """
    cfg = settings.resnet20_cifar10(exp000())
    cfg.experiment_name = "exp004"
    return cfg


def exp005() -> VisionTrainLoop:
    """exp004 + cosine annealing to zero over the run.

    Hypothesis:
      As ``exp001``: annealing the rate should beat holding it constant.

    References:
      https://arxiv.org/abs/1608.03983
        Loshchilov and Hutter 2017, "SGDR: Stochastic Gradient Descent with
        Warm Restarts."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``cosine`` of ``dl-lr-schedule`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"
    cfg.step.schedule = PartialConfig(cosine)
    return cfg


def exp006() -> VisionTrainLoop:
    """exp004 + a 5-epoch linear warmup, then cosine annealing.

    Hypothesis:
      As ``exp002``: a ramp avoids the unstable start, then anneals.

    References:
      https://arxiv.org/abs/1706.02677
        Goyal et al. 2017, "Accurate, Large Minibatch SGD: Training ImageNet in
        1 Hour."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``warmup_cosine`` of ``dl-lr-schedule`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp006"
    schedule = cfg.step.schedule = PartialConfig(warmup_cosine)
    schedule.warmup_epochs = 5
    return cfg


def exp007() -> VisionTrainLoop:
    """exp004 + one cycle: rise to the peak over 30% of the run, then fall.

    Hypothesis:
      As ``exp003``: rising to the peak first reaches a flat region before
      the long anneal.

    References:
      https://arxiv.org/abs/1708.07120
        Smith and Topin 2019, "Super-Convergence: Very Fast Training of Neural
        Networks Using Large Learning Rates."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``one_cycle`` of ``dl-lr-schedule`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp004()
    cfg.experiment_name = "exp007"
    schedule = cfg.step.schedule = PartialConfig(one_cycle)
    schedule.pct_start = 0.3
    schedule.div_factor = 25.0
    schedule.final_div = 25.0
    return cfg


def exp008() -> VisionTrainLoop:
    """exp000 on the ``mobilenetv2-fmnist`` setting: MobileNetV2, constant rate.

    The task's hidden setting.

    Hypothesis:
      A constant rate is the bar on a mobile network and grayscale data too.

    References:
      https://github.com/Imbernoulli/MLS-Bench
        Setting ``mobilenetv2-fmnist`` of ``dl-lr-schedule`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.
      https://arxiv.org/abs/1801.04381
        Sandler et al. 2018, "MobileNetV2: Inverted Residuals and Linear
        Bottlenecks."

    Results:
      TBD.

    """
    cfg = settings.mobilenetv2_fmnist(exp000())
    cfg.experiment_name = "exp008"
    return cfg


def exp009() -> VisionTrainLoop:
    """exp008 + cosine annealing to zero over the run.

    Hypothesis:
      As ``exp001``: annealing the rate should beat holding it constant.

    References:
      https://arxiv.org/abs/1608.03983
        Loshchilov and Hutter 2017, "SGDR: Stochastic Gradient Descent with
        Warm Restarts."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``cosine`` of ``dl-lr-schedule`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp009"
    cfg.step.schedule = PartialConfig(cosine)
    return cfg


def exp010() -> VisionTrainLoop:
    """exp008 + a 5-epoch linear warmup, then cosine annealing.

    Hypothesis:
      MobileNetV2's depthwise layers and BatchNorm are sensitive to a large
      first step, which a ramp should tame.

    References:
      https://arxiv.org/abs/1706.02677
        Goyal et al. 2017, "Accurate, Large Minibatch SGD: Training ImageNet in
        1 Hour."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``warmup_cosine`` of ``dl-lr-schedule`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp010"
    schedule = cfg.step.schedule = PartialConfig(warmup_cosine)
    schedule.warmup_epochs = 5
    return cfg


def exp011() -> VisionTrainLoop:
    """exp008 + one cycle: rise to the peak over 30% of the run, then fall.

    Hypothesis:
      As ``exp003``: rising to the peak first reaches a flat region before
      the long anneal.

    References:
      https://arxiv.org/abs/1708.07120
        Smith and Topin 2019, "Super-Convergence: Very Fast Training of Neural
        Networks Using Large Learning Rates."
      https://github.com/Imbernoulli/MLS-Bench
        Baseline ``one_cycle`` of ``dl-lr-schedule`` at commit
        b873944969c53b01e4ee6e843c77767b8c702485.

    Results:
      TBD.

    """
    cfg = exp008()
    cfg.experiment_name = "exp011"
    schedule = cfg.step.schedule = PartialConfig(one_cycle)
    schedule.pct_start = 0.3
    schedule.div_factor = 25.0
    schedule.final_div = 25.0
    return cfg
